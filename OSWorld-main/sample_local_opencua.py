from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import re
import sys
import shutil
import signal
import time
from typing import List
from multiprocessing import Process, Manager
from multiprocessing import current_process, Queue

import lib_run_single
from desktop_env.desktop_env import DesktopEnv
from mm_agents.opencua import OpenCUAAgentLocal

# Global variables for signal handling
active_environments = []
processes = []
is_terminating = False

# load the environment variables from .env file
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )

    # environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--headless", action="store_true", help="Run in headless machine")
    parser.add_argument("--action_space", type=str, default="pyautogui", help="Action type")
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
        help="Observation type",
    )
    parser.add_argument("--sleep_after_execution", type=float, default=5.0)
    parser.add_argument("--max_steps", type=int, default=100)

    # evaluation config
    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples")

    # lm config
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--stop_token", type=str, default=None)

    # OpenCUAagent config
    parser.add_argument("--cot_level", type=str, default="l2",
                        help="CoT version: l1, l2, l3. Default is l2 includes 'thought' and 'action'")
    parser.add_argument("--history_type", type=str, default="action_history",
                        help="Use action to represent history steps",
                        choices=["action_history", "thought_history", "observation_history"])
    parser.add_argument("--coordinate_type", type=str, default="qwen25",
                        help="Type of coordinate: Qwen2-VL or Kimi-VL based models use 'relative'; "
                             "Qwen2.5-VL based models use 'qwen25'",
                        choices=["relative", "qwen25"])
    parser.add_argument("--max_image_history_length", type=int, default=3,
                        help="The max number of images in the history.")
    parser.add_argument("--max_reward_image_history_length", type=int, default=2,
                        help="The max number of images in the history for reward model.")
    parser.add_argument("--use_old_sys_prompt", action="store_true",
                        help="Use the old system prompt for OpenCUA-7B and OpenCUA-32B")

    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--example", type=str, default="all")
    parser.add_argument("--test_all_meta_path", type=str, default="evaluation_examples/test_nogdrive.json")

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel")
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    # provider config
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region for the VM")
    parser.add_argument("--provider_name", type=str, default="volcengine", choices=["volcengine"],
                        help="Provider name")
    parser.add_argument("--client_password", type=str, default="", help="Client password")
    parser.add_argument("--screen_width", type=int, default=1920, help="Screen width")
    parser.add_argument("--screen_height", type=int, default=1080, help="Screen height")
    parser.add_argument("--password", type=str, default="osworld-public-evaluation",
                        help="The password for the computer if needed")

    # rerun / sharding / sampling
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-run (overwrite) existing results for the selected tasks instead of skipping.",
    )
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_index", type=int, default=0, help="Node index")
    parser.add_argument("--num_rollout_per_trial", type=int, default=1,
                        help="Run each example independently this many times")
    parser.add_argument("--project", type=str, default=None)

    args = parser.parse_args()
    args.sample_k = max(1, getattr(args, "num_rollout_per_trial", 1))
    return args


# parse args once for logger config (same pattern as your qwen3vl sample)
args = config()

# -----------------------------
# Logger configs (ALIGNED)
# -----------------------------
logger = logging.getLogger()
log_level = getattr(logging, args.log_level.upper())
logger.setLevel(log_level)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
os.makedirs("logs", exist_ok=True)  # <-- MUST: create logs dir before FileHandler

file_handler = logging.FileHandler(os.path.join("logs", f"normal-{datetime_str}.log"), encoding="utf-8")
debug_handler = logging.FileHandler(os.path.join("logs", f"debug-{datetime_str}.log"), encoding="utf-8")
stdout_handler = logging.StreamHandler(sys.stdout)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(log_level)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
stdout_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)

logger = logging.getLogger("desktopenv.experiment")

# set project env (safe, aligned)
if args.project is not None:
    os.environ["OSWORLD_PROJECT"] = str(args.project)


def _get_instance_id(env) -> str | None:
    # aligned with qwen3vl sample (defensive)
    return getattr(env, "path_to_vm", None)


def distribute_tasks(test_all_meta: dict, args: argparse.Namespace) -> List[tuple]:

    tasks = []
    for domain, examples in test_all_meta.items():
        for example_id in examples:
            for k in range(args.sample_k):
                tasks.append((domain, example_id, k))
    return tasks


def run_env_tasks(task_queue: Queue, args: argparse.Namespace, shared_scores: list, env_idx: int):
    active_environments = []
    env = None

    def _child_signal_handler(signum, frame):
        logger.info(f"[Child {env_idx+1}] received signal {signum}. Closing envs...")
        for e in active_environments:
            if e is not None:
                try:
                    _ = _get_instance_id(e)
                    e.close()
                except Exception as ce:
                    logger.error(f"[Child {env_idx+1}] error closing env: {ce}")
        logger.info(f"[Child {env_idx+1}] exit after cleanup.")
        sys.exit(0)

    # child handles Ctrl+C / SIGTERM
    signal.signal(signal.SIGINT, _child_signal_handler)
    signal.signal(signal.SIGTERM, _child_signal_handler)

    try:
        REGION = args.region
        screen_size = (args.screen_width, args.screen_height)

        env = DesktopEnv(
            path_to_vm=args.path_to_vm,
            action_space=args.action_space,
            provider_name=args.provider_name,
            region=REGION,
            screen_size=screen_size,
            headless=args.headless,
            os_type="Ubuntu",
            require_a11y_tree=args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
            enable_proxy=True,
            client_password=args.client_password,
        )
        active_environments.append(env)

        logger.info(f"Process {current_process().name} started.")

        while True:
            try:
                item = task_queue.get(timeout=5)
            except Exception:
                break

            domain, example_id, run_idx = item
            try:
                config_file = os.path.join(args.test_config_base_dir, f"examples/{domain}/{example_id}.json")
                with open(config_file, "r", encoding="utf-8") as f:
                    example = json.load(f)

                logger.info(f"[{current_process().name}][Domain]: {domain}")
                logger.info(f"[{current_process().name}][Example ID]: {example_id}[k={run_idx}]")
                logger.info(f"[{current_process().name}][Instruction]: {example['instruction']}")

                model_tag = re.sub(r"[\\/]+", "_", args.model).lstrip("_")
                example_root_dir = os.path.join(
                    args.result_dir,
                    args.action_space,
                    args.observation_type,
                    model_tag,
                    domain,
                    example_id,
                )
                run_dir = os.path.join(example_root_dir, str(run_idx))
                os.makedirs(run_dir, exist_ok=True)

                # aligned pattern: each task creates a new agent instance (avoid state leakage)
                agent = OpenCUAAgentLocal(
                    env=env,
                    model=args.model,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    action_space=args.action_space,
                    observation_type=args.observation_type,
                    cot_level=args.cot_level,
                    history_type=args.history_type,
                    screen_size=(args.screen_width, args.screen_height),
                    coordinate_type=args.coordinate_type,
                    max_image_history_length=args.max_image_history_length,
                    max_reward_image_history_length=args.max_reward_image_history_length,
                    max_steps=args.max_steps,
                    use_old_sys_prompt=args.use_old_sys_prompt,
                    example_result_dir=run_dir,
                    password=args.password,
                )

                try:
                    lib_run_single.run_single_example_opencua(
                        agent,
                        env,
                        example,
                        args.max_steps,
                        example["instruction"],
                        args,
                        run_dir,
                        shared_scores,
                    )
                except Exception as e:
                    import traceback

                    logger.error(
                        f"Exception in {current_process().name} {domain}/{example_id}[k={run_idx}]: {e}"
                    )
                    logger.error(traceback.format_exc())
                    try:
                        env.controller.end_recording(os.path.join(run_dir, "recording.mp4"))
                    except Exception as rec_e:
                        logger.error(f"Failed to end recording: {rec_e}")
                    with open(os.path.join(run_dir, "traj.jsonl"), "a") as f:
                        f.write(json.dumps({"Error": f"{domain}/{example_id}[k={run_idx}] - {e}"}) + "\n")

            except Exception as e:
                import traceback
                logger.error(f"Task-level error in {current_process().name}: {e}")
                logger.error(traceback.format_exc())

    except Exception as e:
        import traceback
        logger.error(f"Process-level error in {current_process().name}: {e}")
        logger.error(traceback.format_exc())

    finally:
        logger.info(f"{current_process().name} cleaning up environment...")
        try:
            if env:
                _ = _get_instance_id(env)
                env.close()
                logger.info(f"{current_process().name} environment closed successfully")
        except Exception as e:
            logger.error(f"{current_process().name} error during environment cleanup: {e}")


def signal_handler(signum, frame):
    """Handle termination signals (SIGINT, SIGTERM) to gracefully shutdown environments."""
    global is_terminating, active_environments, processes

    if is_terminating:
        return
    is_terminating = True
    logger.info(f"Received signal {signum}. Gracefully shutting down...")

    # close envs held by main (usually empty, defensive)
    for env in active_environments:
        try:
            logger.info("Closing environment...")
            env.close()
            logger.info("Environment closed successfully")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")

    # ① send SIGINT to children
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Sending SIGINT to {p.name} (pid={p.pid})...")
                os.kill(p.pid, signal.SIGINT)
            except Exception as e:
                logger.error(f"Error sending SIGINT to {p.name}: {e}")

    # ② wait for them to cleanup
    deadline = time.time() + 15
    for p in processes:
        remaining = max(0, deadline - time.time())
        if p.is_alive():
            p.join(timeout=remaining)

    # ③ terminate remaining
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"{p.name} still alive; calling terminate()...")
                p.terminate()
            except Exception as e:
                logger.error(f"Error terminating {p.name}: {e}")

    time.sleep(2)

    # ④ SIGKILL last resort
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Forcefully killing {p.name} (pid={p.pid})...")
                os.kill(p.pid, signal.SIGKILL)
            except Exception as e:
                logger.error(f"Error SIGKILL {p.name}: {e}")

    logger.info("Shutdown complete. Exiting.")
    sys.exit(0)


def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    global processes
    logger.info("Args: %s", args)

    all_tasks = distribute_tasks(test_all_meta, args)
    logger.info(f"Total tasks: {len(all_tasks)}")

    with Manager() as manager:
        shared_scores = manager.list()
        task_queue = manager.Queue()
        for item in all_tasks:
            task_queue.put(item)

        num_envs = args.num_envs
        processes = []

        for i in range(num_envs):
            p = Process(
                target=run_env_tasks,
                args=(task_queue, args, shared_scores, i),
                name=f"EnvProcess-{i+1}",
            )
            # aligned: do NOT set daemon
            p.start()
            processes.append(p)
            logger.info(f"Started process {p.name} with PID {p.pid}")

        try:
            while True:
                alive_count = 0
                for idx, p in enumerate(processes):
                    if not p.is_alive():
                        logger.warning(f"Process {p.name} died, restarting...")
                        new_p = Process(
                            target=run_env_tasks,
                            args=(task_queue, args, shared_scores, idx),
                            name=f"EnvProcess-Restart-{idx+1}",
                        )
                        new_p.start()
                        processes[idx] = new_p
                        logger.info(f"Restarted process {new_p.name} with PID {new_p.pid}")
                    else:
                        alive_count += 1

                if task_queue.empty():
                    logger.info("All tasks finished.")
                    break
                if alive_count == 0:
                    logger.error("All processes died, exiting.")
                    break
                time.sleep(5)

            for p in processes:
                p.join()

        except KeyboardInterrupt:
            logger.info("Main process received KeyboardInterrupt. Initiating graceful shutdown...")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while waiting for processes: {e}", exc_info=True)
            for p in processes:
                if p.is_alive():
                    try:
                        logger.info(f"Terminating process {p.name} due to error...")
                        p.terminate()
                    except Exception as term_e:
                        logger.error(f"Error terminating process {p.name}: {term_e}")
            raise

        scores = list(shared_scores)

    logger.info(f"Average score: {sum(scores) / len(scores) if scores else 0}")


def get_unfinished(
    action_space,
    use_model,
    observation_type,
    result_dir,
    total_file_json,
    rerun_if_exists=False,
    sample_k: int = 1,
):
    model_tag = re.sub(r"[\\/]+", "_", use_model).lstrip("_")
    target_dir = os.path.join(result_dir, action_space, observation_type, model_tag)

    if rerun_if_exists:
        logger.info("Rerun is ON: existing results for selected tasks will be overwritten.")
        for domain, examples in (total_file_json or {}).items():
            for example_id in (examples or []):
                if example_id == "onboard":
                    continue
                example_path = os.path.join(target_dir, domain, example_id)
                try:
                    if os.path.isdir(example_path):
                        shutil.rmtree(example_path)
                    os.makedirs(example_path, exist_ok=True)
                except Exception as e:
                    logger.warning(f"Failed to reset example dir {example_path}: {e}")
        return total_file_json

    if not os.path.exists(target_dir):
        logger.info("No previous results.")
        return total_file_json
    else:
        logger.info(f"Previous results: {target_dir}")

    finished = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if not os.path.isdir(domain_path):
            continue

        for example_id in os.listdir(domain_path):
            if example_id == "onboard":
                continue
            example_path = os.path.join(domain_path, example_id)
            if not os.path.isdir(example_path):
                continue

            done_k = 0
            if os.path.exists(os.path.join(example_path, "result.txt")):
                done_k = max(done_k, 1)

            try:
                for sub in os.listdir(example_path):
                    sub_path = os.path.join(example_path, sub)
                    if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "result.txt")):
                        done_k += 1
            except Exception:
                pass

            if done_k >= sample_k:
                finished[domain].append(example_id)

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [x for x in total_file_json[domain] if x not in examples]

    return total_file_json


def get_result(action_space, use_model, observation_type, result_dir, total_file_json):
    model_tag = re.sub(r"[\\/]+", "_", use_model).lstrip("_")
    target_dir = os.path.join(result_dir, action_space, observation_type, model_tag)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []
    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if not os.path.isdir(domain_path):
            continue

        for example_id in os.listdir(domain_path):
            example_path = os.path.join(domain_path, example_id)
            if not os.path.isdir(example_path):
                continue

            root_result = os.path.join(example_path, "result.txt")
            if os.path.exists(root_result):
                try:
                    all_result.append(float(open(root_result, "r").read()))
                except Exception:
                    all_result.append(0.0)

            try:
                for sub in os.listdir(example_path):
                    sub_path = os.path.join(example_path, sub)
                    if os.path.isdir(sub_path):
                        result_file = os.path.join(sub_path, "result.txt")
                        if os.path.exists(result_file):
                            try:
                                all_result.append(float(open(result_file, "r").read()))
                            except Exception:
                                all_result.append(0.0)
            except Exception:
                pass

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print("Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
        return all_result


def get_data_chunk(data, num_nodes, node_idx):
    total = len(data)
    start = (total * node_idx) // num_nodes
    end = (total * (node_idx + 1)) // num_nodes
    return data[start:end]


def shard_by_file_order(test_all_meta, num_nodes, node_idx):
    pairs = [(d, e) for d, exs in test_all_meta.items() for e in exs]
    chunk = get_data_chunk(pairs, num_nodes, node_idx)
    out = {}
    for d, e in chunk:
        out.setdefault(d, []).append(e)
    return out


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        args = config()
        if args.project is not None:
            os.environ["OSWORLD_PROJECT"] = str(args.project)

        model_tag = re.sub(r"[\\/]+", "_", args.model).lstrip("_")
        path_to_args = os.path.join(
            args.result_dir,
            args.action_space,
            args.observation_type,
            model_tag,
            "args.json",
        )
        os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
        with open(path_to_args, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=4)

        with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
            test_all_meta = json.load(f)

        if args.domain != "all":
            test_all_meta = {args.domain: test_all_meta[args.domain]}

        if args.example != "all":
            test_all_meta = {args.domain: [args.example]}

        if args.num_nodes > 1:
            test_all_meta = shard_by_file_order(test_all_meta, args.num_nodes, args.node_index)

        test_file_list = get_unfinished(
            args.action_space,
            args.model,
            args.observation_type,
            args.result_dir,
            test_all_meta,
            args.rerun,
            args.sample_k,
        )

        left_info = ""
        for domain in test_file_list:
            left_info += f"{domain}: {len(test_file_list[domain])}\n"
        logger.info(f"Left tasks:\n{left_info}")

        get_result(args.action_space, args.model, args.observation_type, args.result_dir, test_all_meta)
        test(args, test_file_list)

    except KeyboardInterrupt:
        logger.info("Main process received KeyboardInterrupt.")
    except Exception as e:
        logger.error(f"Unexpected error in main process: {e}", exc_info=True)
        signal_handler(signal.SIGTERM, None)
    finally:
        logger.info("Main process final cleanup...")

        for env in active_environments:
            if env is not None:
                try:
                    logger.info("Closing environment in final cleanup...")
                    env.close()
                    logger.info("Environment closed successfully in final cleanup")
                except Exception as e:
                    logger.error(f"Error during final environment cleanup: {e}")

        # gentle terminate
        for p in processes:
            if p is not None and p.is_alive():
                try:
                    logger.info(f"Terminating process {p.name}...")
                    p.terminate()
                except Exception as e:
                    logger.error(f"Error terminating process: {e}")

        time.sleep(1)

        # force kill
        for p in processes:
            if p is not None and p.is_alive():
                try:
                    logger.info(f"Force killing process {p.name}...")
                    os.kill(p.pid, signal.SIGKILL)
                    logger.info(f"Process {p.name} force killed")
                except Exception as e:
                    logger.error(f"Error force killing process: {e}")
