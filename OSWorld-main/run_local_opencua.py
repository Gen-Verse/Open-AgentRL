"""
    This is the script to run OpenCUA agents on OSWorld tasks using AWS provider.

    You should first host the OpenCUA model on your local machine or a server.

    Command for OpenCUA-72B:
    ```
        python run_multienv_opencua.py \
            --headless \
            --observation_type screenshot \
            --model OpenCUA-72B \
            --result_dir ./results\
            --test_all_meta_path evaluation_examples/test_nogdrive.json \
            --max_steps 100 \
            --num_envs 30  \
            --coordinate_type qwen25 

    ```


    Command for OpenCUA-7B and OpenCUA-32B:
    ```
        python run_multienv_opencua.py \
            --headless \
            --observation_type screenshot \
            --model OpenCUA-32B \
            --result_dir ./results\
            --test_all_meta_path evaluation_examples/test_nogdrive.json \
            --max_steps 100 \
            --num_envs 30  \
            --coordinate_type qwen25 \
            --use_old_sys_prompt

    ```

"""

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

import uuid  # ← NEW

# Global variables for signal handling
active_environments = []
processes = []
is_terminating = False

# load the environment variables from .env file
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()

#  Logger Configs 
def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark"
    )

    # environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless machine"
    )
    parser.add_argument(
        "--action_space", type=str, default="pyautogui", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
        help="Observation type",
    )
    parser.add_argument("--sleep_after_execution", type=float, default=5.0)
    parser.add_argument("--max_steps", type=int, default=100)
    
    # evaluation config
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples"
    )

    # lm config
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--stop_token", type=str, default=None)

    # OpenCUAagent config
    parser.add_argument("--cot_level", type=str, default="l2", help="CoT version: l1, l2, l3. Default is l2 includes 'thought' and 'action'")
    parser.add_argument("--history_type", type=str, default="action_history", help="Use action to represent history steps", choices=["action_history", "thought_history", "observation_history"])
    parser.add_argument("--coordinate_type", type=str, default="qwen25", help="Type of coordinate: Qwen2-VL or Kimi-VL based models use 'relative'; Qwen2.5-VL based models use 'qwen25'", choices=["relative", "qwen25"])
    parser.add_argument("--max_image_history_length", type=int, default=3, help="The max number of images in the history.")
    parser.add_argument("--max_reward_image_history_length", type=int, default=2, help="The max number of images in the history for reward model.")
    parser.add_argument("--use_old_sys_prompt", action="store_true", help="Use the old system prompt for OpenCUA-7B and OpenCUA-32B")
    
    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--example", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path", type=str, default="evaluation_examples/test_nogdrive.json"
    )

    # logging related
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to run in parallel")  
    parser.add_argument("--log_level", type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                       default='INFO', help="Set the logging level")
    # aws config
    parser.add_argument(
        "--region", type=str, default="us-east-1", help="AWS region for the VM"
    )
    parser.add_argument(
        "--provider_name", type=str, default="volcengine", choices=["volcengine"], help="Provider name"
    )
    parser.add_argument(
        "--client_password", type=str, default="", help="Client password"
    )
    parser.add_argument(
        "--screen_width", type=int, default=1920, help="Screen width"
    )
    parser.add_argument(
        "--screen_height", type=int, default=1080, help="Screen height"
    )
    parser.add_argument(
        "--password", type=str, default="osworld-public-evaluation", help="The password for the computer if needed"
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-run (overwrite) existing results for the selected tasks instead of skipping."
    )
    args = parser.parse_args()
    return args

args = config()  # Get command line arguments first

logger = logging.getLogger()
log_level = getattr(logging, args.log_level.upper())
logger.setLevel(log_level)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

file_handler = logging.FileHandler(
    os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
os.makedirs("logs", exist_ok=True)
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

# === Session registry: only track instances created by THIS script ===
SESSION_ID = f"osworld-{datetime_str}-{uuid.uuid4().hex[:8]}"
REGISTRY_DIR = os.path.join("logs", "session_registry")
os.makedirs(REGISTRY_DIR, exist_ok=True)
REGISTRY_FILE = os.path.join(REGISTRY_DIR, f"{SESSION_ID}.jsonl")

def _append_registry(event: str, **fields):
    """Write one event line into session registry."""
    record = {"event": event, "time": time.time(), "session": SESSION_ID, **fields}
    try:
        with open(REGISTRY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write registry: {e}")

def _get_instance_id(env) -> str | None:
    """Best-effort probe the instance id from DesktopEnv."""
    for attr in ("instance_id", "id"):
        v = getattr(env, attr, None)
        if v:
            return v
    ctrl = getattr(env, "controller", None)
    if ctrl is not None:
        for attr in ("instance_id", "id", "vm_id", "instanceId"):
            v = getattr(ctrl, attr, None)
            if v:
                return v
    try:
        return env.get_instance_id()  # 如果类实现了这个方法
    except Exception:
        return None

def _delete_registry_file():
    """Delete session registry file at the very end."""
    try:
        if os.path.exists(REGISTRY_FILE):
            os.remove(REGISTRY_FILE)
            logger.info(f"Session registry removed: {REGISTRY_FILE}")
    except Exception as e:
        logger.warning(f"Failed to remove session registry: {e}")


def distribute_tasks(test_all_meta: dict) -> List[tuple]:
    all_tasks = []
    for domain, examples in test_all_meta.items():
        for example_id in examples:
            all_tasks.append((domain, example_id))
    return all_tasks



def run_env_tasks(task_queue: Queue, args: argparse.Namespace, shared_scores: list, env_idx: int):
    active_environments = []
    env = None

    def _child_signal_handler(signum, frame):
        # 子进程优雅清理：确保 env.close() 被调用，并在注册簿里记“closed”
        logger.info(f"[Child {env_idx+1}] received signal {signum}. Closing envs...")
        for e in active_environments:
            if e is not None:
                try:
                    iid = _get_instance_id(e)
                    e.close()
                    if iid:
                        _append_registry("closed", pid=os.getpid(), instance_id=iid, provider=args.provider_name, region=args.region)
                except Exception as ce:
                    logger.error(f"[Child {env_idx+1}] error closing env: {ce}")
        logger.info(f"[Child {env_idx+1}] exit after cleanup.")
        sys.exit(0)

    # 子进程自己处理 Ctrl+C / SIGTERM
    signal.signal(signal.SIGINT, _child_signal_handler)
    signal.signal(signal.SIGTERM, _child_signal_handler)

    try:
        REGION = args.region
        screen_size = (args.screen_width, args.screen_height)

        os.environ['HTTP_PROXY'] = ''
        os.environ['HTTPS_PROXY'] = ''

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
            client_password=args.client_password
        )
        active_environments.append(env)

        # 记录“创建了一个实例”
        iid = _get_instance_id(env)
        if iid:
            _append_registry("created", pid=os.getpid(), instance_id=iid, provider=args.provider_name, region=REGION)

        logger.info(f"Process {current_process().name} started.")

        while True:
            try:
                item = task_queue.get(timeout=5)
            except Exception:
                break
            domain, example_id = item
            try:
                config_file = os.path.join(
                    args.test_config_base_dir, f"examples/{domain}/{example_id}.json"
                )
                with open(config_file, "r", encoding="utf-8") as f:
                    example = json.load(f)

                logger.info(f"[{current_process().name}][Domain]: {domain}")
                logger.info(f"[{current_process().name}][Example ID]: {example_id}")
                logger.info(f"[{current_process().name}][Instruction]: {example['instruction']}")

                model_tag = re.sub(r'[\\/]+', '_', args.model).lstrip('_')
                example_result_dir = os.path.join(
                    args.result_dir,
                    args.action_space,
                    args.observation_type,
                    model_tag,
                    domain,
                    example_id,
                )
                os.makedirs(example_result_dir, exist_ok=True)

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
                    example_result_dir = example_result_dir,
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
                        example_result_dir,
                        shared_scores,
                    )
                except Exception as e:
                    import traceback
                    logger.error(f"Exception in {current_process().name} {domain}/{example_id}: {e}")
                    logger.error(traceback.format_exc())
                    try:
                        env.controller.end_recording(
                            os.path.join(example_result_dir, "recording.mp4")
                        )
                    except Exception as rec_e:
                        logger.error(f"Failed to end recording: {rec_e}")
                    with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                        f.write(json.dumps({"Error": f"{domain}/{example_id} - {e}"}) + "\n")
            except Exception as e:
                logger.error(f"Task-level error in {current_process().name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Process-level error in {current_process().name}: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        logger.info(f"{current_process().name} cleaning up environment...")
        try:
            if env:
                iid = _get_instance_id(env)
                env.close()
                if iid:
                    _append_registry("closed", pid=os.getpid(), instance_id=iid, provider=args.provider_name, region=args.region)
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

    # 关闭主进程自己持有的 env（通常为空，防御性）
    for env in active_environments:
        try:
            logger.info("Closing environment...")
            env.close()
            logger.info("Environment closed successfully")
        except Exception as e:
            logger.error(f"Error closing environment: {e}")

    # ① 给子进程发 SIGINT（让它们跑各自的 _child_signal_handler → env.close）
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Sending SIGINT to {p.name} (pid={p.pid})...")
                os.kill(p.pid, signal.SIGINT)
            except Exception as e:
                logger.error(f"Error sending SIGINT to {p.name}: {e}")

    # ② 等待它们自己清理
    deadline = time.time() + 15  # 最多等 15 秒
    for p in processes:
        remaining = max(0, deadline - time.time())
        if p.is_alive():
            p.join(timeout=remaining)

    # ③ 仍未退出的，温和 terminate（某些平台相当于 SIGTERM）
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"{p.name} still alive; calling terminate()...")
                p.terminate()
            except Exception as e:
                logger.error(f"Error terminating {p.name}: {e}")

    time.sleep(2)

    # ④ 还活着的，最后 SIGKILL
    for p in processes:
        if p.is_alive():
            try:
                logger.info(f"Forcefully killing {p.name} (pid={p.pid})...")
                os.kill(p.pid, signal.SIGKILL)
            except Exception as e:
                logger.error(f"Error SIGKILL {p.name}: {e}")

    # 无论如何，删除 session registry 文件
    _delete_registry_file()

    logger.info("Shutdown complete. Exiting.")
    sys.exit(0)



def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    global processes
    logger.info("Args: %s", args)
    all_tasks = distribute_tasks(test_all_meta)
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
                name=f"EnvProcess-{i+1}"
            )
            #p.daemon = True
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
                            name=f"EnvProcess-Restart-{idx+1}"
                        )
                        #new_p.daemon = True
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
    action_space, use_model, observation_type, result_dir, total_file_json, rerun_if_exists=False
):
    model_tag = re.sub(r'[\\/]+', '_', use_model).lstrip('_')
    target_dir = os.path.join(result_dir, action_space, observation_type, model_tag)

    # 覆盖模式：只清理“待评估列表”里的样本目录，其它不动
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
        # 返回原列表，不做过滤（全部重跑）
        return total_file_json

    # 非覆盖模式：保持原先“已完成就跳过”的逻辑
    if not os.path.exists(target_dir):
        logger.info(f"No previous results.")
        return total_file_json
    else:
        logger.info(f"Previous results: {target_dir}")

    finished = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                if example_id == "onboard":
                    continue
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" not in os.listdir(example_path):
                        # 清空未完整结束的残留
                        for file in os.listdir(example_path):
                            try:
                                os.remove(os.path.join(example_path, file))
                            except Exception:
                                pass
                    else:
                        finished[domain].append(example_id)

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [x for x in total_file_json[domain] if x not in examples]

    return total_file_json


def get_result(action_space, use_model, observation_type, result_dir, total_file_json):
    model_tag = re.sub(r'[\\/]+', '_', use_model).lstrip('_')
    target_dir = os.path.join(result_dir, action_space, observation_type, model_tag)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return None

    all_result = []

    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" in os.listdir(example_path):
                        # empty all files under example_id
                        try:
                            all_result.append(
                                float(
                                    open(
                                        os.path.join(example_path, "result.txt"), "r"
                                    ).read()
                                )
                            )
                        except:
                            all_result.append(0.0)

    if not all_result:
        print("New experiment, no result yet.")
        return None
    else:
        print("Current Success Rate:", sum(all_result) / len(all_result) * 100, "%")
        return all_result


if __name__ == "__main__":
    ####### The complete version of the list of examples #######
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Register signal handlers for graceful termination
    signal.signal(signal.SIGINT, signal_handler)  # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Handle termination signal
    
    try:
        args = config()
        
        # save args to json in result_dir/action_space/observation_type/model/args.json
        model_tag = re.sub(r'[\\/]+', '_', args.model).lstrip('_')
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

        test_file_list = get_unfinished(
            args.action_space,
            args.model,
            args.observation_type,
            args.result_dir,
            test_all_meta,
            args.rerun,
        )
        left_info = ""
        for domain in test_file_list:
            left_info += f"{domain}: {len(test_file_list[domain])}\n"
        logger.info(f"Left tasks:\n{left_info}")

        get_result(
            args.action_space,
            args.model,
            args.observation_type,
            args.result_dir,
            test_all_meta,
        )
        test(args, test_file_list)
    except KeyboardInterrupt:
        logger.info("Main process received KeyboardInterrupt.")
        # Signal handler will take care of cleanup
    except Exception as e:
        logger.error(f"Unexpected error in main process: {e}", exc_info=True)
        # Also trigger cleanup for unhandled exceptions
        signal_handler(signal.SIGTERM, None)
    finally:
        # Final cleanup in case any environments or processes remain
        logger.info("Main process final cleanup...")
        for env in active_environments:
            if env is not None:
                try:
                    logger.info(f"Closing environment in final cleanup...")
                    env.close()
                    logger.info(f"Environment closed successfully in final cleanup")
                except Exception as e:
                    logger.error(f"Error during final environment cleanup: {e}")
        
        # First try gentle termination
        for p in processes:
            if p is not None and p.is_alive():
                try:
                    logger.info(f"Terminating process {p.name}...")
                    p.terminate()
                except Exception as e:
                    logger.error(f"Error terminating process: {e}")
        
        # Wait a moment for processes to terminate
        time.sleep(1)
        
        # Then force kill if needed
        for p in processes:
            if p is not None and p.is_alive():
                try:
                    logger.info(f"Force killing process {p.name}...")
                    os.kill(p.pid, signal.SIGKILL)
                    logger.info(f"Process {p.name} force killed")
                except Exception as e:
                    logger.error(f"Error force killing process: {e}")
        
        # Remove session registry file at the very end
        _delete_registry_file()
