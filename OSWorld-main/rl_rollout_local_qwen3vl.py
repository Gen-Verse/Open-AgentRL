# -*- coding: utf-8 -*-
"""
rl_rollout_local_qwen3vl_robust.py

Drop-in robust multiprocess runner for your RL rollout script:
- spawn + ctx.Queue (no Manager.Queue, no queue.empty)
- supervisor tracks inflight (job_id -> pid). If worker dies, requeue inflight jobs (no loss)
- worker catches BaseException
- tasks are full job dict templates, never enqueue partial dict
- keeps ALL algorithm-level logic identical to your original rollout script
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import re
import shutil
import signal
import sys
import time
from typing import List, Dict, Any, Optional, Tuple

import multiprocessing as mp
from queue import Empty

import lib_run_single
from desktop_env.desktop_env import DesktopEnv
from mm_agents.qwen3vl_agent_local import Qwen3VLAgentLocal


# load the environment variables from .env file
if os.path.exists(".env"):
    from dotenv import load_dotenv
    load_dotenv()


# -----------------------------
# Args (kept same + legacy num_node alias)
# -----------------------------
def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the benchmark (Qwen3VL)"
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
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)
    parser.add_argument("--max_steps", type=int, default=15)

    # agent config
    parser.add_argument("--max_image_history_length", type=int, default=3, help="The max number of images in the history.")
    parser.add_argument("--max_reward_image_history_length", type=int, default=1, help="The max number of images in the history for reward model.") # total max_reward_image_history_length + 1 images
    parser.add_argument("--test_config_base_dir", type=str, default="evaluation_examples")

    # lm config
    parser.add_argument("--model", type=str, default="qwen3-vl")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--current_step", type=int, default=1)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--coordinate_type",
        type=str,
        choices=["absolute", "relative", "qwen25"],
        default="relative",
        help="Coordinate system for agent outputs (absolute or relative)",
    )
    parser.add_argument(
        "--add_thought_prefix",
        action="store_true",
        help="Add thought prefix to the response",
    )
    parser.add_argument("--use_old_sys_prompt", action="store_true", help="Use the old system prompt (kept for compatibility)")

    # example config
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument("--example", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path",
        type=str,
        default="evaluation_examples/test_nogdrive.json",
    )

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

    # provider config (默认走 volcengine)
    parser.add_argument("--region", type=str, default="us-east-1", help="Region for the VM")
    parser.add_argument(
        "--provider_name",
        type=str,
        default="volcengine",
        choices=["volcengine"],
        help="Provider name",
    )
    parser.add_argument("--client_password", type=str, default="", help="Client password")
    parser.add_argument("--screen_width", type=int, default=1920, help="Screen width")
    parser.add_argument("--screen_height", type=int, default=1080, help="Screen height")

    # rerun / sharding / sampling
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-run (overwrite) existing results for the selected tasks instead of skipping.",
    )
    parser.add_argument("--num_nodes", type=int, default=1, help="Number of nodes")
    parser.add_argument("--node_index", type=int, default=0, help="Node index")
    parser.add_argument(
        "--num_rollout_per_trial",
        type=int,
        default=1,
        help="Run each example independently this many times",
    )

    # legacy alias (some orchestrators still use --num_node)
    parser.add_argument("--num_node", type=int, default=None, help="(legacy) alias of --num_nodes")

    # ===== RL rollout =====
    parser.add_argument(
        "--rollout_type", type=str, default="train",
        help="rollout type, train or evaluation"
    )
    parser.add_argument("--num_trial", type=int, default=1, help="number of samples (examples) to draw globally")
    parser.add_argument("--save_example_json", type=str, default=None, help="path to save sampled example ids (train only)")

    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--coevolveenv", type=str, default="TRUE")

    # supervisor knobs (infrastructure only)
    parser.add_argument("--max_attempts", type=int, default=0,
                        help="Max attempts per job when worker DIES mid-job. 0 means infinite retry.")
    parser.add_argument("--no_progress_abort_minutes", type=float, default=0.0,
                        help="If >0, abort if no job completes for this many minutes.")
    parser.add_argument("--worker_restart_delay", type=float, default=2.0,
                        help="Seconds to wait before restarting a dead worker.")
    parser.add_argument("--job_watchdog_minutes", type=float, default=0.0,
                        help="If >0, kill & restart a worker if a single job exceeds this many minutes (handles hangs).")

    args = parser.parse_args()
    if args.num_node is not None:
        args.num_nodes = int(args.num_node)

    args.sample_k = max(1, getattr(args, "num_rollout_per_trial", 1))
    return args


# -----------------------------
# Logging (like your robust runner)
# -----------------------------
def setup_logging(args: argparse.Namespace) -> logging.Logger:
    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        try:
            root_logger.removeHandler(h)
        except Exception:
            pass

    level = getattr(logging, args.log_level.upper())
    root_logger.setLevel(level)

    datetime_str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")
    pid = os.getpid()

    log_dir = os.path.join("logs", f"node{int(args.node_index)}")
    os.makedirs(log_dir, exist_ok=True)

    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"normal-{datetime_str}-pid{pid}.log"),
        encoding="utf-8",
    )
    debug_handler = logging.FileHandler(
        os.path.join(log_dir, f"debug-{datetime_str}-pid{pid}.log"),
        encoding="utf-8",
    )
    stdout_handler = logging.StreamHandler(sys.stdout)

    file_handler.setLevel(logging.INFO)
    debug_handler.setLevel(logging.DEBUG)
    stdout_handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s "
            "\x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
    )
    file_handler.setFormatter(formatter)
    debug_handler.setFormatter(formatter)
    stdout_handler.setFormatter(formatter)
    stdout_handler.addFilter(logging.Filter("desktopenv"))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(debug_handler)
    root_logger.addHandler(stdout_handler)

    return logging.getLogger("desktopenv.experiment")


def _get_instance_id(env) -> str | None:
    return getattr(env, "path_to_vm", None)


# -----------------------------
# Helpers (kept identical)
# -----------------------------
def distribute_tasks(test_all_meta: dict, args: argparse.Namespace, kind: str) -> List[tuple]:
    """
    展开成 (kind, domain, example_id, k)
    """
    tasks = []
    for domain, examples in test_all_meta.items():
        for example_id in examples:
            for k in range(args.sample_k):
                tasks.append((kind, domain, example_id, k))
    return tasks


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
        logger = logging.getLogger("desktopenv.experiment")
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

    logger = logging.getLogger("desktopenv.experiment")
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


def load_rollout_manifest(path: str, id2domain: dict[str, str]) -> dict:
    """
    返回 {"active":[{"domain":..,"example_id":..}, ...], "temp":[...]}
    兼容旧格式：如果文件里是 list[str]，当作 active，temp=[]
    """
    if not path or (not os.path.exists(path)):
        return {"active": [], "temp": []}

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, list):
        active_pairs = []
        for eid in obj:
            d = id2domain.get(eid)
            if d is None:
                continue
            active_pairs.append({"domain": d, "example_id": eid})
        return {"active": active_pairs, "temp": []}

    if isinstance(obj, dict):
        obj.setdefault("active", [])
        obj.setdefault("temp", [])
        def _normalize(lst):
            out = []
            for x in lst:
                if isinstance(x, dict) and "domain" in x and "example_id" in x:
                    out.append({"domain": x["domain"], "example_id": x["example_id"]})
                elif isinstance(x, str):
                    d = id2domain.get(x)
                    if d is not None:
                        out.append({"domain": d, "example_id": x})
            return out
        obj["active"] = _normalize(obj["active"])
        obj["temp"] = _normalize(obj["temp"])
        return obj

    return {"active": [], "temp": []}


def save_rollout_manifest(path: str, manifest: dict) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def sample_random_k_excluding(
    test_all_meta: dict[str, list],
    k: int,
    exclude_pairs: set[tuple[str, str]],
) -> dict[str, list]:
    """
    从 test_all_meta 全局随机抽 k 个 (domain, example_id)，但排除 exclude_pairs
    """
    if k <= 0:
        return {}

    pairs = [(d, e) for d, exs in test_all_meta.items() for e in exs]
    pairs = [p for p in pairs if p not in exclude_pairs]
    if not pairs:
        return {}

    chosen = random.sample(pairs, k=min(k, len(pairs)))
    out: dict[str, list] = {}
    for d, e in chosen:
        out.setdefault(d, []).append(e)
    return out


# -----------------------------
# Infrastructure: score sink + job id
# -----------------------------
class ScoreSink:
    """
    A minimal "list-like" sink that supports .append(x).
    run_single_example(agent, env, ..., shared_scores) uses shared_scores.append(score).
    Here we forward that score back to the supervisor via event_queue.
    """
    def __init__(self, event_queue: mp.Queue, pid: int):
        self._q = event_queue
        self._pid = pid

    def append(self, x):
        try:
            self._q.put(("score", self._pid, float(x)))
        except Exception:
            # never crash worker due to scoring
            pass


def make_job_id(kind: str, domain: str, example_id: str, run_idx: int) -> str:
    return f"{kind}/{domain}/{example_id}/k={int(run_idx)}"


# -----------------------------
# Worker (algorithm logic kept identical)
# -----------------------------
def worker_loop(
    task_queue: mp.Queue,
    event_queue: mp.Queue,
    args: argparse.Namespace,
    env_idx: int,
):
    logger = setup_logging(args)
    pid = os.getpid()

    env = None

    def _child_signal_handler(signum, frame):
        logger.warning(f"[Child env={env_idx} pid={pid}] got signal {signum}, closing env then exit.")
        try:
            if env is not None:
                _ = _get_instance_id(env)
                env.close()
        except Exception:
            logger.exception("error during env.close() in child signal handler")
        sys.exit(0)

    signal.signal(signal.SIGINT, _child_signal_handler)
    signal.signal(signal.SIGTERM, _child_signal_handler)

    # Create env (retry forever, same spirit as your robust runner)
    while True:
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
            logger.info(f"[Worker pid={pid} env={env_idx}] DesktopEnv created: vm={_get_instance_id(env)}")
            break
        except BaseException as e:
            logger.exception(f"[Worker pid={pid} env={env_idx}] DesktopEnv init failed; retry in 10s: {e}")
            time.sleep(10)

    score_sink = ScoreSink(event_queue=event_queue, pid=pid)

    try:
        logger.info(f"[Worker pid={pid} env={env_idx}] loop started.")
        while True:
            job = task_queue.get()  # blocking
            if job is None:
                logger.info(f"[Worker pid={pid} env={env_idx}] got sentinel, exit loop.")
                return

            job_id = job.get("job_id", "UNKNOWN")
            event_queue.put(("start", pid, job_id))

            # IMPORTANT: we treat the task as "done" even if run_single throws,
            # matching your original logic (no automatic retry on task errors).
            ok_done = True
            err = ""

            try:
                kind = job["kind"]            # "active" or "temp"
                domain = job["domain"]
                example_id = job["example_id"]
                run_idx = int(job["run_idx"])

                if args.coevolveenv == "TRUE":
                    base = "train" if kind == "active" else "temp"
                    test_sub_dir = f"{args.project}/{base}/{domain}/{example_id}.json"
                else:
                    test_sub_dir = f"examples/{domain}/{example_id}.json"

                config_file = os.path.join(args.test_config_base_dir, test_sub_dir)
                with open(config_file, "r", encoding="utf-8") as f:
                    example = json.load(f)

                if args.coevolveenv == "TRUE":
                    cur = example["current_task"]
                    current_eval_name = next(iter(cur.keys()))
                    eval_list = example["evaluator_list"]
                    example["instruction"] = cur[current_eval_name]
                    import copy
                    example["evaluator"] = copy.deepcopy(eval_list[current_eval_name])

                logger.info(f"[{mp.current_process().name}][Domain]: {domain}")
                logger.info(f"[{mp.current_process().name}][Example ID]: {example_id}[k={run_idx}]")
                logger.info(f"[{mp.current_process().name}][Instruction]: {example.get('instruction', '')}")

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

                agent = Qwen3VLAgentLocal(
                    model=args.model,
                    max_steps=args.max_steps,
                    max_image_history_length=args.max_image_history_length,
                    max_reward_image_history_length=args.max_reward_image_history_length,
                    max_tokens=args.max_tokens,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    action_space=args.action_space,
                    observation_type=args.observation_type,
                    coordinate_type=args.coordinate_type,
                    add_thought_prefix=args.add_thought_prefix,
                    example_result_dir=run_dir,
                )

                try:
                    lib_run_single.run_single_example(
                        agent,
                        env,
                        example,
                        args.max_steps,
                        example["instruction"],
                        args,
                        run_dir,
                        score_sink,  # supports .append(), replaces Manager.list without changing call
                    )
                except Exception as e:
                    import traceback
                    err = repr(e)
                    logger.error(f"Exception in {mp.current_process().name} {domain}/{example_id}[k={run_idx}]: {e}")
                    logger.error(traceback.format_exc())
                    try:
                        env.controller.end_recording(os.path.join(run_dir, "recording.mp4"))
                    except Exception as rec_e:
                        logger.error(f"Failed to end recording: {rec_e}")
                    try:
                        with open(os.path.join(run_dir, "traj.jsonl"), "a", encoding="utf-8") as f:
                            f.write(json.dumps({"Error": f"{domain}/{example_id}[k={run_idx}] - {e}"}) + "\n")
                    except Exception:
                        pass

            except BaseException as e:
                # keep worker alive; mark done (matches original: task errors don't retry)
                ok_done = True
                err = repr(e)
                logger.exception(f"[Worker pid={pid}] Task-level error {job_id}: {e}")

            finally:
                event_queue.put(("end", pid, job_id, bool(ok_done), err))

    finally:
        logger.info(f"[Worker pid={pid} env={env_idx}] finally: closing env...")
        try:
            if env is not None:
                _ = _get_instance_id(env)
                env.close()
                logger.info(f"[Worker pid={pid}] env closed.")
        except Exception as e:
            logger.exception(f"[Worker pid={pid}] error closing env: {e}")


# -----------------------------
# Supervisor runner (robust)
# -----------------------------
def test_robust(args: argparse.Namespace, test_all_meta: dict, temp_all_meta: dict | None = None) -> float:
    logger = logging.getLogger("desktopenv.experiment")
    logger.info("Args: %s", args)

    all_tasks = []
    all_tasks.extend(distribute_tasks(test_all_meta, args, kind="active"))
    if temp_all_meta and args.coevolveenv == "TRUE":
        all_tasks.extend(distribute_tasks(temp_all_meta, args, kind="temp"))
    logger.info(f"Total tasks (active+temp): {len(all_tasks)}")

    if not all_tasks:
        logger.info("No tasks to run.")
        return 0.0

    # Build job templates (full dict)
    job_templates: Dict[str, dict] = {}
    for (kind, domain, example_id, k) in all_tasks:
        jid = make_job_id(kind, domain, example_id, int(k))
        job_templates[jid] = {
            "job_id": jid,
            "kind": kind,
            "domain": domain,
            "example_id": example_id,
            "run_idx": int(k),
        }

    total_jobs = len(job_templates)
    logger.info(f"Total jobs for this node: {total_jobs} (num_nodes={args.num_nodes}, node_index={args.node_index})")

    attempts: Dict[str, int] = {jid: 0 for jid in job_templates.keys()}

    ctx = mp.get_context("spawn")
    task_queue: mp.Queue = ctx.Queue()
    event_queue: mp.Queue = ctx.Queue()

    # initial enqueue
    for jid in job_templates.keys():
        job0 = dict(job_templates[jid])
        job0["_attempt"] = 0
        task_queue.put(job0)

    # Start workers
    num_envs = int(args.num_envs)
    workers: List[mp.Process] = []
    for i in range(num_envs):
        p = ctx.Process(
            target=worker_loop,
            args=(task_queue, event_queue, args, i),
            name=f"EnvProcess-{i+1}",
        )
        p.start()
        workers.append(p)
        logger.info(f"Started {p.name} pid={p.pid}")

    completed: set[str] = set()
    inflight: Dict[str, Tuple[int, float]] = {}  # jid -> (pid, start_ts)
    scores: List[float] = []
    done_count = 0
    last_progress_ts = time.time()
    abort_after_s = float(args.no_progress_abort_minutes) * 60.0 if float(args.no_progress_abort_minutes) > 0 else 0.0
    watchdog_s = float(args.job_watchdog_minutes) * 60.0 if float(args.job_watchdog_minutes) > 0 else 0.0

    def enqueue_job_id(jid: str, reason: str) -> bool:
        attempts[jid] = int(attempts.get(jid, 0)) + 1
        max_attempts = int(args.max_attempts)
        if max_attempts > 0 and attempts[jid] > max_attempts:
            logger.error(f"[GIVEUP] {jid} attempts={attempts[jid]-1} reason={reason}")
            # treat as finished (lost)
            return False
        job_full = dict(job_templates[jid])
        job_full["_attempt"] = attempts[jid] - 1
        logger.warning(f"[ENQUEUE] {jid} attempt={job_full['_attempt']} reason={reason}")
        task_queue.put(job_full)
        return True

    def restart_worker(dead: mp.Process, slot_idx: int):
        logger.warning(f"[RESTART] {dead.name} pid={dead.pid} exitcode={dead.exitcode}")
        time.sleep(float(args.worker_restart_delay))
        new_p = ctx.Process(
            target=worker_loop,
            args=(task_queue, event_queue, args, slot_idx),
            name=f"EnvProcess-Restart-{slot_idx+1}",
        )
        new_p.start()
        workers[slot_idx] = new_p
        logger.info(f"[RESTART] started {new_p.name} pid={new_p.pid}")

    try:
        while done_count < total_jobs:
            # consume events
            try:
                ev = event_queue.get(timeout=5)
            except Empty:
                ev = None

            if ev is not None:
                etype = ev[0]

                if etype == "start":
                    _, pid, jid = ev
                    inflight[jid] = (pid, time.time())

                elif etype == "end":
                    _, pid, jid, _ok_done, err = ev
                    inflight.pop(jid, None)

                    if jid not in completed:
                        completed.add(jid)
                        done_count += 1
                        last_progress_ts = time.time()
                        if err:
                            logger.warning(f"[DONE-WITH-ERR] {jid} err={err}")
                        if done_count % 20 == 0 or done_count == total_jobs:
                            logger.info(f"[PROGRESS] done {done_count}/{total_jobs}")

                elif etype == "score":
                    _, _pid, s = ev
                    try:
                        scores.append(float(s))
                    except Exception:
                        pass

            # watchdog: kill stuck jobs (optional)
            if watchdog_s > 0 and inflight:
                now = time.time()
                stuck = [(jid, owner_pid, st) for jid, (owner_pid, st) in inflight.items() if (now - st) > watchdog_s]
                for jid, owner_pid, st in stuck:
                    logger.error(f"[WATCHDOG] jid={jid} pid={owner_pid} stuck>{watchdog_s:.0f}s, terminating worker to recover.")
                    # find the process slot
                    for idx, p in enumerate(workers):
                        if p is not None and p.is_alive() and p.pid == owner_pid:
                            try:
                                p.terminate()
                            except Exception:
                                pass
                            break
                    # inflight will be requeued when dead-worker detection triggers

            # detect dead workers and requeue their inflight
            for idx, p in enumerate(list(workers)):
                if p is None:
                    continue
                if not p.is_alive():
                    dead_pid = p.pid
                    if dead_pid is not None:
                        lost = [jid for jid, (owner, _) in inflight.items() if owner == dead_pid]
                        for jid in lost:
                            inflight.pop(jid, None)
                            if jid in job_templates and jid not in completed:
                                enqueue_job_id(jid, reason=f"worker_died pid={dead_pid}")
                    restart_worker(p, idx)

            # no-progress abort
            if abort_after_s > 0 and (time.time() - last_progress_ts) > abort_after_s:
                logger.error(f"[ABORT] No progress for {abort_after_s:.0f}s, aborting.")
                break

        # Finish: send sentinel
        logger.info("[FINAL] sending sentinel to workers...")
        for _ in range(len(workers)):
            try:
                task_queue.put(None)
            except Exception:
                pass

        # Join workers
        deadline = time.time() + 120
        for p in workers:
            if p is None:
                continue
            p.join(timeout=max(0, deadline - time.time()))

        # Terminate leftovers
        for p in workers:
            if p is not None and p.is_alive():
                logger.warning(f"[FINAL] {p.name} still alive, terminating...")
                try:
                    p.terminate()
                except Exception:
                    pass

        time.sleep(2)
        for p in workers:
            if p is not None and p.is_alive():
                logger.warning(f"[FINAL] {p.name} still alive, killing...")
                try:
                    os.kill(p.pid, signal.SIGKILL)
                except Exception:
                    pass

    except KeyboardInterrupt:
        logger.warning("Main received KeyboardInterrupt, terminating workers...")
        for _ in range(len(workers)):
            try:
                task_queue.put(None)
            except Exception:
                pass
        for p in workers:
            if p is not None and p.is_alive():
                try:
                    p.terminate()
                except Exception:
                    pass
        raise

    avg_score = (sum(scores) / len(scores)) if scores else 0.0
    logger.info(f"Average score (from ScoreSink): {avg_score}")
    return avg_score


# -----------------------------
# Main (kept identical rollout logic)
# -----------------------------
def main():
    args = config()
    logger = setup_logging(args)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # set project env (safe)
    if args.project is not None:
        os.environ["OSWORLD_PROJECT"] = str(args.project)

    # also keep these lock files pattern (helps multi-node safety in your other runner)
    os.environ["VOLCENGINE_RUNINST_LOCK_FILE"] = f"/tmp/volcengine_runinstances-{int(args.node_index)}.lock"
    os.environ["VOLCENGINE_DELINST_LOCK_FILE"] = f"/tmp/volcengine_deleteinstance-{int(args.node_index)}.lock"

    try:
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

        num_nodes = args.num_nodes
        node_index = args.node_index
        if num_nodes > 1:
            test_all_meta = shard_by_file_order(test_all_meta, num_nodes, node_index)

        active_meta = test_all_meta
        temp_meta = None

        if args.rollout_type == "train":
            if not args.save_example_json:
                raise ValueError("--save_example_json must be provided when --rollout_type=train")

            # 1) id->domain
            id2domain = {}
            for d, exs in test_all_meta.items():
                for e in exs:
                    id2domain[e] = d

            # 2) 
            manifest = load_rollout_manifest(args.save_example_json, id2domain)
            prev_temp_pairs = {(x["domain"], x["example_id"]) for x in manifest.get("temp", [])}

            # 3) sample active：exclude last round temp
            active_meta = sample_random_k_excluding(
                test_all_meta,
                int(args.num_trial / max(1, num_nodes)),
                exclude_pairs=prev_temp_pairs,
            )

            # 4) active go to manifest (replace active；temp keep)
            new_active_list = [{"domain": d, "example_id": e} for d, exs in active_meta.items() for e in exs]
            manifest["active"] = new_active_list

            save_rollout_manifest(args.save_example_json, manifest)
            logger.info(f"Updated rollout manifest to: {args.save_example_json} (active overwritten, temp kept)")
            selected_examples = [x["example_id"] for x in new_active_list]
            logger.info(f"Active sampled examples: {len(selected_examples)}")

        temp_meta = None
        if args.rollout_type == "train" and args.save_example_json and os.path.exists(args.save_example_json):
            id2domain = {e: d for d, exs in test_all_meta.items() for e in exs}
            manifest = load_rollout_manifest(args.save_example_json, id2domain)
            temp_meta = {}
            for x in manifest.get("temp", []):
                temp_meta.setdefault(x["domain"], []).append(x["example_id"])
            left_temp = sum(len(v) for v in temp_meta.values())
            logger.info(f"Loaded temp tasks from manifest: {left_temp}")

        active_file_list = get_unfinished(
            args.action_space, args.model, args.observation_type, args.result_dir,
            active_meta,
            args.rerun,
            args.sample_k,
        )

        temp_file_list = get_unfinished(
            args.action_space, args.model, args.observation_type, args.result_dir,
            temp_meta or {},
            args.rerun,
            args.sample_k,
        )

        get_result(args.action_space, args.model, args.observation_type, args.result_dir, test_all_meta)

        # robust runner
        test_robust(args, active_file_list, temp_file_list)

    except KeyboardInterrupt:
        logger.info("Main process received KeyboardInterrupt.")
    except Exception as e:
        logger.error(f"Unexpected error in main process: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
