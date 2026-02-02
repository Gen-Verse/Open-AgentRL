# -*- coding: utf-8 -*-
import os
import re
import json
import random
import time
import inspect
import faulthandler
import multiprocessing as mp
from queue import Empty
from typing import List, Any, Dict, Optional

from jinja2 import Template
from termcolor import cprint
from omegaconf import OmegaConf


os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# single Engine + TP=8 + chunk_size=1024
TP_SIZE = 8
CHUNK_SIZE = 1024

# Hang debug (seconds)
HANG_DUMP_EVERY_SEC = int(os.environ.get("HANG_DUMP_EVERY_SEC", "300"))

# Parent waiting timeout for each worker generate_results call (seconds)
GEN_TIMEOUT_S = float(os.environ.get("GEN_TIMEOUT_S", "1800"))

# If you *must* guarantee exit even if something hangs at interpreter exit, set FORCE_EXIT=1
FORCE_EXIT = bool(int(os.environ.get("FORCE_EXIT", "0")))


############################
# Config
############################
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


############################
# Paths / State / Pending
############################
def get_state_paths(project_name, dataset, node_index, config):
    state_prefix = OmegaConf.select(config, "dataset.dataset_state_prefix", default="dataset_state")
    pending_prefix = OmegaConf.select(config, "dataset.env_pending_prefix", default="env_pending")
    state_path = f"../{project_name}/temp_data/{state_prefix}-{dataset}-node{node_index}.json"
    pending_path = f"../{project_name}/temp_data/{pending_prefix}-{dataset}-node{node_index}.json"
    return state_path, pending_path


def load_state_or_init_if_missing(base_dataset_path, state_path, num_node, node_index, config):

    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)

    with open(base_dataset_path, "r", encoding="utf-8") as f:
        base = json.load(f)

    total = len(base)
    chunk_size = (total + int(num_node) - 1) // int(num_node)
    start_idx = int(node_index) * chunk_size
    end_idx = min((int(node_index) + 1) * chunk_size, total)
    base_chunk = base[start_idx:end_idx]

    slots = []
    for local_i, item in enumerate(base_chunk):
        slot_id = start_idx + local_i
        v0 = {
            "version": 0,
            "question": item.get("question", ""),
            "test_input": item.get("test_input", []) if isinstance(item.get("test_input", []), list) else [],
            "test_output": item.get("test_output", []) if isinstance(item.get("test_output", []), list) else [],
            "test_method": item.get("test_method", "stdio"),
            "test_time_limit": item.get("test_time_limit", 1),
            "created_step": 0,
            "source": "init",
        }
        slots.append({"slot_id": slot_id, "active": v0, "history": [v0]})

    state = {"meta": {"dataset": os.path.basename(base_dataset_path), "node_index": int(node_index)}, "slots": slots}
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    return state


def save_state(state_path, state):
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def load_pending(pending_path):
    if not os.path.exists(pending_path):
        return []
    with open(pending_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
        return obj if isinstance(obj, list) else []


def save_pending(pending_path, pending_list):
    os.makedirs(os.path.dirname(pending_path), exist_ok=True)
    with open(pending_path, "w", encoding="utf-8") as f:
        json.dump(pending_list, f, indent=2, ensure_ascii=False)


def get_policy_stage_output_path(project_name, outputs_name, num_node, node_index):
    if int(num_node) > 1:
        return f"../{project_name}/temp_data/outputs-{int(node_index)}-{outputs_name}.json"
    return f"../{project_name}/temp_data/outputs-{outputs_name}.json"


############################
# Env JSON parsing (curator output)
############################
def parse_json_obj(text: str):
    t = (text or "").strip()
    t = re.sub(r"```(json)?", "", t)
    t = t.replace("```", "").strip()

    last_ok = None
    in_str = False
    esc = False
    depth = 0
    start = None

    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        cand = t[start:i + 1]
                        try:
                            last_ok = json.loads(cand)
                        except Exception:
                            pass
                        start = None
    return last_ok


############################
# Helpers: per-test accuracy + prompt formatting
############################
def _pass_all(row: Any) -> bool:
    return isinstance(row, list) and len(row) > 0 and all(bool(x) for x in row)


def _per_test_pass_rate(mat: Any) -> List[float]:
    """
    mat: [m_code][m_test] bool
    returns: [m_test] pass_rate across codes
    """
    if not isinstance(mat, list) or len(mat) == 0:
        return []
    max_w = 0
    for row in mat:
        if isinstance(row, list):
            max_w = max(max_w, len(row))
    if max_w == 0:
        return []
    rates = []
    for j in range(max_w):
        ok = 0
        cnt = 0
        for row in mat:
            if not isinstance(row, list) or j >= len(row):
                continue
            cnt += 1
            ok += 1 if bool(row[j]) else 0
        rates.append((ok / cnt) if cnt > 0 else 0.0)
    return rates


def _truncate(s: str, n: int = 800) -> str:
    if not isinstance(s, str):
        s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + "...(truncated)"


def _tests_json_for_prompt(inputs: List[str], outputs: List[str], rates: List[float], max_chars_each: int = 800) -> str:
    if not isinstance(inputs, list):
        inputs = []
    if not isinstance(outputs, list):
        outputs = []
    L = min(len(inputs), len(outputs))
    out = []
    for i in range(L):
        acc = rates[i] if i < len(rates) else 0.0
        out.append(
            {
                "idx": i,
                "pass_rate": round(float(acc), 4),
                "test_input": _truncate(inputs[i], max_chars_each),
                "test_output": _truncate(outputs[i], max_chars_each),
            }
        )
    return json.dumps(out, ensure_ascii=False, indent=2)


def _env_k_sample_from_config(config) -> int:
    k = OmegaConf.select(config, "environment.num_response_per_task", default=None)
    if k is None:
        k = OmegaConf.select(config, "rollout.environment.num_response_per_task", default=1)
    try:
        k = int(k)
    except Exception:
        k = 1
    return max(1, k)


def _filter_ut_pairs_for_prompt(inputs: Any, outputs: Any, max_chars: int):
    """
    Filter out (input, output) pairs if either side exceeds max_chars.
    Return filtered_inputs, filtered_outputs, dropped_count.
    """
    if not isinstance(inputs, list):
        inputs = []
    if not isinstance(outputs, list):
        outputs = []

    L = min(len(inputs), len(outputs))
    fin, fout = [], []
    dropped = 0

    for i in range(L):
        inp = inputs[i]
        out = outputs[i]
        if not isinstance(inp, str):
            inp = "" if inp is None else str(inp)
        if not isinstance(out, str):
            out = "" if out is None else str(out)

        if inp == "" or out == "":
            dropped += 1
            continue

        if len(inp) > max_chars or len(out) > max_chars:
            dropped += 1
            continue

        fin.append(inp)
        fout.append(out)

    return fin, fout, dropped


def _get_root_from_slot(slot: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the most original version (anchor) from slot history.
    Prefer version==0; fallback to history[0].
    """
    if not isinstance(slot, dict):
        return {}
    hist = slot.get("history", [])
    if not isinstance(hist, list) or len(hist) == 0:
        return {}

    for h in hist:
        if isinstance(h, dict) and int(h.get("version", -1)) == 0:
            return h

    h0 = hist[0]
    return h0 if isinstance(h0, dict) else {}


############################
# vLLM in worker process (same infra as之前那份改动)
############################
def manual_vllm_shutdown(llm_obj) -> None:
    if llm_obj is None:
        return
    for name in ("shutdown", "close"):
        if hasattr(llm_obj, name):
            try:
                getattr(llm_obj, name)()
                return
            except Exception:
                pass
    for eng_name in ("llm_engine", "_llm_engine", "_engine", "engine"):
        eng = getattr(llm_obj, eng_name, None)
        if eng is None:
            continue
        me = getattr(eng, "model_executor", None)
        if me is not None and hasattr(me, "shutdown"):
            try:
                me.shutdown()
                return
            except Exception:
                pass
        if hasattr(eng, "shutdown"):
            try:
                eng.shutdown()
                return
            except Exception:
                pass


def split_prompts(prompts, n):
    k, m = divmod(len(prompts), n)
    return [prompts[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def worker_fn(
    pretrained_model: str,
    gpu_ids: List[int],
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    max_model_len: int,
    max_generation_token: int,
    temp: float,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    try:
        faulthandler.enable()
        if HANG_DUMP_EVERY_SEC > 0:
            faulthandler.dump_traceback_later(HANG_DUMP_EVERY_SEC, repeat=True)
    except Exception:
        pass

    print(f"[vLLM] (worker {gpu_ids}) Loading env model: {pretrained_model}", flush=True)

    try:
        from vllm import LLM, SamplingParams

        llm_kwargs = dict(
            model=pretrained_model,
            dtype="bfloat16",
            tensor_parallel_size=len(gpu_ids),
            gpu_memory_utilization=0.85,
            max_model_len=int(max_model_len),
            enforce_eager=False,              # allow CUDA graph
            disable_custom_all_reduce=True,   # stability first
            trust_remote_code=True,
        )
        sig = inspect.signature(LLM.__init__)
        llm_kwargs = {k: v for k, v in llm_kwargs.items() if k in sig.parameters}

        llm = LLM(**llm_kwargs)

        sampling_params = SamplingParams(
            temperature=float(temp),
            top_p=0.95,
            top_k=-1,
            min_p=0.0,
            max_tokens=int(max_generation_token),
            stop=["</answer>", "User:", "Human:", "Assistant:", "<|im_end|>", "<|endoftext|>"],
        )
    except Exception as e:
        result_queue.put(("ERROR", f"init failed on GPUs {gpu_ids}: {repr(e)}"))
        return

    while True:
        task = task_queue.get()
        if task == "STOP":
            print(f"[vLLM] (worker {gpu_ids}) STOP received, shutting down...", flush=True)
            break

        task_id, prompts = task
        try:
            outs = llm.generate(prompts, sampling_params=sampling_params)
            result_texts = [
                o.outputs[0].text if (o.outputs and len(o.outputs) > 0) else ""
                for o in outs
            ]
            result_queue.put((task_id, result_texts))
        except Exception as e:
            result_queue.put(("ERROR", f"generate failed on GPUs {gpu_ids}: {repr(e)}"))
            break

    try:
        manual_vllm_shutdown(llm)
    except Exception:
        pass

    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def start_workers(pretrained_model, gpu_configs, max_model_len, max_generation_token, temp):
    task_queues, result_queues, processes = [], [], []
    for gpu_ids in gpu_configs:
        tq = mp.Queue()
        rq = mp.Queue()
        p = mp.Process(
            target=worker_fn,
            args=(pretrained_model, gpu_ids, tq, rq, max_model_len, max_generation_token, temp),
        )
        p.start()
        task_queues.append(tq)
        result_queues.append(rq)
        processes.append(p)
    return task_queues, result_queues, processes


def stop_workers(task_queues, result_queues, processes, join_timeout_s: float = 120.0):
    for q in task_queues:
        try:
            q.put("STOP")
        except Exception:
            pass

    for i, p in enumerate(processes):
        p.join(timeout=join_timeout_s)
        if p.is_alive():
            print(f"[WARN] worker {i} pid={p.pid} still alive after {join_timeout_s}s, terminate()", flush=True)
            p.terminate()
            p.join(timeout=10.0)
            if p.is_alive():
                print(f"[WARN] worker {i} pid={p.pid} still alive after terminate, kill()", flush=True)
                try:
                    p.kill()
                except Exception:
                    pass
                p.join(timeout=10.0)

    for q in (task_queues or []):
        try:
            q.cancel_join_thread()
            q.close()
        except Exception:
            pass
    for q in (result_queues or []):
        try:
            q.cancel_join_thread()
            q.close()
        except Exception:
            pass


def generate_results(
    all_prompts,
    gpu_groups,
    task_queues,
    result_queues,
    desc: str = "env",
    timeout_s: float = 1800.0,
):
    if not all_prompts:
        return []

    chunks = split_prompts(all_prompts, len(gpu_groups))

    jobs = []
    for i, (q, prompts) in enumerate(zip(task_queues, chunks)):
        if prompts:
            q.put((i, prompts))
            jobs.append(i)

    results_by_job = {}
    remaining = set(jobs)
    start_time = time.time()

    while remaining:
        if time.time() - start_time > timeout_s:
            raise RuntimeError(f"[{desc}] timeout waiting results; still missing jobs {sorted(remaining)}")

        for i, rq in enumerate(result_queues):
            if i not in remaining:
                continue
            try:
                task_id, result = rq.get(timeout=0.1)
            except Empty:
                continue

            if task_id == "ERROR":
                raise RuntimeError(f"[{desc}] worker {i} reported error: {result}")

            results_by_job[task_id] = result
            remaining.remove(task_id)

    out = []
    for i, prompts in enumerate(chunks):
        if not prompts:
            continue
        if i not in results_by_job:
            raise RuntimeError(f"[{desc}] missing result for job {i} (this should not happen)")
        out.extend(results_by_job[i])
    return out


############################
# Main: env rollout only (vLLM in child process)
############################
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    import sys

    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    faulthandler.enable()
    if HANG_DUMP_EVERY_SEC > 0:
        faulthandler.dump_traceback_later(HANG_DUMP_EVERY_SEC, repeat=True)

    config = get_config()
    project_name = config.experiment.project

    if int(config.experiment.current_epoch) == 1:
        policy_model = config.model.policy_model
    else:
        policy_model = "../" + project_name + "/ckpt/" + config.model.optimized_name

    fn = str(config.experiment.function)
    is_train = (fn == "train")
    is_eval = (fn in ("eval", "evaluation", "evaluate", "test"))

    if is_train:
        dataset = config.dataset.train_dataset
        num_node = int(config.experiment.num_node)
        node_index = int(config.experiment.node_index)
    else:
        dataset = config.dataset.eval_dataset if OmegaConf.select(config, "dataset.eval_dataset") else config.dataset.train_dataset
        num_node = int(config.experiment.num_node)
        node_index = int(config.experiment.node_index)

    outputs_name = ("rl-" if is_train else "eval-") + policy_model.replace("/", ".") + "-" + dataset

    policy_stage_path = get_policy_stage_output_path(project_name, outputs_name, num_node, node_index)
    if not os.path.exists(policy_stage_path):
        raise FileNotFoundError(f"policy-stage file not found: {policy_stage_path}")

    with open(policy_stage_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    data = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    if not isinstance(data, list):
        raise ValueError("policy-stage payload is not a list of task dicts")

    # ---- compute acc + per-test accuracy (GT & SYN) from existing correctness tables ----
    for item in data:
        corr_gt = item.get("correctness", [])
        corr_syn = item.get("syn_correctness", [])

        gt_pass_all = []
        if isinstance(corr_gt, list) and corr_gt:
            for row in corr_gt:
                gt_pass_all.append(bool(_pass_all(row)))

        item["gt_pass_all"] = gt_pass_all
        item["acc"] = (float(sum(1 for x in gt_pass_all if x)) / float(len(gt_pass_all))) if gt_pass_all else 0.0

        item["gt_test_acc"] = _per_test_pass_rate(corr_gt)
        item["syn_test_acc"] = _per_test_pass_rate(corr_syn)

    do_env = bool(is_train)
    if do_env:
        step = int(config.experiment.current_epoch)

        base_dataset_path = "../data/" + dataset + ".json"
        state_path, pending_path = get_state_paths(project_name, dataset, node_index, config)

        state = load_state_or_init_if_missing(base_dataset_path, state_path, num_node, node_index, config)
        slots = state["slots"]
        slot_map = {s["slot_id"]: s for s in slots}

        # 1) accept temp tasks (same logic)
        accepted, rejected = 0, 0
        for item in data:
            if item.get("source") != "temp":
                continue
            goal = item.get("goal")
            prev_acc = item.get("prev_acc")
            now_acc = float(item.get("acc", 0.0))

            ok = False
            if prev_acc is not None and goal in ("harder", "easier"):
                prev_acc = float(prev_acc)
                if now_acc > 0:
                    ok = ((now_acc < prev_acc) and (now_acc >= 0.2)) if goal == "harder" else ((now_acc > prev_acc) and (now_acc <= 0.8))

            item["temp_accept"] = bool(ok)

            if not ok:
                rejected += 1
                continue

            slot_id = item.get("slot_id")
            if slot_id not in slot_map:
                item["temp_accept"] = False
                rejected += 1
                continue

            slot = slot_map[slot_id]
            next_v = int(slot["active"]["version"]) + 1

            new_active = {
                "version": next_v,
                "question": item.get("question", ""),
                "test_input": item.get("test_input", []) if isinstance(item.get("test_input", []), list) else [],
                "test_output": item.get("test_output", []) if isinstance(item.get("test_output", []), list) else [],
                "test_method": item.get("test_method", "stdio"),
                "test_time_limit": item.get("test_time_limit", 4),
                "created_step": int(item.get("created_step", step)),
                "accepted_step": step,
                "source": "env",
                "goal": goal,
                "prev_acc": float(prev_acc),
                "acc_after": float(now_acc),
            }
            slot["active"] = new_active
            slot["history"].append(new_active)
            accepted += 1

        save_state(state_path, state)

        # 2) build env_requests
        acc_low = 0.8
        acc_high = 0.2

        env_requests = []
        for item in data:
            if item.get("source") != "dataset":
                continue

            a = float(item.get("acc", 0.0))
            if a >= acc_low:
                goal = "harder"
            elif a < acc_high:
                goal = "easier"
            else:
                continue

            slot_id = item.get("slot_id", None)
            if slot_id is None or slot_id not in slot_map:

                continue

            slot = slot_map[slot_id]
            root = _get_root_from_slot(slot)

            max_ut_chars_in_prompt = 100

            # current active UT (from item)
            gt_in_f, gt_out_f, gt_drop = _filter_ut_pairs_for_prompt(
                item.get("test_input", []),
                item.get("test_output", []),
                max_chars=max_ut_chars_in_prompt,
            )

            syn_in_f, syn_out_f, syn_drop = _filter_ut_pairs_for_prompt(
                item.get("syn_input", []),
                item.get("syn_output", []),
                max_chars=max_ut_chars_in_prompt,
            )

            # root anchor UT (from state history v0)
            root_in_f, root_out_f, root_drop = _filter_ut_pairs_for_prompt(
                root.get("test_input", []),
                root.get("test_output", []),
                max_chars=max_ut_chars_in_prompt,
            )

            gt_rate_f = item.get("gt_test_acc", [])
            gt_rate_f = gt_rate_f[:min(len(gt_in_f), len(gt_out_f))] if isinstance(gt_rate_f, list) else []

            syn_rate_f = item.get("syn_test_acc", [])
            syn_rate_f = syn_rate_f[:min(len(syn_in_f), len(syn_out_f))] if isinstance(syn_rate_f, list) else []

            gt_tests_json = _tests_json_for_prompt(gt_in_f, gt_out_f, gt_rate_f)
            syn_tests_json = _tests_json_for_prompt(syn_in_f, syn_out_f, syn_rate_f)

            root_tests_json = _tests_json_for_prompt(root_in_f, root_out_f, rates=[])

            if (gt_drop + syn_drop + root_drop) > 0:
                cprint(
                    f"[env] slot_id={slot_id} dropped_ut: gt={gt_drop}, syn={syn_drop}, root={root_drop}, "
                    f"thr={max_ut_chars_in_prompt}",
                    "yellow",
                )

            root_question = (root.get("question", "") or "").strip()
            parent_question = (item.get("question", "") or "").strip()

            env_requests.append(
                {
                    "slot_id": slot_id,
                    "goal": goal,
                    "prev_acc": a,

                    # anchor (root)
                    "root_question": root_question if root_question else parent_question,
                    "root_tests_json": root_tests_json,

                    # current
                    "parent_question": parent_question,
                    "parent_test_input": gt_in_f,
                    "parent_test_output": gt_out_f,
                    "parent_gt_tests_json": gt_tests_json,
                    "parent_syn_tests_json": syn_tests_json,

                    "created_step": step,
                }
            )

        # 3) env model selection
        env_model_sel = config.model.environment_model
        env_model = env_model_sel

        env_max_model_len = int(config.rollout.environment.model_length)
        env_max_generation_token = int(config.rollout.environment.max_gen_length)
        env_temp = float(config.rollout.environment.temperature)

        env_num_tests = int(OmegaConf.select(config, "rollout.environment.num_unit_tests", default=4))
        env_num_tests = max(1, env_num_tests)

        env_k_sample = _env_k_sample_from_config(config)

        ENV_PROMPT = r"""<|im_start|>system
You are a coding dataset curator. Generate ONE new coding problem and its ground-truth unit tests.

Return ONLY a JSON object with EXACT keys:
{
  "question": "...",
  "test_method": "stdio",
  "test_time_limit": 4,
  "test_input": ["...", "...", ...],
  "test_output": ["...", "...", ...]
}

IMPORTANT:
- test_input and test_output must be lists of the SAME length (>= {{num_tests}}).
- Each test_input is the exact stdin content; each test_output is the exact expected stdout content.
- Output JSON only. No extra text.
<|im_end|>
<|im_start|>user
Goal: make the new problem {{goal}}.

CRITICAL ANCHOR (do NOT drift away):
The new problem MUST stay close to the ORIGINAL (root) problem in topic, core skill, input/output format style, and constraints.
You may only adjust difficulty in a controlled way. Do NOT switch to a different domain/task type.

[ORIGINAL ROOT PROBLEM]
{{root_question}}

[ORIGINAL ROOT UNIT TESTS (reference; truncated)]
{{root_tests_json}}

[CURRENT ACTIVE PROBLEM]
{{parent_question}}

Reference unit tests (ground truth for current active; truncated):
- test_input examples:
{{parent_test_input}}
- test_output examples:
{{parent_test_output}}

Policy performance on each unit test (pass_rate over current policy code samples):
Ground-truth unit tests:
{{parent_gt_tests_json}}

Synthetic unit tests (may not be accurate):
{{parent_syn_tests_json}}

Constraints:
- Keep similar general topic/skill to the ORIGINAL ROOT problem.
- The new problem must be self-contained and unambiguous.
- Provide at least {{num_tests}} ground-truth tests in "test_input"/"test_output".
<|im_end|>
<|im_start|>assistant
"""

        env_prompts_expanded = []
        for req in env_requests:
            base_pmt = Template(ENV_PROMPT).render(
                goal=req["goal"],

                root_question=req.get("root_question", ""),
                root_tests_json=req.get("root_tests_json", "[]"),

                parent_question=req["parent_question"],
                parent_test_input=req.get("parent_test_input", []),
                parent_test_output=req.get("parent_test_output", []),
                parent_gt_tests_json=req.get("parent_gt_tests_json", "[]"),
                parent_syn_tests_json=req.get("parent_syn_tests_json", "[]"),
                num_tests=env_num_tests,
            )
            for _ in range(env_k_sample):
                env_prompts_expanded.append(base_pmt)

        if len(env_prompts_expanded) == 0:
            next_pending = []
            save_pending(pending_path, next_pending)

        else:
            cprint(
                f"[env] generating {len(env_requests)} tasks x {env_k_sample} rollouts = {len(env_prompts_expanded)} ...",
                "yellow",
            )

            # ---- parent-side visible GPU check (avoid cuda init if possible) ----
            cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            if cvd:
                visible = len([x for x in cvd.split(",") if x.strip() != ""])
            else:
                try:
                    import torch
                    visible = torch.cuda.device_count()
                except Exception:
                    visible = 0

            if visible < TP_SIZE:
                raise RuntimeError(
                    f"Need at least {TP_SIZE} visible GPUs for TP={TP_SIZE}, but got visible_gpus={visible}. "
                    f"Please set CUDA_VISIBLE_DEVICES to expose {TP_SIZE} GPUs."
                )

            cprint(f"[vLLM] parent CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", "cyan")
            cprint(f"[vLLM] visible_gpus={visible}, TP_SIZE={TP_SIZE}, CHUNK_SIZE={CHUNK_SIZE}", "cyan")

            # ---- single worker with one TP engine ----
            gpu_groups = [list(range(TP_SIZE))]
            task_queues, result_queues, processes = start_workers(
                str(env_model),
                gpu_groups,
                env_max_model_len,
                env_max_generation_token,
                env_temp,
            )

            env_outputs_expanded: List[str] = []
            try:
                cs = int(CHUNK_SIZE) if CHUNK_SIZE is not None else 0
                if cs <= 0:
                    cs = len(env_prompts_expanded)

                cursor = 0
                while cursor < len(env_prompts_expanded):
                    sub = env_prompts_expanded[cursor: cursor + cs]
                    print(f"[ENV] batch {cursor}..{min(cursor + cs, len(env_prompts_expanded))} (n={len(sub)})", flush=True)

                    sub_outs = generate_results(
                        sub,
                        gpu_groups,
                        task_queues,
                        result_queues,
                        desc="env",
                        timeout_s=float(GEN_TIMEOUT_S),
                    )
                    env_outputs_expanded.extend([x or "" for x in sub_outs])
                    cursor += cs

            finally:
                stop_workers(task_queues, result_queues, processes)

                # best-effort cache cleanup; NO synchronize by default
                try:
                    import torch
                    time.sleep(1)
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            # Choose FIRST valid per request (same logic)
            next_pending = []
            for ridx, req in enumerate(env_requests):
                chosen_obj = None
                chosen_out = None
                chosen_pmt = None

                base = ridx * env_k_sample
                for aidx in range(env_k_sample):
                    out = env_outputs_expanded[base + aidx] if (base + aidx) < len(env_outputs_expanded) else ""
                    obj = parse_json_obj(out)
                    if not isinstance(obj, dict):
                        continue

                    q = obj.get("question", "")
                    tin = obj.get("test_input", [])
                    tout = obj.get("test_output", [])

                    if not isinstance(q, str) or not q.strip():
                        continue
                    if not isinstance(tin, list) or not isinstance(tout, list):
                        continue
                    if len(tin) != len(tout):
                        continue
                    if len(tin) < env_num_tests:
                        continue

                    chosen_obj = obj
                    chosen_out = out
                    chosen_pmt = env_prompts_expanded[base + aidx]
                    break

                if chosen_obj is None:
                    continue

                q = chosen_obj.get("question", "").strip()
                tin = [str(x) for x in chosen_obj.get("test_input", [])]
                tout = [str(x) for x in chosen_obj.get("test_output", [])]

                next_pending.append(
                    {
                        "slot_id": req["slot_id"],
                        "goal": req["goal"],
                        "prev_acc": req["prev_acc"],
                        "created_step": step,

                        # (debug / audit)
                        "root_question": req.get("root_question", ""),
                        "root_tests_json": req.get("root_tests_json", "[]"),

                        "env_prompt": chosen_pmt,
                        "env_full_output": chosen_out,

                        "question": q,
                        "test_method": "stdio",
                        "test_time_limit": 4,
                        "test_input": tin,
                        "test_output": tout,
                    }
                )

            save_pending(pending_path, next_pending)
            cprint("[env] co-evolve done!", "yellow")

    else:
        cprint("[env] skipped (not train).", "yellow")

    # write back same file name
    output_file_name = policy_stage_path
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    cprint(f"[env rollout] saved outputs: {output_file_name}", "cyan")

    import gc
    gc.collect()

    if FORCE_EXIT:
        os._exit(0)
