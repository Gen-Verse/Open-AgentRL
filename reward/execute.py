# -*- coding: utf-8 -*-
"""
Execute unit tests (GT + SYN) for a policy-stage output JSON file.

Input:
- policy_stage_path: ../{project_name}/temp_data/outputs-({node_index}-){outputs_name}.json

Output:
- overwrite the SAME file, only adding/updating these keys per item:
  - execution_result:        [m_code][m_gt]  (str)
  - correctness:             [m_code][m_gt]  (bool)
  - syn_execution_result:    [m_code][m_syn] (str)
  - syn_correctness:         [m_code][m_syn] (bool)

"Standard execution" behavior:
- Use multiprocessing Processes to exec python code with redirected stdin/stdout
- Chunked execution: each chunk launches ~max_procs processes at once
- Prefer mp context "fork" on Linux to avoid spawn overhead/jitter
"""

import os
import io
import sys
import json
import time
import math
import argparse
import multiprocessing as mp
from typing import List, Any, Tuple, Optional

from omegaconf import OmegaConf
from termcolor import cprint


# -------------------------
# Path helper (as you gave)
# -------------------------
def get_policy_stage_output_path(project_name, outputs_name, num_node, node_index):
    return f"../{project_name}/temp_data/outputs-{node_index}-{outputs_name}.json"


# -------------------------
# Config (optional)
# -------------------------
def get_config():
    cli_conf = OmegaConf.from_cli()
    cfg_path = OmegaConf.select(cli_conf, "config", default=None)
    if cfg_path is None:
        raise ValueError('Missing "config=...". Either pass --policy_stage_path directly, or provide config=path/to.yaml')
    yaml_conf = OmegaConf.load(cfg_path)
    return OmegaConf.merge(yaml_conf, cli_conf)


# -------------------------
# Standard execution core
# -------------------------
def worker(script: str, input_val: str, output_queue):
    # Create an iterator over the input lines.
    input_lines = iter((input_val or "").splitlines())

    # Override the input() function in the exec context.
    def fake_input(prompt=""):
        try:
            return next(input_lines)
        except StopIteration:
            raise EOFError("No more input")

    # Redirect sys.stdout to capture printed output.
    stdout_capture = io.StringIO()
    original_stdout = sys.stdout
    original_stdin = sys.stdin  # Save original stdin
    sys.stdout = stdout_capture
    sys.stdin = io.StringIO(input_val or "")  # Simulate stdin with input_val

    context = {
        "__name__": "__main__",   # Ensures that `if __name__ == "__main__": ...` will fire
        "input": fake_input,
    }

    try:
        exec(script, context)
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except SystemExit:
        printed_output = stdout_capture.getvalue()
        output_queue.put(printed_output)

    except Exception as e:
        output_queue.put(f"error: {e}")

    finally:
        sys.stdout = original_stdout
        sys.stdin = original_stdin


def _exec_ctx() -> mp.context.BaseContext:
    """
    Prefer fork on posix to reduce spawn overhead / scheduling jitter.
    """
    if os.name == "posix":
        try:
            return mp.get_context("fork")
        except Exception:
            pass
    return mp.get_context()


def run_scripts_with_timeout(
    scripts: List[str],
    inputs: List[str],
    time_limits: List[float],
    worker_fn,
):
    results = [None] * len(scripts)
    processes = []
    queues = []
    deadlines = []

    ctx = _exec_ctx()

    for i in range(len(scripts)):
        q = ctx.Queue()
        p = ctx.Process(target=worker_fn, args=(scripts[i], inputs[i], q))
        processes.append(p)
        queues.append(q)
        p.start()
        deadlines.append(time.time() + float(time_limits[i]))

    while any(p.is_alive() for p in processes):
        now = time.time()
        for i, p in enumerate(processes):
            if p.is_alive() and now >= deadlines[i]:
                p.terminate()
                results[i] = "Timeout Error"
        time.sleep(0.001)

    for i, p in enumerate(processes):
        if results[i] is None:
            try:
                results[i] = queues[i].get_nowait()
            except Exception as e:
                results[i] = f"Execution Error: {e}"

    return results


def test_if_eq(x: str, y: str) -> bool:
    return " ".join((x or "").split()) == " ".join((y or "").split())


def get_chunk_indices(n: int, num_chunks: int):
    # standard-style chunk indices
    chunk_size = n // num_chunks
    remainder = n % num_chunks
    indices = []
    start = 0
    for i in range(num_chunks):
        extra = 1 if i < remainder else 0
        end = start + chunk_size + extra
        indices.append((start, end))
        start = end
    return indices


def run_scripts_with_chunk(
    code_list: List[str],
    test_input_list: List[str],
    time_limit_list: List[float],
    worker_fn,
    num_chunks: int,
):
    chunks = get_chunk_indices(len(code_list), num_chunks)
    exe_results = []
    for start, end in chunks:
        sub_code_list = code_list[start:end]
        sub_test_input_list = test_input_list[start:end]
        sub_time_limit_list = time_limit_list[start:end]
        sub_exe_results = run_scripts_with_timeout(
            sub_code_list, sub_test_input_list, sub_time_limit_list, worker_fn
        )
        exe_results = exe_results + sub_exe_results
    return exe_results


# -------------------------
# Dataset-level execution
# -------------------------
def _safe_list(x):
    return x if isinstance(x, list) else []


def compute_num_chunks_by_max_procs(
    data: List[dict],
    input_key: str,
    output_key: str,
    max_procs: int,
) -> int:
    """
    Let each chunk launch at most ~max_procs subprocesses at once.
    total_pairs = sum_over_items( len(codes) * n_valid_tests )
    num_chunks = ceil(total_pairs / max_procs)
    """
    max_procs = max(1, int(max_procs))
    total_pairs = 0

    for item in data:
        codes = _safe_list(item.get("extracted_output", []))
        if len(codes) == 0:
            continue

        test_inp = _safe_list(item.get(input_key, []))
        test_out = _safe_list(item.get(output_key, []))
        m_case = min(len(test_inp), len(test_out))
        if m_case <= 0:
            continue

        n_valid = 0
        for k in range(m_case):
            inp = test_inp[k]
            out = test_out[k]
            if isinstance(inp, str) and isinstance(out, str) and inp != "" and out != "":
                n_valid += 1

        if n_valid <= 0:
            continue

        total_pairs += len(codes) * n_valid

    if total_pairs <= 0:
        return 1
    return max(1, int(math.ceil(total_pairs / max_procs)))


def execute_unit_tests_on_key(
    data: List[dict],
    input_key: str,
    output_key: str,
    exec_key: str,
    corr_key: str,
    max_procs: int,
):
    """
    Execute all codes against tests specified by (input_key, output_key).
    Writes:
      item[exec_key]  : [m_code][m_case] str
      item[corr_key]  : [m_code][m_case] bool
    Invalid test ("" input or "" output) => exec="Invalid Test", correctness=False
    """
    # Flatten valid pairs
    idx_code: List[Tuple[int, int]] = []
    idx_case: List[int] = []
    valid_mask: List[bool] = []

    code_list: List[str] = []
    inp_list: List[str] = []
    tl_list: List[float] = []

    for idx, item in enumerate(data):
        tl = float(item.get("test_time_limit", 4))
        codes = _safe_list(item.get("extracted_output", []))
        test_inp = _safe_list(item.get(input_key, []))
        test_out = _safe_list(item.get(output_key, []))

        m_code = len(codes)
        m_case = min(len(test_inp), len(test_out))

        item[exec_key] = [[""] * m_case for _ in range(m_code)]
        item[corr_key] = [[False] * m_case for _ in range(m_code)]

        for c_idx, code in enumerate(codes):
            for k in range(m_case):
                inp = test_inp[k]
                out = test_out[k]
                is_valid = (isinstance(inp, str) and isinstance(out, str) and inp != "" and out != "")
                idx_code.append((idx, c_idx))
                idx_case.append(k)
                valid_mask.append(is_valid)
                if is_valid:
                    code_list.append(code)
                    inp_list.append(inp)
                    tl_list.append(tl)

    if code_list:
        num_chunks = compute_num_chunks_by_max_procs(data, input_key, output_key, max_procs=max_procs)
        cprint(f"[exec:{input_key}] valid_pairs={len(code_list)}  max_procs={max_procs}  chunks={num_chunks}", "green")
        exe_results_valid = run_scripts_with_chunk(code_list, inp_list, tl_list, worker, num_chunks)
    else:
        exe_results_valid = []

    ptr = 0
    for i in range(len(idx_code)):
        idx, c_idx = idx_code[i]
        k = idx_case[i]
        item = data[idx]

        if not valid_mask[i]:
            item[exec_key][c_idx][k] = "Invalid Test"
            item[corr_key][c_idx][k] = False
            continue

        res = exe_results_valid[ptr]
        ptr += 1
        item[exec_key][c_idx][k] = res
        exp_out = _safe_list(item.get(output_key, []))[k]
        item[corr_key][c_idx][k] = test_if_eq(res, exp_out)

    return data


# -------------------------
# Main
# -------------------------
def _resolve_policy_stage_path_from_config(cfg) -> str:
    project_name = cfg.experiment.project

    fn = str(cfg.experiment.function)
    is_train = (fn == "train")

    num_node = int(OmegaConf.select(cfg, "experiment.num_node", default=1))
    node_index = int(OmegaConf.select(cfg, "experiment.node_index", default=0))

    # policy model name resolution (same logic as your pipeline)
    if int(cfg.experiment.current_epoch) == 1:
        policy_model = cfg.model.policy_model
    else:
        policy_model = "../" + project_name + "/ckpt/" + cfg.model.optimized_name

    if is_train:
        dataset = cfg.dataset.train_dataset
    else:
        dataset = cfg.dataset.eval_dataset if OmegaConf.select(cfg, "dataset.eval_dataset") else cfg.dataset.train_dataset

    outputs_name = ("rl-" if is_train else "eval-") + policy_model.replace("/", ".") + "-" + dataset
    return get_policy_stage_output_path(project_name, outputs_name, num_node, node_index)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy_stage_path",
        type=str,
        default=None,
        help="If provided, use this path directly. Otherwise will compute from OmegaConf config=...",
    )
    parser.add_argument(
        "--max_procs",
        type=int,
        default=None,
        help="Max concurrent subprocesses per chunk. If not set, use execute.num_chunk from config or default=8.",
    )
    args, _unknown = parser.parse_known_args()

    cfg = get_config()
    policy_stage_path = _resolve_policy_stage_path_from_config(cfg)
    max_procs = cfg.execute.num_chunk

    if not os.path.exists(policy_stage_path):
        raise FileNotFoundError(f"policy_stage_path not found: {policy_stage_path}")

    cprint(f"[load] {policy_stage_path}", "cyan")
    with open(policy_stage_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    # support payload either as list or {"data": [...]}
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
        data = payload["data"]
        payload_is_dict = True
    elif isinstance(payload, list):
        data = payload
        payload_is_dict = False
    else:
        raise ValueError("policy_stage file must be a list OR a dict with key 'data' as a list")

    # Execute GT UTs
    cprint("[exec] running ground-truth unit tests (test_input/test_output)...", "yellow")
    data = execute_unit_tests_on_key(
        data,
        input_key="test_input",
        output_key="test_output",
        exec_key="execution_result",
        corr_key="correctness",
        max_procs=max_procs,
    )

    # Execute SYN UTs
    cprint("[exec] running synthetic unit tests (syn_input/syn_output)...", "yellow")
    data = execute_unit_tests_on_key(
        data,
        input_key="syn_input",
        output_key="syn_output",
        exec_key="syn_execution_result",
        corr_key="syn_correctness",
        max_procs=max_procs,
    )

    # Write back to the same file (preserve original shape)
    if payload_is_dict:
        payload["data"] = data
        out_obj = payload
    else:
        out_obj = data

    with open(policy_stage_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    cprint(f"[done] wrote execution results back to: {policy_stage_path}", "cyan")


if __name__ == "__main__":
    # keep your global default if you want; execution uses fork context internally
    mp.set_start_method("spawn", force=True)
    main()
