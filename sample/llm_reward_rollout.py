
from __future__ import annotations

import os
import re
import sys
import gc
import json
import time
import random
import inspect
import faulthandler
import multiprocessing as mp
from queue import Empty
from typing import List, Any, Dict, Tuple

from jinja2 import Template
from termcolor import cprint
from omegaconf import OmegaConf


# ----------------------------
# Env (best-effort)
# ----------------------------
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# TP / batch chunk
TP_SIZE = 8
CHUNK_SIZE = 1024

# Hang debug: dump tracebacks periodically if stuck (seconds)
HANG_DUMP_EVERY_SEC = int(os.environ.get("HANG_DUMP_EVERY_SEC", "300"))

# Write tuning
JSON_USE_INDENT = False  # keep False for speed/size
JSON_INDENT = 2

# Save-size control (avoid massive json)
STORE_SYNONYMOUS_PROMPT = True
STORE_FULL_OUTPUT = True
MAX_SAVE_CHARS_PROMPT = int(os.environ.get("MAX_SAVE_CHARS_PROMPT", "4000"))
MAX_SAVE_CHARS_OUTPUT = int(os.environ.get("MAX_SAVE_CHARS_OUTPUT", "8000"))

# If you *must* guarantee exit even if shutdown hangs, set FORCE_EXIT=1
FORCE_EXIT = bool(int(os.environ.get("FORCE_EXIT", "0")))

# Parent waiting timeout (seconds) for each generate_results call
# (you can override with env var if needed)
GEN_TIMEOUT_S = float(os.environ.get("GEN_TIMEOUT_S", "1800"))


# ----------------------------
# Config
# ----------------------------
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


# ----------------------------
# Paths
# ----------------------------
def get_policy_stage_output_path(project_name: str, outputs_name: str, num_node: int, node_index: int) -> str:
    return f"../{project_name}/temp_data/outputs-{int(node_index)}-{outputs_name}.json"


# ----------------------------
# Unit test extraction
# ----------------------------
def _normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\r\n", "\n")
    return s.strip("\n")


def extract_test_cases(full_output: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Parse:
      **Test Input:**
      ```...```
      **Test Output:**
      ```...```
    Return ([test_input], [test_output], [example_text]) or ([""],[""],[""]) on failure.
    """
    full_output = full_output or ""
    pattern_input_backticks = r"\*\*Test Input:\*\*\s*```(.*?)```"
    pattern_output_backticks = r"\*\*Test Output:\*\*\s*```(.*?)```"
    matches_input = re.findall(pattern_input_backticks, full_output, re.DOTALL)
    matches_output = re.findall(pattern_output_backticks, full_output, re.DOTALL)

    fail = [""]

    if matches_input:
        test_input = [_normalize(matches_input[-1].lstrip("\n"))]
    else:
        pattern_input_plain = r"\*\*Test Input:\*\*\s*([\s\S]*?)(?=\*\*Test Output:\*\*)"
        matches_input_plain = re.findall(pattern_input_plain, full_output, re.DOTALL)
        test_input = [_normalize(matches_input_plain[-1].strip())] if matches_input_plain else fail

    if matches_output:
        test_output = [_normalize(matches_output[-1].lstrip("\n"))]
    else:
        pattern_output_plain = r"\*\*Test Output:\*\*\s*([\s\S]*?)(?=\*\*Explanation:|\*\*Test Input:|$)"
        matches_output_plain = re.findall(pattern_output_plain, full_output, re.DOTALL)
        test_output = [_normalize(matches_output_plain[-1].strip())] if matches_output_plain else fail

    idx = full_output.rfind("**Test Input:**")
    example_text = [full_output[idx:]] if idx != -1 else fail

    if test_input == fail or test_output == fail or example_text == fail:
        return fail, fail, fail
    return test_input, test_output, example_text


# ----------------------------
# Robust JSON write (atomic)
# ----------------------------
def atomic_write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"

    t0 = time.time()
    with open(tmp, "w", encoding="utf-8") as f:
        if JSON_USE_INDENT:
            json.dump(obj, f, ensure_ascii=False, indent=JSON_INDENT)
        else:
            json.dump(obj, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    cprint(f"[io] json saved: {path} (write_time={time.time()-t0:.1f}s)", "cyan")
    sys.stdout.flush()


# ----------------------------
# vLLM shutdown helpers
# ----------------------------
def manual_vllm_shutdown(llm_obj) -> None:
    """
    Best-effort: stop executor to avoid hanging at interpreter exit.
    Works across some vLLM versions (attribute names vary).
    """
    if llm_obj is None:
        return

    # direct shutdown if exists
    for name in ("shutdown", "close"):
        if hasattr(llm_obj, name):
            try:
                getattr(llm_obj, name)()
                return
            except Exception:
                pass

    # engine shutdown
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


# ----------------------------
# vLLM Worker Pool (same infra pattern as your "good" code)
# ----------------------------
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
    # isolate CUDA context inside worker
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    # optional: worker-side faulthandler too
    try:
        faulthandler.enable()
        if HANG_DUMP_EVERY_SEC > 0:
            faulthandler.dump_traceback_later(HANG_DUMP_EVERY_SEC, repeat=True)
    except Exception:
        pass

    print(f"[vLLM] (worker {gpu_ids}) Loading model: {pretrained_model}", flush=True)

    try:
        from vllm import LLM, SamplingParams

        llm_kwargs = dict(
            model=pretrained_model,
            dtype="bfloat16",
            tensor_parallel_size=len(gpu_ids),
            gpu_memory_utilization=0.85,
            max_model_len=int(max_model_len),
            enforce_eager=False,              # keep original
            disable_custom_all_reduce=True,   # keep original
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

    # best-effort shutdown
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
    # 1) tell workers to stop
    for q in task_queues:
        try:
            q.put("STOP")
        except Exception:
            pass

    # 2) join / terminate / kill
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

    # 3) IMPORTANT: do NOT join_thread() (it can hang forever). cancel it.
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
    desc: str = "reward",
    timeout_s: float = 1800.0,
):
    """
    Same semantics as your "good" code:
    - split prompts by worker count
    - send one job per worker
    - poll result queues with timeout
    """
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
        now = time.time()
        if now - start_time > timeout_s:
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


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  

    # Make stdout/stderr line-buffered (helps over ssh), and enable hang stack dumps.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    faulthandler.enable()
    if HANG_DUMP_EVERY_SEC > 0:
        faulthandler.dump_traceback_later(HANG_DUMP_EVERY_SEC, repeat=True)

    config = get_config()
    project_name = str(config.experiment.project)

    # policy_model selection (same as your policy script)
    if int(config.experiment.current_epoch) == 1:
        policy_model = str(config.model.policy_model)
    else:
        policy_model = "../" + project_name + "/ckpt/" + str(config.model.optimized_name)

    fn = str(config.experiment.function)
    is_train = (fn == "train")
    is_eval = (fn in ("eval", "evaluation", "evaluate", "test"))

    if is_train:
        dataset = str(config.dataset.train_dataset)
        num_node = int(config.experiment.num_node)
        node_index = int(config.experiment.node_index)
    else:
        dataset = (
            str(config.dataset.eval_dataset)
            if OmegaConf.select(config, "dataset.eval_dataset")
            else str(config.dataset.train_dataset)
        )
        num_node = int(config.experiment.num_node)
        node_index = int(config.experiment.node_index)

    outputs_name = ("rl-" if is_train else "eval-") + policy_model.replace("/", ".") + "-" + dataset
    policy_stage_path = get_policy_stage_output_path(project_name, outputs_name, num_node, node_index)

    cprint(f"[reward rollout] policy_stage_path = {policy_stage_path}", "cyan")
    sys.stdout.flush()

    if not os.path.exists(policy_stage_path):
        raise FileNotFoundError(f"policy-stage file not found: {policy_stage_path}")

    # Load policy-stage payload
    with open(policy_stage_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    data = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    if not isinstance(data, list):
        raise ValueError("policy-stage payload is not a list (or dict with key 'data')")

    num_task = len(data)
    cprint(f"[reward rollout] loaded tasks: {num_task}", "cyan")
    sys.stdout.flush()

    # reward model selection
    if int(config.experiment.current_epoch) == 1:
        reward_model = str(config.model.reward_model)
    else:
        reward_model = "../" + project_name + "/ckpt/" + str(config.model.optimized_reward_name)

    reward_max_model_len = int(config.rollout.reward.model_length)
    reward_max_generation_token = int(config.rollout.reward.max_gen_length)
    reward_temp = float(config.rollout.reward.temperature)
    reward_k_sample = int(config.rollout.reward.num_response_per_task)

    REWARD_TEST_PROMPT = r"""<|im_start|>system
You are a rigorous unit-test designer for coding problems.
You must produce exactly ONE new test example that is correct and discriminative.
<|im_end|>
<|im_start|>user
You need to provide a new test example. A good test example should be completely accurate and conform to the problem’s format requirements, while also possessing enough discriminative power to distinguish correct code from incorrect code.

Before providing a test example, you must think carefully and reason step by step to derive an input and output you are very confident are correct. For example, start by designing an input you can reliably handle, then compute the output step by step. If you're unsure about the output, revise or re-design the input to ensure accuracy. Directly providing input/output pairs without this process is discouraged, as it often results in low accuracy.

Finally, after completing these previous thinking and derivation steps (you should not write the final test example unless you have gone through these steps very thoroughly), you MUST put your final test example in the following format:

**Test Input:**
```<put the EXACT stdin content here>```

**Test Output:**
```<put the EXACT stdout content here>```

**Explanation:**
<brief explanation here>

IMPORTANT:
- Output must contain exactly one **Test Input:** block and one **Test Output:** block.
- Use triple backticks exactly as shown.
- The test must be self-contained and match the problem format.

Problem:
{{problem}}
<|im_end|>
<|im_start|>assistant
"""

    if bool(OmegaConf.select(config, "rollout.reward.start_with_think", default=False)):
        REWARD_TEST_PROMPT += "<think>"

    # Build prompts
    all_prompts: List[str] = []
    meta: List[Tuple[int, int, str]] = []  # (task_idx, k, prompt)
    for ti in range(num_task):
        prob = data[ti].get("question", "") or ""
        for k in range(reward_k_sample):
            pmt = Template(REWARD_TEST_PROMPT).render(problem=prob)
            all_prompts.append(pmt)
            meta.append((ti, k, pmt))

    cprint(f"[reward rollout] start generating: total_prompts={len(all_prompts)}", "yellow")
    sys.stdout.flush()

    # parent-side visible GPU check (avoid initializing CUDA if possible)
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        visible = len([x for x in cvd.split(",") if x.strip() != ""])
    else:
        import torch
        visible = torch.cuda.device_count()

    if visible < TP_SIZE:
        raise RuntimeError(
            f"Need at least {TP_SIZE} visible GPUs for TP={TP_SIZE}, but got visible_gpus={visible}. "
            f"Please set CUDA_VISIBLE_DEVICES to expose {TP_SIZE} GPUs."
        )
    if visible != TP_SIZE:
        cprint(f"[vLLM][WARN] visible_gpus={visible} != TP_SIZE={TP_SIZE}. Will only use first TP_SIZE GPUs in worker.", "yellow")

    cprint(f"[vLLM] parent CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", "cyan")
    cprint(f"[vLLM] visible_gpus={visible}, TP_SIZE={TP_SIZE}, CHUNK_SIZE={CHUNK_SIZE}", "cyan")
    sys.stdout.flush()

    # single worker holding one TP engine (same as your old "single engine TP")
    gpu_groups = [list(range(TP_SIZE))]

    # start worker(s)
    cprint(f"[vLLM] start worker(s) for reward model: {reward_model}", "green")
    sys.stdout.flush()
    task_queues, result_queues, processes = start_workers(
        reward_model,
        gpu_groups,
        reward_max_model_len,
        reward_max_generation_token,
        reward_temp,
    )

    # Shuffle -> generate -> restore (same logic; only inference method changed)
    R = len(all_prompts)
    idxs = list(range(R))
    shuf = idxs[:]
    random.shuffle(shuf)
    shuf_prompts = [all_prompts[i] for i in shuf]

    outs: List[str] = [""] * R

    try:
        cprint("[reward rollout] generating with vLLM worker ...", "yellow")
        sys.stdout.flush()

        # chunked submission (preserves your previous chunk behavior)
        cs = int(CHUNK_SIZE) if CHUNK_SIZE is not None else 0
        if cs <= 0:
            cs = len(shuf_prompts)

        cursor = 0
        while cursor < len(shuf_prompts):
            sub = shuf_prompts[cursor: cursor + cs]
            print(f"[REWARD] batch {cursor}..{min(cursor + cs, len(shuf_prompts))} (n={len(sub)})", flush=True)

            sub_outs = generate_results(
                sub,
                gpu_groups,
                task_queues,
                result_queues,
                desc="reward",
                timeout_s=float(GEN_TIMEOUT_S),
            )

            # write back to outs in shuffled order first
            for j, out_text in enumerate(sub_outs):
                global_shuf_pos = cursor + j
                orig_i = shuf[global_shuf_pos]  # original index in [0..R)
                outs[orig_i] = out_text or ""

            cursor += cs

    finally:
        # always stop worker(s), even if timeout/error
        stop_workers(task_queues, result_queues, processes)

        # Cleanup (avoid synchronize hang)
        try:
            import torch
            time.sleep(1)
            torch.cuda.empty_cache()
        except Exception:
            pass

    cprint("[reward rollout] generation done. parsing outputs ...", "yellow")
    sys.stdout.flush()

    # Initialize fields
    for item in data:
        item["syn_input"] = [""] * reward_k_sample
        item["syn_output"] = [""] * reward_k_sample
        item["syn_full_output"] = [""] * reward_k_sample
        item["syn_example_text"] = [""] * reward_k_sample
        item["syn_prompt"] = [""] * reward_k_sample

    # Fill fields (same mapping: zip(outs, meta) where meta order == all_prompts order)
    for out, (ti, k, pmt) in zip(outs, meta):
        test_inp, test_out, example_text = extract_test_cases(out)

        # size control
        save_prompt = pmt if STORE_SYNONYMOUS_PROMPT else ""
        save_out = out if STORE_FULL_OUTPUT else ""

        if save_prompt and MAX_SAVE_CHARS_PROMPT > 0 and len(save_prompt) > MAX_SAVE_CHARS_PROMPT:
            save_prompt = save_prompt[-MAX_SAVE_CHARS_PROMPT:]
        if save_out and MAX_SAVE_CHARS_OUTPUT > 0 and len(save_out) > MAX_SAVE_CHARS_OUTPUT:
            save_out = save_out[-MAX_SAVE_CHARS_OUTPUT:]

        data[ti]["syn_prompt"][k] = save_prompt
        data[ti]["syn_full_output"][k] = save_out
        data[ti]["syn_input"][k] = test_inp[0] if test_inp and isinstance(test_inp[0], str) else ""
        data[ti]["syn_output"][k] = test_out[0] if test_out and isinstance(test_out[0], str) else ""
        data[ti]["syn_example_text"][k] = example_text[0] if example_text and isinstance(example_text[0], str) else ""

    cprint("[reward rollout] parse done. writing back json ...", "yellow")
    sys.stdout.flush()

    # Write back (atomic)
    atomic_write_json(policy_stage_path, data)

    cprint("[reward rollout] finished.", "green")
    sys.stdout.flush()
    sys.stderr.flush()

    if FORCE_EXIT:
        os._exit(0)

    gc.collect()
