# -*- coding: utf-8 -*-
import os
import re
import json
import random
import time
import multiprocessing as mp
from queue import Empty
from typing import Any, Dict, List

from jinja2 import Template
from termcolor import cprint

from transformers import AutoTokenizer
from omegaconf import OmegaConf


os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 固定：单 Engine + TP=TP_SIZE + chunk_size=CHUNK_SIZE
TP_SIZE = 4
CHUNK_SIZE = 1024


def get_token_lengths(strings, tokenizer):
    return [len(tokenizer.encode(s, add_special_tokens=False)) for s in strings]


############################
# Config
############################
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


############################
# ENV-COEVO state/pending helpers
############################
def get_state_paths(project_name, dataset, node_index, config):
    state_prefix = OmegaConf.select(config, "dataset.dataset_state_prefix", default="dataset_state")
    pending_prefix = OmegaConf.select(config, "dataset.env_pending_prefix", default="env_pending")
    state_path = f"../{project_name}/temp_data/{state_prefix}-{dataset}-node{node_index}.json"
    pending_path = f"../{project_name}/temp_data/{pending_prefix}-{dataset}-node{node_index}.json"
    return state_path, pending_path


def _as_list(x):
    return x if isinstance(x, list) else []


def load_or_init_state(base_dataset_path, state_path, num_node, node_index, config, *, is_code: bool):

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

        if is_code:
            v0 = {
                "version": 0,
                "question": item.get("question", ""),
                "test_input": _as_list(item.get("test_input", [])),
                "test_output": _as_list(item.get("test_output", [])),
                "test_method": item.get("test_method", "stdio"),
                "test_time_limit": item.get("test_time_limit", 1),
                "created_step": 0,
                "source": "init",
            }
            if v0["test_method"] == "function":
                v0["test_list"] = _as_list(item.get("test_list", []))
        else:
            v0 = {
                "version": 0,
                "question": item.get("question", ""),
                "ground_truth_answer": item.get("ground_truth_answer", ""),
                "created_step": 0,
                "source": "init",
            }

        slots.append({"slot_id": slot_id, "active": v0, "history": [v0]})

    state = {"meta": {"dataset": os.path.basename(base_dataset_path), "node_index": int(node_index)}, "slots": slots}
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    return state


def load_pending(pending_path):
    if not os.path.exists(pending_path):
        return []
    with open(pending_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
        return obj if isinstance(obj, list) else []


############################
# Output path helpers
############################
def get_policy_stage_output_path(project_name, outputs_name, num_node, node_index):
    return f"../{project_name}/temp_data/outputs-{int(node_index)}-{outputs_name}.json"


############################
# Extractors
############################
def extract_final_boxed_answer(s: str):
    tag = r"\boxed{"
    start = s.rfind(tag)
    if start == -1:
        return "Can not extract the answer!"
    i = start + len(tag)
    depth = 1
    buf = []
    while i < len(s) and depth:
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1
    return "".join(buf) if depth == 0 else "Can not extract the answer!"


def extract_code(full_output: str):
    matches = re.findall(r"```python(.*?)```", full_output, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return "We can not extract the code in the output. "


############################
# vLLM Worker Pool (copy the style from your “good” code)
############################
def split_prompts(prompts, n):
    k, m = divmod(len(prompts), n)
    return [prompts[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def worker_fn(pretrained_model, gpu_ids, task_queue, result_queue,
              max_model_len, max_generation_token, temp):
    # IMPORTANT: set per-process visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print(f"[vLLM] (worker {gpu_ids}) Loading model: {pretrained_model}", flush=True)

    try:
        import inspect
        import torch  # noqa
        from vllm import LLM, SamplingParams  # noqa

        llm_kwargs = dict(
            model=pretrained_model,
            dtype="bfloat16",
            tensor_parallel_size=len(gpu_ids),
            gpu_memory_utilization=0.85,
            max_model_len=int(max_model_len),
            enforce_eager=False,             
            disable_custom_all_reduce=True, 
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

    def _manual_vllm_shutdown(llm_obj):
        for eng_name in ("llm_engine", "_llm_engine", "_engine", "engine"):
            eng = getattr(llm_obj, eng_name, None)
            if eng is None:
                continue
            me = getattr(eng, "model_executor", None)
            if me is not None and hasattr(me, "shutdown"):
                me.shutdown()
                return True
            if hasattr(eng, "shutdown"):
                eng.shutdown()
                return True
        return False

    while True:
        task = task_queue.get()
        if task == "STOP":
            print(f"[vLLM] (worker {gpu_ids}) STOP received, shutting down...", flush=True)
            break
        task_id, prompts = task
        try:
            outputs = llm.generate(prompts, sampling_params)
            result_texts = [
                out.outputs[0].text if (out.outputs and len(out.outputs) > 0) else ""
                for out in outputs
            ]
            result_queue.put((task_id, result_texts))
        except Exception as e:
            result_queue.put(("ERROR", f"generate failed on GPUs {gpu_ids}: {repr(e)}"))
            break

    try:
        _manual_vllm_shutdown(llm)
    except Exception:
        pass
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
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

    # IMPORTANT: do NOT join_thread() (it can hang forever). cancel it.
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
    desc: str = "policy",
    timeout_s: float = 1800.0,
):
    """
    Same contract as your “good” code:
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


############################
# Main: POLICY ONLY
############################
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  

    config = get_config()
    project_name = config.experiment.project

    # policy_model selection
    if int(config.experiment.current_epoch) == 1:
        policy_model = config.model.policy_model
    else:
        policy_model = "../" + project_name + "/ckpt/" + config.model.optimized_name

    fn = str(config.experiment.function)
    is_train = (fn == "train")
    is_eval = (fn in ("eval", "evaluation", "evaluate", "test"))

    if is_train:
        dataset = config.dataset.train_dataset
        max_model_len = int(config.rollout.policy.model_length)
        max_generation_token = int(config.rollout.policy.max_gen_length)
        temp = float(config.rollout.policy.temperature)
        k_sample = int(config.rollout.policy.num_response_per_task)
        num_node = int(config.experiment.num_node)
        node_index = int(config.experiment.node_index)
        start_with_think = bool(config.rollout.policy.start_with_think)
    else:
        dataset = config.dataset.eval_dataset
        max_model_len = int(config.evaluation.policy.model_length)
        max_generation_token = int(config.evaluation.policy.max_gen_length)
        temp = float(config.evaluation.policy.temperature)
        k_sample = int(config.evaluation.policy.num_response_per_task)
        num_node = int(config.experiment.num_node)
        node_index = int(config.experiment.node_index)
        start_with_think = bool(config.evaluation.policy.if_start_with_think)

    # system prompt
        
    system_prompts = (
        "<|im_start|>You are a helpful assistant help user solve problems. <|im_end|>\n"
        "<|im_start|>User: You need to think first then write python script. You should use input() to input and print() to output in your script."
        "This is the problem:\n{{problem}} <|im_end|>\n<|im_start|>Assistant: "
    )
    if is_eval:
        system_prompts = (
            "<|im_start|>You are a helpful assistant help user solve problems. <|im_end|>\n"
            "<|im_start|>User: You need to think first then write python script. You should use input() to input and print() to output in your script. Your code should output the results based on the input read in, rather than generating the given test example."
            "This is the problem:\n{{problem}} <|im_end|>\n<|im_start|>Assistant: "
        )
    

    if start_with_think:
        system_prompts = system_prompts + "<think>"

    # ===== ENV-COEVO: build this round's data from state + pending =====
    base_dataset_path = "../data/" + dataset + ".json"
    state_path, pending_path = get_state_paths(project_name, dataset, node_index, config)

    state = load_or_init_state(
        base_dataset_path, state_path, num_node, node_index, config, is_code=True
    )
    slots = state["slots"]

    pending = load_pending(pending_path)
    pending_slot_ids = {p.get("slot_id") for p in pending if "slot_id" in p}

    if is_train:
        num_task_total = int(config.rollout.policy.num_task)
        num_task = max(1, int(num_task_total / max(1, num_node)))
    else:
        num_task = len(slots)

    candidate_slots = [s for s in slots if s["slot_id"] not in pending_slot_ids]
    random.shuffle(candidate_slots)
    normal_slots = candidate_slots[: min(num_task, len(candidate_slots))]

    data: List[Dict[str, Any]] = []
    for s in normal_slots:
        active = s["active"]
        item = {
            "source": "dataset",
            "slot_id": s["slot_id"],
            "active_version": active.get("version", 0),
            "question": active.get("question", ""),
            "test_input": _as_list(active.get("test_input", [])),
            "test_output": _as_list(active.get("test_output", [])),
            "test_method": active.get("test_method", "stdio"),
            "test_time_limit": active.get("test_time_limit", 1),
        }
        if item["test_method"] == "function":
            item["test_list"] = _as_list(active.get("test_list", []))
        data.append(item)
        

    # pending(temp) tasks
    if is_train:
        for p in pending:
            data.append(
                {
                    "source": "temp",
                    "slot_id": p.get("slot_id"),
                    "goal": p.get("goal"),
                    "prev_acc": p.get("prev_acc"),
                    "created_step": p.get("created_step"),
                    "env_prompt": p.get("env_prompt"),
                    "env_full_output": p.get("env_full_output"),
                    "question": p.get("question", ""),
                    "test_input": _as_list(p.get("test_input", [])),
                    "test_output": _as_list(p.get("test_output", [])),
                    "test_method": p.get("test_method", "stdio"),
                    "test_time_limit": p.get("test_time_limit", 1),
                }
            )
            

    outputs_name = ("rl-" if is_train else "eval-") + policy_model.replace("/", ".") + "-" + dataset
    num = len(data)

    # ===== build prompts (same as before) =====
    tokenizer = AutoTokenizer.from_pretrained(policy_model, trust_remote_code=True)

    def build_prompt(problem_text: str) -> str:
        return Template(system_prompts).render(problem=problem_text)

    generation_prompts: List[str] = []
    index_list: List[int] = []

    for i in range(num):
        pmt = build_prompt(data[i]["question"])
        data[i]["prompt"] = pmt
        data[i]["full_output"] = []
        data[i]["extracted_output"] = []
        data[i]["response_length"] = []
        generation_prompts.extend([pmt] * k_sample)
        index_list.extend([i] * k_sample)

    # ===== vLLM policy rollout (UPDATED: worker pool + timeout, algorithm unchanged) =====
    # We only build ONE gpu group of size TP_SIZE (same “single engine TP” behavior as your original).
    # This still inherits the robustness: isolated process + timeout + killable worker.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        visible = len([x for x in cvd.split(",") if x.strip() != ""])
    else:
        # fallback only; avoids depending on torch in most cases
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

    gpu_groups = [list(range(TP_SIZE))]  # single worker, TP engine

    cprint(f"[vLLM] start worker(s) for model: {policy_model}", "green")
    task_queues, result_queues, processes = start_workers(
        policy_model, gpu_groups, max_model_len, max_generation_token, temp
    )

    try:
        cprint("start generation...", "green")

        N = len(generation_prompts)
        shuffled_idx = list(range(N))
        random.shuffle(shuffled_idx)
        shuffled_prompts = [generation_prompts[i] for i in shuffled_idx]

        shuffled_outputs: List[str] = []
        cs = int(CHUNK_SIZE) if CHUNK_SIZE is not None else 0
        if cs <= 0:
            cs = len(shuffled_prompts)

        # chunked submission (keeps your original batching idea; now robust with timeout)
        for s in range(0, len(shuffled_prompts), cs):
            sub = shuffled_prompts[s:s + cs]
            print(f"[POLICY] batch {s}..{min(s + cs, len(shuffled_prompts))} (n={len(sub)})", flush=True)

            # per-chunk timeout can be tuned; keep a generous default
            outs = generate_results(
                sub,
                gpu_groups,
                task_queues,
                result_queues,
                desc="policy",
                timeout_s=1800.0,
            )
            shuffled_outputs.extend(outs)

        restored_outputs = [None] * N
        for out, idx in zip(shuffled_outputs, shuffled_idx):
            restored_outputs[idx] = out

        cprint("generation job done!", "green")

    finally:
        # always stop workers (even if timeout/error)
        stop_workers(task_queues, result_queues, processes)

        try:
            import torch
            time.sleep(2)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass

    # ===== postprocess outputs (same as before) =====
    response_length = get_token_lengths(restored_outputs, tokenizer)

    for i, full_output in enumerate(restored_outputs):
        extracted_output = extract_code(full_output)

        index_i = index_list[i]
        data[index_i]["full_output"].append(full_output)
        data[index_i]["extracted_output"].append(extracted_output)
        data[index_i]["response_length"].append(response_length[i])

    policy_stage_path = get_policy_stage_output_path(project_name, outputs_name, num_node, node_index)
    os.makedirs(os.path.dirname(policy_stage_path), exist_ok=True)
    with open(policy_stage_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    cprint(f"[policy] saved policy-stage file: {policy_stage_path}", "cyan")

    # keep your hard-exit behavior
    import os as _os
    _os._exit(0)
