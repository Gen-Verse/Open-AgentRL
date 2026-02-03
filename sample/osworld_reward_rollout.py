

import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import ast
import json
import copy
import argparse
from typing import List, Dict, Any, Tuple

import multiprocessing as mp
import torch
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


def parse_gpu_groups_any(spec: Any) -> List[List[int]]:

    if spec is None:
        return [list(range(torch.cuda.device_count()))]

    if isinstance(spec, list):
        groups = spec
    elif isinstance(spec, str):
        s = spec.strip()
        if not s:
            return [list(range(torch.cuda.device_count()))]
        try:
            groups = json.loads(s)
        except Exception:
            groups = ast.literal_eval(s)
    else:
        raise ValueError(f"--gpu-groups not support: {type(spec)}")

    if not isinstance(groups, list) or not all(isinstance(g, list) for g in groups):
        raise ValueError("--gpu-groups must be 2d list, like [[0,1,2,3],[4,5,6,7]]")

    norm = []
    for g in groups:
        if not all(isinstance(x, int) for x in g):
            raise ValueError("--gpu-groups elements should be GPU id")
        if len(g) == 0:
            continue
        norm.append(g)

    if not norm:
        norm = [list(range(torch.cuda.device_count()))]
    return norm


def extract_final_boxed_answer(s: str) -> str:
    tag = r'\boxed{'
    start = s.rfind(tag)
    if start == -1:
        return "Can not extract the answer!"
    i = start + len(tag)
    depth = 1
    buf = []
    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                break
        buf.append(ch)
        i += 1
    return ''.join(buf) if depth == 0 else "Can not extract the answer!"


def normalize_reward_from_response(resp: str) -> int:
    boxed = extract_final_boxed_answer(resp or "")
    if boxed == "Can not extract the answer!":
        return 0
    val = boxed.strip()
    m = re.fullmatch(r'[+\-]?1', val)
    if not m:
        return 0
    try:
        v = int(val)
    except Exception:
        return 0
    return v if v in (1, -1) else 0


def extract_boxed_or_fallback(resp: str) -> str:
    boxed = extract_final_boxed_answer(resp or "")
    if boxed and boxed != "Can not extract the answer!":
        return boxed.strip()
    return ""


def _resolve_local_images(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    msgs = copy.deepcopy(messages)
    for m in msgs:
        if "content" not in m:
            continue
        for chunk in m["content"]:
            if isinstance(chunk, dict) and chunk.get("type") == "image":
                img_val = chunk.get("image")
                if isinstance(img_val, str) and os.path.exists(img_val):
                    try:
                        chunk["image"] = Image.open(img_val).convert("RGB")
                    except Exception:
                        pass
    return msgs


def _list_imgs(messages):
    imgs = []
    for m in messages:
        for c in (m.get("content") or []):
            if isinstance(c, dict) and c.get("type") == "image":
                imgs.append(c.get("image"))
    return imgs


def prepare_inputs_for_vllm(messages: List[Dict[str, Any]], processor: AutoProcessor) -> Dict[str, Any]:

    messages2 = _resolve_local_images(messages)
    text = processor.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
    patch_size = getattr(getattr(processor, "image_processor", object()), "patch_size", 14)
    try:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages2, image_patch_size=patch_size, return_video_kwargs=True, return_video_metadata=True
        )
    except TypeError:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages2, image_patch_size=patch_size, return_video_kwargs=True
        )

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    return {"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}


def make_text_messages(text: str) -> List[Dict[str, Any]]:
    # Qwen-VL chat format: content is a list of chunks
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]



def is_dir(p: str) -> bool:
    return os.path.isdir(p)


def list_subdirs(p: str) -> List[str]:
    try:
        return [d for d in sorted(os.listdir(p)) if is_dir(os.path.join(p, d))]
    except FileNotFoundError:
        return []


def list_numeric_dirs(p: str) -> List[str]:

    try:
        subs = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
    except FileNotFoundError:
        return []
    if subs and all(s.isdigit() for s in subs):
        return sorted(subs, key=lambda x: int(x))
    return sorted(subs)


def load_examples_with_kinds(path: str):
    """
    Returns:
      wanted_ids: set[str]                      
      active_pairs: set[tuple[str,str]]         
      temp_pairs: set[tuple[str,str]]
      active_ids_wo_domain: set[str]            
      temp_ids_wo_domain: set[str]
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    wanted_ids = set()
    active_pairs = set()
    temp_pairs = set()
    active_ids_wo_domain = set()
    temp_ids_wo_domain = set()

    def _add_item(x, bucket_pairs, bucket_ids_wo_domain):
        if isinstance(x, str):
            wanted_ids.add(x)
            bucket_ids_wo_domain.add(x)
        elif isinstance(x, dict):
            eid = x.get("example_id") or x.get("example")
            dom = x.get("domain")
            if eid is None:
                return
            eid = str(eid)
            wanted_ids.add(eid)
            if isinstance(dom, str) and dom:
                bucket_pairs.add((dom, eid))
            else:
                bucket_ids_wo_domain.add(eid)

    if isinstance(obj, list):

        for x in obj:
            _add_item(x, active_pairs, active_ids_wo_domain)
        return wanted_ids, active_pairs, temp_pairs, active_ids_wo_domain, temp_ids_wo_domain

    if isinstance(obj, dict):
        for x in (obj.get("active") or []):
            _add_item(x, active_pairs, active_ids_wo_domain)
        for x in (obj.get("temp") or []):
            _add_item(x, temp_pairs, temp_ids_wo_domain)

        return wanted_ids, active_pairs, temp_pairs, active_ids_wo_domain, temp_ids_wo_domain

    raise ValueError(f"examples-json only support list or dict, but {type(obj)}")



def sort_traj_list_inplace(traj_list: List[Dict[str, Any]]):

    def _key(x):
        v = x.get("run_id")
        if isinstance(v, int):
            return (0, v)
        try:
            return (0, int(v))
        except Exception:
            return (1, str(v))
    traj_list.sort(key=_key)


def stop_workers(task_queues: List[mp.Queue],
                 result_queues: List[mp.Queue],
                 processes: List[mp.Process]):
    for q in task_queues:
        try:
            q.put("STOP")
        except Exception:
            pass

    for q in task_queues:
        try:
            q.close()
            q.join_thread()
        except Exception:
            pass
    for rq in result_queues:
        try:
            rq.close()
            rq.join_thread()
        except Exception:
            pass

    for p in processes:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
            p.join(timeout=10)


def split_even(items: List[Any], n: int) -> List[List[Any]]:
    k, m = divmod(len(items), n)
    return [items[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]





def worker_fn(pretrained_model: str,
              gpu_ids: List[int],
              task_queue: mp.Queue,
              result_queue: mp.Queue,
              dtype: str,
              gpu_mem: float,
              max_model_len: int,
              temperature: float,
              top_p: float,
              max_tokens: int,
              stop_words: List[str]):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    from vllm import LLM, SamplingParams

    print(
        f"[Worker-{gpu_ids}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
        f"torch.cuda.device_count()={torch.cuda.device_count()} expected={len(gpu_ids)}",
        flush=True
    )
    assert torch.cuda.device_count() == len(gpu_ids), \
        f"CUDA_VISIBLE_DEVICES not applied early enough: got {torch.cuda.device_count()} vs {len(gpu_ids)}"

    print(f"[Worker-{gpu_ids}] Loading model...", flush=True)
    processor = AutoProcessor.from_pretrained(pretrained_model)
    llm = LLM(
        model=pretrained_model,
        trust_remote_code=True,
        dtype=dtype,
        tensor_parallel_size=len(gpu_ids),
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len if max_model_len > 0 else None,
        enforce_eager=False,
    )
    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop=stop_words,
    )

    while True:
        task = task_queue.get()
        if task == "STOP":
            print(f"[Worker-{gpu_ids}] Stopping...", flush=True)
            break

        task_id, msg_chunk = task
        if not msg_chunk:
            result_queue.put((task_id, []))
            continue

        prep_cache: Dict[int, Dict[str, Any]] = {}
        batch_inputs = []
        for messages in msg_chunk:
            k = id(messages)
            inp = prep_cache.get(k)
            if inp is None:
                inp = prepare_inputs_for_vllm(messages, processor)
                prep_cache[k] = inp
            batch_inputs.append(inp)

        outs = llm.generate(batch_inputs, sampling_params=sampling)
        texts = [o.outputs[0].text if o.outputs else "" for o in outs]
        result_queue.put((task_id, texts))


def start_workers(pretrained_model: str,
                  gpu_groups: List[List[int]],
                  dtype: str,
                  gpu_mem: float,
                  max_model_len: int,
                  temperature: float,
                  top_p: float,
                  max_tokens: int,
                  stop_words: List[str]):
    task_queues, result_queues, procs = [], [], []
    for gpu_ids in gpu_groups:
        tq = mp.Queue()
        rq = mp.Queue()
        p = mp.Process(
            target=worker_fn,
            args=(pretrained_model, gpu_ids, tq, rq, dtype, gpu_mem,
                  max_model_len, temperature, top_p, max_tokens, stop_words)
        )
        p.start()
        task_queues.append(tq)
        result_queues.append(rq)
        procs.append(p)
    return task_queues, result_queues, procs


def generate_results(messages_all: List[List[Dict[str, Any]]],
                     R: int,
                     gpu_groups: List[List[int]],
                     task_queues: List[mp.Queue],
                     result_queues: List[mp.Queue],
                     procs: List[mp.Process]) -> List[str]:

    chunks = split_even(messages_all, len(gpu_groups))

    jobs = []
    for i, (q, chunk) in enumerate(zip(task_queues, chunks)):
        if not chunk:
            continue

        chunk_expanded = []
        for messages in chunk:
            chunk_expanded.extend([messages] * R)

        q.put((i, chunk_expanded))
        jobs.append(i)

    results_by_job = {}
    remaining = set(jobs)
    while remaining:
        for p in procs:
            if not p.is_alive():
                raise RuntimeError(f"Worker died pid={p.pid}, exitcode={p.exitcode}")

        for i, rq in enumerate(result_queues):
            if i not in remaining:
                continue
            try:
                task_id, result = rq.get(timeout=0.1)
            except Exception:
                continue
            results_by_job[task_id] = result
            remaining.remove(task_id)

    flat = []
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        flat.extend(results_by_job.get(i, []))
    return flat



def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--examples-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-8B-Thinking")


    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--gpu-mem", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=40960, help="<=0")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--stop-words", type=str,
                        default="</answer>,User:,Human:,Assistant:,<|im_end|>,<|endoftext|>")

    parser.add_argument("--gpu-groups", default=None,
                        help="example '[[0,1,2,3],[4,5,6,7]]'")

    parser.add_argument("--num-rollout-per-query", type=int, default=1)

    # NEW: summary generation controls
    parser.add_argument("--summary-temperature", type=float, default=0.2,
                        help="summary temperature")
    parser.add_argument("--summary-top-p", type=float, default=1.0)
    parser.add_argument("--summary-max-tokens", type=int, default=256)
    parser.add_argument("--no-summary", action="store_true",
                        help="stop step/trajectory summary generation")

    parser.add_argument("--download_proxy", type=str)

    args = parser.parse_args()

    if args.download_proxy:
        os.environ["HTTP_PROXY"] = args.download_proxy
        os.environ["HTTPS_PROXY"] = args.download_proxy

    # 1) read example id list
    wanted_examples, active_pairs, temp_pairs, active_ids_wo_domain, temp_ids_wo_domain = \
    load_examples_with_kinds(args.examples_json)

    # 2) scan tasks
    domains = list_subdirs(args.root_dir)
    tasks: List[Tuple[str, str, str, str]] = []  # (domain, example, run, traj_json)
    for domain in domains:
        domain_dir = os.path.join(args.root_dir, domain)
        examples = list_subdirs(domain_dir)
        for ex in examples:
            if ex not in wanted_examples:
                continue
            ex_dir = os.path.join(domain_dir, ex)
            runs = list_numeric_dirs(ex_dir)
            for run in runs:
                traj_json = os.path.join(ex_dir, run, "trajectory.json")
                if os.path.isfile(traj_json):
                    tasks.append((domain, ex, run, traj_json))

    if active_ids_wo_domain or temp_ids_wo_domain:
        for (domain, ex, _run, _traj_json) in tasks:
            if ex in active_ids_wo_domain:
                active_pairs.add((domain, ex))
            if ex in temp_ids_wo_domain:
                temp_pairs.add((domain, ex))


    out_index: Dict[Tuple[str, str], int] = {}
    result: List[Dict[str, Any]] = []
    for domain, ex, _run, _ in tasks:
        key = (domain, ex)
        if key not in out_index:
            out_index[key] = len(result)
            result.append({"domain": domain, "example": ex, "trajctory_list": []})

    # 4) read data，create pending
    def normalize_reward_trajectory(rt_any: Any) -> List[List[Dict[str, Any]]]:
        if not isinstance(rt_any, list):
            if isinstance(rt_any, dict) and "role" in rt_any:
                return [[rt_any]]
            raise ValueError("reward_trajectory must be list or messages(list)")
        if len(rt_any) == 0:
            return []
        if all(isinstance(x, dict) and "reward_messages" in x for x in rt_any):
            return [x["reward_messages"] for x in rt_any]
        if all(isinstance(x, list) for x in rt_any):
            return rt_any
        if isinstance(rt_any[0], dict) and "role" in rt_any[0]:
            return [rt_any]
        out = []
        for x in rt_any:
            if isinstance(x, list):
                out.append(x)
            elif isinstance(x, dict) and "reward_messages" in x and isinstance(x["reward_messages"], list):
                out.append(x["reward_messages"])
        if out:
            return out
        raise ValueError("can not extract reward_trajectory structure")

    pending: List[Tuple[int, int, int, List[Dict[str, Any]]]] = []
    for domain, ex, run, traj_json in tasks:
        try:
            with open(traj_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] fail to open, skip {traj_json} | reason: {e}")
            continue

        meta_result = (data.get("meta") or {}).get("result")
        rt = data.get("reward_trajectory", [])
        try:
            messages_list = normalize_reward_trajectory(rt)
        except Exception as e:
            print(f"[WARN] open reward_trajectory failed, skip {traj_json} | reason: {e}")
            continue

        domain_idx = out_index[(domain, ex)]
        trajctory_list = result[domain_idx]["trajctory_list"]
        this_traj_idx = len(trajctory_list)
        run_id = int(run) if str(run).isdigit() else run

        trajctory_list.append({
            "run_id": run_id,
            "message": messages_list,
            "response": [[] for _ in range(len(messages_list))],
            "extracted_reward": [[] for _ in range(len(messages_list))],
            "process_reward": [0.0 for _ in range(len(messages_list))],
            "result": meta_result,

            # NEW: only negative steps recorded in step_summary
            "step_summary": [],        # [{ "step": k, "summary": "..." }, ...]
            "trajectory_summary": "",  # string
        })

        for msg_idx, messages in enumerate(messages_list):
            pending.append((domain_idx, this_traj_idx, msg_idx, messages))

    if not pending:
        for item in result:
            sort_traj_list_inplace(item["trajctory_list"])
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[DONE] no task, {args.output_json}")
        return


    gpu_groups = parse_gpu_groups_any(args.gpu_groups)
    R = max(1, int(args.num_rollout_per_query))
    print(f"[INFO] Launch {len(gpu_groups)} engines: {gpu_groups} | R={R}")

    stop_words = [w for w in args.stop_words.split(",") if w]
    task_queues, result_queues, procs = start_workers(
        pretrained_model=args.model,
        gpu_groups=gpu_groups,
        dtype=args.dtype,
        gpu_mem=args.gpu_mem,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop_words=stop_words,
    )

    # 6) reward inference
    messages_all = [m for (_d, _t, _m, m) in pending]
    flat_texts = generate_results(messages_all, R, gpu_groups, task_queues, result_queues, procs)

    # 7) reward response / extracted_reward / process_reward
    pos = 0
    for (domain_idx, traj_idx, msg_idx, _messages) in pending:
        texts = flat_texts[pos:pos + R]
        pos += R

        result[domain_idx]["trajctory_list"][traj_idx]["response"][msg_idx] = texts
        rewards = [normalize_reward_from_response(t) for t in texts]
        result[domain_idx]["trajctory_list"][traj_idx]["extracted_reward"][msg_idx] = rewards
        mean_val = float(sum(rewards) / len(rewards)) if rewards else 0.0
        result[domain_idx]["trajctory_list"][traj_idx]["process_reward"][msg_idx] = mean_val


    stop_workers(task_queues, result_queues, procs)

    for item in result:
        sort_traj_list_inplace(item["trajctory_list"])

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[DONE] write results to: {args.output_json}")


if __name__ == "__main__":
    main()
