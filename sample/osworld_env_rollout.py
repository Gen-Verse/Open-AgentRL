

from __future__ import annotations

import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import ast
import inspect
from typing import Any, Dict, List, Tuple, Optional

from jinja2 import Template
from termcolor import cprint

import torch
from transformers import AutoTokenizer  # kept for parity with your template
from omegaconf import OmegaConf


########################################
# Config (must follow your style)
########################################
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


########################################
# vLLM inference (single engine, TP)
########################################
def _to_pylist(x: Any) -> Any:
    """OmegaConf ListConfig -> python list recursively; otherwise passthrough."""
    try:
        from omegaconf import ListConfig, DictConfig
        if isinstance(x, ListConfig):
            return [_to_pylist(v) for v in list(x)]
        if isinstance(x, DictConfig):
            return {str(k): _to_pylist(v) for k, v in dict(x).items()}
    except Exception:
        pass
    if isinstance(x, list):
        return [_to_pylist(v) for v in x]
    return x


def infer_tp_size_from_cfg(cfg) -> int:
    """
    Prefer tp_size = len(cfg.rollout.environment.gpu_groups[0]) if provided.
    Else use cfg.rollout.environment.tp_size if exists.
    Else fallback 8.
    """
    try:
        gg = OmegaConf.select(cfg, "rollout.environment.gpu_groups", default=None)
        gg = _to_pylist(gg)
        if isinstance(gg, list) and gg and isinstance(gg[0], list) and gg[0]:
            return int(len(gg[0]))
    except Exception:
        pass

    try:
        tp = OmegaConf.select(cfg, "rollout.environment.tp_size", default=None)
        if tp is not None:
            return int(tp)
    except Exception:
        pass

    return 8


def generate_results_single_engine(
    llm,
    sampling_params,
    prompts_all: List[str],
    R: int,
    chunk_size: int,
    tag: str,
) -> List[str]:

    if not prompts_all:
        return []

    out_texts: List[str] = []
    N = len(prompts_all)

    for s in range(0, N, chunk_size):
        sub = prompts_all[s:s + chunk_size]

        expanded: List[str] = []
        for p in sub:
            expanded.extend([p] * R)

        print(f"[GEN:{tag}] batch {s}..{min(s + chunk_size, N)} (expanded={len(expanded)})", flush=True)
        outs = llm.generate(expanded, sampling_params=sampling_params)
        texts = [o.outputs[0].text if (o.outputs and len(o.outputs) > 0) else "" for o in outs]
        out_texts.extend(texts)

    return out_texts


########################################
# Utilities
########################################
def safe_success_flag(result_val: Any) -> int:
    """result > 0 => success(1), else 0"""
    try:
        return 1 if float(result_val) > 0 else 0
    except Exception:
        return 0


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def dump_json(path: str, obj: Any) -> None:
    ensure_dir_for_file(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def index_rollout_preacc(rl_rollout_results_json: str) -> Dict[Tuple[str, str], float]:
    """
    rl_rollout_results.json format:
      [
        {"domain": d, "example": e, "trajctory_list":[{"run_id":..., "result":...}, ...]},
        ...
      ]
    return dict[(domain, example)] = pre_acc
    """
    data = load_json(rl_rollout_results_json)
    out: Dict[Tuple[str, str], float] = {}
    for item in data:
        d = item.get("domain")
        e = item.get("example")
        if not d or not e:
            continue
        tl = item.get("trajctory_list", []) or []
        flags = [safe_success_flag(t.get("result")) for t in tl]
        out[(d, e)] = (sum(flags) / len(flags)) if flags else 0.0
    return out


def _sort_key_run_id(v: Any):
    if isinstance(v, int):
        return (0, v)
    try:
        return (0, int(v))
    except Exception:
        return (1, str(v))


def _clip(s: str, n: int) -> str:
    s2 = (s or "").strip().replace("\n", " ")
    return s2 if len(s2) <= n else (s2[:n] + "...")


def index_reward_trajectory_summaries(
    reward_rollout_json: str,
    max_traj: int = 12,
    max_summary_chars: int = 320,
) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    """
      [
        {"domain": d, "example": e, "trajctory_list":[
            {
              "run_id": ...,
              "trajectory_summary": "...",          # NEW key (preferred)
              "error_trajectory_summary": "...",    # legacy fallback
              ...
            }, ...
        ]},
        ...
      ]

    Return:
      (domain, example) -> [
         {"run_id":..., "trajectory_summary": "..."},
         ...
      ]
    """
    if not reward_rollout_json or (not os.path.isfile(reward_rollout_json)):
        return {}

    try:
        data = load_json(reward_rollout_json)
    except Exception:
        return {}

    out: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    if not isinstance(data, list):
        return out

    for item in data:
        d = item.get("domain")
        e = item.get("example")
        if not d or not e:
            continue

        tl = item.get("trajctory_list", []) or []
        if not isinstance(tl, list):
            continue

        rows: List[Dict[str, Any]] = []
        for t in tl:
            if not isinstance(t, dict):
                continue
            run_id = t.get("run_id")

            traj_sum = t.get("trajectory_summary")
            if not isinstance(traj_sum, str) or not traj_sum.strip():
                traj_sum = t.get("error_trajectory_summary")  # legacy
            if not isinstance(traj_sum, str) or not traj_sum.strip():
                continue

            rows.append({
                "run_id": run_id,
                "trajectory_summary": _clip(str(traj_sum), max_summary_chars),
            })

        if not rows:
            continue

        rows.sort(key=lambda r: _sort_key_run_id(r.get("run_id")))
        out[(d, e)] = rows[:max_traj]

    return out


def normalize_manifest_obj(manifest_obj: Any, id2domain: Dict[str, str], infer_domain_fn) -> Dict[str, List[Dict[str, str]]]:
    """
    Supports:
      - legacy list[str] -> treated as active only; upgraded to {"active":[...],"temp":[]}
      - dict {"active": [...], "temp": [...]}
    Each element can be either:
      - {"domain":..,"example_id":..}
      - "example_id"  (domain inferred from id2domain, else infer_domain_fn)
    """
    def _norm_list(lst: Any) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if not isinstance(lst, list):
            return out
        for x in lst:
            if isinstance(x, dict) and ("domain" in x) and ("example_id" in x):
                out.append({"domain": str(x["domain"]), "example_id": str(x["example_id"])})
            elif isinstance(x, str):
                d = id2domain.get(x)
                if d is None:
                    d = infer_domain_fn(x)
                if d is not None:
                    out.append({"domain": d, "example_id": x})
        return out

    if isinstance(manifest_obj, list):
        return {"active": _norm_list(manifest_obj), "temp": []}

    if isinstance(manifest_obj, dict):
        return {
            "active": _norm_list(manifest_obj.get("active", [])),
            "temp": _norm_list(manifest_obj.get("temp", [])),
        }

    return {"active": [], "temp": []}


def list_example_node_files(manifest_dir: str) -> List[str]:
    """Find example_node_*.json under manifest_dir."""
    if not os.path.isdir(manifest_dir):
        return []
    out = []
    for fn in sorted(os.listdir(manifest_dir)):
        if re.fullmatch(r"example_node_\d+\.json", fn):
            out.append(os.path.join(manifest_dir, fn))
    return out


def extract_last_json_obj(text: str) -> Optional[dict]:
    """
    Extract the LAST JSON object from model output (reasoning models often emit multiple).
    Returns a dict if found, else None.
    """
    if not isinstance(text, str):
        return None
    s = text.strip()

    # 1) direct parse
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    # 2) scan from end: find last balanced {...} outside quotes
    end = s.rfind("}")
    while end != -1:
        depth = 0
        in_str = False
        esc = False

        for i in range(end, -1, -1):
            ch = s[i]

            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue

            if ch == "}":
                depth += 1
            elif ch == "{":
                depth -= 1
                if depth == 0:
                    cand = s[i:end + 1].strip()

                    try:
                        obj = json.loads(cand)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass

                    try:
                        obj = ast.literal_eval(cand)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        pass

                    break

        end = s.rfind("}", 0, end)

    return None


def normalize_current_task_obj(obj: dict) -> Optional[dict]:
    """
    Return the actual current_task dict mapping evaluator_name -> instruction.
    Must be a dict with exactly one key.
    Accepts:
      - {"evaluatorX": "..."}
      - {"current_task": {"evaluatorX": "..."}}
    """
    if not isinstance(obj, dict):
        return None

    ct = obj["current_task"] if ("current_task" in obj and isinstance(obj["current_task"], dict)) else obj
    if not isinstance(ct, dict) or len(ct) != 1:
        return None

    k = next(iter(ct.keys()))
    v = ct[k]
    if not isinstance(k, str) or not isinstance(v, str):
        return None
    if not v.strip():
        return None
    return {k: v.strip()}


########################################
# Prompting (NEW: include previous trajectory summaries)
########################################
SYSTEM_PROMPT = """<|im_start|>You are a helpful assistant. <|im_end|>
<|im_start|>user
You will help me adjust the difficulty of an OSWorld/desktop task.

You are given:
(1) current_task (JSON)
(2) task_template: a JSON object mapping evaluator_name -> a canonical instruction for that evaluator.
(3) previous_rollout_trajectory_summaries: OPTIONAL historical error analyses from earlier rollouts.
    - A task may have multiple trajectories (runs).
    - Each trajectory_summary is already deduplicated across steps (no repeated same error across steps).

Goal: make the task {{goal}}.

Rules:
- You MAY switch to a different evaluator from task_template (by changing the key), OR keep the same evaluator.
- You MAY rewrite the instruction to increase/decrease hint strength (add hints to make easier, remove hints to make harder).
- The instruction can NOT be too long.
- You MUST NOT change the essential task meaning compared to the chosen evaluator's template. Do NOT invent a new task.
- Do NOT invent new evaluator names. The output key must be one of the keys in task_template.
- You SHOULD use previous_rollout_trajectory_summaries to guide how you adjust difficulty:
  - If goal is EASIER: add minimal, targeted clarifying hints addressing recurring failure modes.
  - If goal is HARDER: remove such hints, but still keep the same essential task and stay within the chosen evaluator template.
- Output MUST be valid JSON ONLY (no markdown, no extra text).
- Output format MUST be the new current_task JSON object with EXACTLY ONE key:
  {"evaluatorX": "your rewritten instruction"}
- If you accidentally output other text, ensure the FINAL output segment is the JSON object.

current_task:
{{current_task_json}}

task_template:
{{task_template_json}}

previous_rollout_trajectory_summaries:
{{traj_summaries_json}}
<|im_end|>
<|im_start|>assistant
"""


def build_prompt(goal: str, current_task: dict, task_template: dict, traj_summaries: Optional[List[Dict[str, Any]]]) -> str:
    ts = traj_summaries if traj_summaries is not None else []
    return Template(SYSTEM_PROMPT).render(
        goal=goal,
        current_task_json=json.dumps(current_task, ensure_ascii=False, indent=2),
        task_template_json=json.dumps(task_template, ensure_ascii=False, indent=2),
        traj_summaries_json=json.dumps(ts, ensure_ascii=False, indent=2),
    )


########################################
# Main
########################################
if __name__ == "__main__":
    cfg = get_config()

    # ---- model / rollout keys (your updated keys) ----
    pretrained_model = str(cfg.model.environment_model)
    max_model_len = int(cfg.rollout.environment.model_length)
    max_generation_token = int(cfg.rollout.environment.max_gen_length)
    temperature = float(cfg.rollout.environment.temperature)
    num_rollout_per_query = int(getattr(cfg.rollout.environment, "num_rollout_per_query", 1))
    num_rollout_per_query = max(1, num_rollout_per_query)

    # batching (single engine)
    chunk_size = int(getattr(cfg.rollout.environment, "chunk_size", 1024))
    chunk_size = max(1, chunk_size)

    harder_threshold = 0.8
    easier_threshold = 0.2

    # ---- base paths (your required absolute rules) ----
    BASE_DIR = str(cfg.system.rl_base_dir).rstrip("/")
    project_name = str(cfg.experiment.project)

    rl_rollout_results_json = os.path.join(BASE_DIR, project_name, "temp_data", "rl_rollout_results.json")
    manifest_dir = os.path.join(BASE_DIR, project_name, "temp_data")

    eval_base_dir = os.path.join(BASE_DIR, "OSWorld-main", "evaluation_examples")
    train_root = os.path.join(eval_base_dir, project_name, "train")
    temp_root = os.path.join(eval_base_dir, project_name, "temp")

    # ---- NEW: reward rollout summary file candidates ----
    reward_rollout_candidates = [
        os.path.join(BASE_DIR, project_name, "temp_data", "rl_reward_rollout_results.json"),
        os.path.join(BASE_DIR, project_name, "temp_data", "reward_rollout_results.json"),
    ]
    reward_rollout_json = next((p for p in reward_rollout_candidates if os.path.isfile(p)), "")

    cprint(f"[PATH] rl_rollout_results_json: {rl_rollout_results_json}", "green")
    cprint(f"[PATH] manifest_dir:          {manifest_dir}", "green")
    cprint(f"[PATH] train_root:            {train_root}", "green")
    cprint(f"[PATH] temp_root:             {temp_root}", "green")
    cprint(f"[PATH] reward_rollout_json:   {reward_rollout_json if reward_rollout_json else '(not found)'}", "green")
    cprint(f"[CFG ] num_rollout_per_query: {num_rollout_per_query}", "green")
    cprint(f"[CFG ] chunk_size:            {chunk_size}", "green")
    cprint(f"[CFG ] thresholds: harder>={harder_threshold}, easier<={easier_threshold}", "green")

    if num_rollout_per_query > 1 and temperature <= 0:
        cprint(f"[WARN] temperature={temperature} <= 0, multiple samples may be identical.", "yellow")

    if not os.path.isfile(rl_rollout_results_json):
        raise FileNotFoundError(f"rl_rollout_results_json not found: {rl_rollout_results_json}")
    if not os.path.isdir(manifest_dir):
        raise FileNotFoundError(f"manifest_dir not found: {manifest_dir}")
    if not os.path.isdir(train_root):
        raise FileNotFoundError(f"train_root not found: {train_root}")

    # cache for infer_domain
    _infer_cache: Dict[str, Optional[str]] = {}

    def infer_domain_by_train_file(example_id: str) -> Optional[str]:
        if example_id in _infer_cache:
            return _infer_cache[example_id]
        try:
            for d in sorted(os.listdir(train_root)):
                p = os.path.join(train_root, d, f"{example_id}.json")
                if os.path.isfile(p):
                    _infer_cache[example_id] = d
                    return d
        except Exception:
            pass
        _infer_cache[example_id] = None
        return None

    # 1) compute pre_acc from rl_rollout_results
    cprint("[INFO] Indexing pre_acc from rl_rollout_results.json ...", "green")
    preacc_map = index_rollout_preacc(rl_rollout_results_json)

    # build id2domain for legacy manifests (example_id -> domain)
    id2domain = {ex: dom for (dom, ex), _ in preacc_map.items()}

    # 1.5) NEW: load reward trajectory summaries index (if available)
    traj_summary_index: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    if reward_rollout_json:
        cprint("[INFO] Loading reward rollout trajectory summaries ...", "green")
        traj_summary_index = index_reward_trajectory_summaries(
            reward_rollout_json=reward_rollout_json,
            max_traj=12,
            max_summary_chars=10000,
        )
        cprint(f"[INFO] trajectory summary indexed keys: {len(traj_summary_index)}", "green")
    else:
        cprint("[INFO] reward rollout summary file not found; will not feed summaries.", "yellow")

    # 2) load ALL example_node_i.json, aggregate ACTIVE, and remember ownership
    node_files = list_example_node_files(manifest_dir)
    if not node_files:
        raise FileNotFoundError(f"No example_node_*.json found in {manifest_dir}")

    cprint(f"[INFO] Found node manifests: {len(node_files)}", "green")

    # node_path -> normalized manifest dict
    node_manifest: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    # (domain, example_id) -> list of node_paths that contain it in ACTIVE
    owners: Dict[Tuple[str, str], List[str]] = {}

    total_active = 0
    for p in node_files:
        obj = load_json(p)
        norm = normalize_manifest_obj(obj, id2domain, infer_domain_by_train_file)
        node_manifest[p] = norm

        act = norm.get("active", []) or []
        total_active += len(act)
        for it in act:
            k = (it["domain"], it["example_id"])
            owners.setdefault(k, []).append(p)

    active_pairs = sorted(list(owners.keys()))
    if not active_pairs:
        cprint("[DONE] No active tasks found across node manifests.", "yellow")
        raise SystemExit(0)

    cprint(f"[INFO] Active unique tasks: {len(active_pairs)} (raw active entries={total_active})", "green")

    # 3) build prompts (one per ACTIVE task) [NEW: include trajectory summaries]
    prompts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for domain, example_id in active_pairs:
        pre_acc = float(preacc_map.get((domain, example_id), 0.0))
        #goal = "HARDER" if pre_acc >= 0.5 else "EASIER"
        if pre_acc >= harder_threshold:
            goal = "HARDER"
        elif pre_acc <= easier_threshold:
            goal = "EASIER"
        else:
            # skip perturb for mid-accuracy tasks
            continue

        train_path = os.path.join(train_root, domain, f"{example_id}.json")
        if not os.path.isfile(train_path):
            cprint(f"[WARN] Train json missing, skip: {train_path}", "yellow")
            continue

        try:
            ex_obj = load_json(train_path)
        except Exception as e:
            cprint(f"[WARN] Failed to load train json, skip: {train_path} | err={e}", "yellow")
            continue

        current_task = ex_obj.get("current_task")
        task_template = ex_obj.get("task_template")

        if not isinstance(current_task, dict) or not isinstance(task_template, dict) or len(task_template) == 0:
            cprint(f"[WARN] Invalid current_task/task_template, skip: {train_path}", "yellow")
            continue

        # NEW: attach per-trajectory summaries if available
        traj_summ = traj_summary_index.get((domain, example_id))

        prompt = build_prompt(
            goal=goal,
            current_task=current_task,
            task_template=task_template,
            traj_summaries=traj_summ,
        )

        prompts.append(prompt)
        metas.append({
            "domain": domain,
            "example_id": example_id,
            "pre_acc": pre_acc,
            "goal": goal,
            "train_path": train_path,
            "train_obj": ex_obj,
            "owner_nodes": owners.get((domain, example_id), []),
        })

    if not prompts:
        #cprint("[DONE] No valid tasks to perturb (after loading train json).", "yellow")
        #raise SystemExit(0)
        cprint("[DONE] No tasks meet thresholds; clearing temp lists in node manifests.", "yellow")
        # still overwrite temp lists to empty, keep active unchanged
        for np in node_files:
            man = node_manifest.get(np, {"active": [], "temp": []})
            man["temp"] = []
            dump_json(np, man)
        raise SystemExit(0)

    cprint(f"[INFO] Prepared prompts: {len(prompts)}", "green")

    # 4) expand prompts: sample num_rollout_per_query times per task
    prompts_expanded: List[str] = []
    owner_idx: List[int] = []  # expanded prompt -> meta index
    for i, p in enumerate(prompts):
        prompts_expanded.extend([p] * num_rollout_per_query)
        owner_idx.extend([i] * num_rollout_per_query)

    cprint(f"[INFO] Expanded prompts: {len(prompts_expanded)} (R={num_rollout_per_query})", "green")

    # 5) vLLM infer (single engine, TP)
    from vllm import LLM, SamplingParams

    tp_size = infer_tp_size_from_cfg(cfg)
    if torch.cuda.device_count() <= 0:
        raise RuntimeError("No CUDA device visible. Please set CUDA_VISIBLE_DEVICES properly.")
    if tp_size > torch.cuda.device_count():
        raise RuntimeError(f"tp_size={tp_size} > visible GPUs={torch.cuda.device_count()} (check CUDA_VISIBLE_DEVICES)")

    # optional knobs (defaults chosen to match your “reward single-engine” style)
    enforce_eager = bool(getattr(cfg.rollout.environment, "enforce_eager", False))  # False => allow cudagraph
    enable_custom_all_reduce = bool(getattr(cfg.rollout.environment, "enable_custom_all_reduce", False))
    disable_custom_all_reduce = (not enable_custom_all_reduce)

    cprint(
        f"[INIT] Single-engine vLLM | TP={tp_size} enforce_eager={enforce_eager} "
        f"disable_custom_all_reduce={disable_custom_all_reduce} max_model_len={max_model_len}",
        "green",
    )

    llm_kwargs = dict(
        model=pretrained_model,
        dtype="bfloat16",
        tensor_parallel_size=int(tp_size),
        gpu_memory_utilization=0.85,
        max_model_len=int(max_model_len) if int(max_model_len) > 0 else None,
        enforce_eager=bool(enforce_eager),
        disable_custom_all_reduce=bool(disable_custom_all_reduce),
        trust_remote_code=True,
    )
    sig = inspect.signature(LLM.__init__)
    llm_kwargs = {k: v for k, v in llm_kwargs.items() if k in sig.parameters}

    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=float(temperature),
        top_p=0.95,
        top_k=-1,
        min_p=0.0,
        max_tokens=int(max_generation_token),
        stop=["</answer>", "User:", "Human:", "Assistant:", "<|im_end|>", "<|endoftext|>"],
    )

    _ = AutoTokenizer.from_pretrained(pretrained_model)  # parity with your template

    cprint("start generation...", "green")
    outputs_expanded = generate_results_single_engine(
        llm=llm,
        sampling_params=sampling_params,
        prompts_all=prompts,
        R=num_rollout_per_query,
        chunk_size=chunk_size,
        tag="ENV",
    )
    cprint("generation job done!", "green")

    expected = len(prompts) * num_rollout_per_query
    if len(outputs_expanded) != expected:
        raise RuntimeError(f"generation mismatch: got={len(outputs_expanded)} expected={expected}")

    # group outputs back per task
    outputs_by_task: List[List[str]] = [[] for _ in range(len(metas))]
    pos = 0
    for i in range(len(metas)):
        outputs_by_task[i] = outputs_expanded[pos:pos + num_rollout_per_query]
        pos += num_rollout_per_query

    # 6) write temp json files + prepare per-node temp lists
    per_node_temp: Dict[str, List[Dict[str, str]]] = {p: [] for p in node_files}
    n_ok = 0
    n_bad = 0
    n_written = 0

    for i, info in enumerate(metas):
        domain = info["domain"]
        example_id = info["example_id"]
        pre_acc = float(info["pre_acc"])
        goal = info["goal"]
        train_obj = info["train_obj"]
        owner_nodes = info["owner_nodes"]

        task_template = train_obj.get("task_template") or {}
        old_ct = train_obj.get("current_task")

        chosen_ct: Optional[dict] = None
        candidates = outputs_by_task[i]

        # pick first valid among R samples (valid JSON + valid evaluator key in template)
        for out_text in candidates:
            obj = extract_last_json_obj(out_text)
            ct = normalize_current_task_obj(obj) if obj else None
            if ct is None:
                continue
            k = next(iter(ct.keys()))
            if k not in task_template:
                continue
            chosen_ct = ct
            break

        if chosen_ct is None:
            n_bad += 1
            chosen_ct = old_ct
            cprint(
                f"[WARN] Bad JSON -> fallback old current_task "
                f"({domain}/{example_id}, goal={goal}, pre_acc={pre_acc:.3f}, R={num_rollout_per_query})",
                "yellow",
            )
        else:
            n_ok += 1

        # write temp example json (keep everything, overwrite current_task, add pre_acc + difficulty)
        temp_obj = dict(train_obj)
        temp_obj["current_task"] = chosen_ct
        temp_obj["pre_acc"] = pre_acc
        #temp_obj["difficulty"] = "harder" if pre_acc >= 0.5 else "easier"
        temp_obj["difficulty"] = "harder" if pre_acc >= harder_threshold else "easier"

        temp_path = os.path.join(temp_root, domain, f"{example_id}.json")
        dump_json(temp_path, temp_obj)
        n_written += 1

        # add to each owner node's temp list
        for np in owner_nodes:
            per_node_temp.setdefault(np, []).append({"domain": domain, "example_id": example_id})

    # 7) write back to each example_node_i.json: overwrite temp, keep active
    for np in node_files:
        man = node_manifest.get(np, {"active": [], "temp": []})
        man["temp"] = per_node_temp.get(np, [])
        dump_json(np, man)

    cprint(f"[DONE] Temp examples written: {n_written} -> {temp_root}", "green")
    cprint(f"[DONE] Node manifests updated (temp overwritten, active kept): {len(node_files)} files", "green")
    cprint(f"[STAT] ok={n_ok}, fallback={n_bad}", "green")
