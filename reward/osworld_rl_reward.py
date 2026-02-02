

import os
import re
import json
import argparse
from termcolor import cprint
from typing import Any, Dict, List, Tuple, Optional, Set
import math
from collections import OrderedDict, defaultdict



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



def _response_to_str(resp: Any) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        for k in ("text", "response", "output", "content"):
            v = resp.get(k)
            if isinstance(v, str):
                return v
        chs = resp.get("choices")
        if isinstance(chs, list) and chs:
            ch = chs[0]
            if isinstance(ch, dict):
                msg = ch.get("message")
                if isinstance(msg, dict):
                    c = msg.get("content")
                    if isinstance(c, str):
                        return c
                txt = ch.get("text")
                if isinstance(txt, str):
                    return txt
    if isinstance(resp, list):
        texts = []
        for it in resp:
            if isinstance(it, str):
                texts.append(it)
            elif isinstance(it, dict):
                t = it.get("text") or it.get("content")
                if isinstance(t, str):
                    texts.append(t)
        if texts:
            return "\n".join(texts)
    try:
        return json.dumps(resp, ensure_ascii=False)
    except Exception:
        return str(resp)



def _normalize_one_messages(ms: Any) -> List[Dict[str, Any]]:
    if isinstance(ms, list) and all(isinstance(x, dict) and ("role" in x) for x in ms):
        return ms
    if isinstance(ms, dict) and "role" in ms:
        return [ms]
    return []

def normalize_message_and_response_from_trajectory(traj: Any) -> Tuple[List[List[Dict[str, Any]]], List[str]]:
    messages_list: List[List[Dict[str, Any]]] = []
    responses: List[str] = []

    def _push(ms_any: Any, resp_any: Any):
        ms = _normalize_one_messages(ms_any)
        if not ms:
            return
        messages_list.append(ms)
        responses.append(_response_to_str(resp_any))

    if isinstance(traj, list):
        for step in traj:
            if not isinstance(step, dict):
                continue
            ms_any = step.get("messages") or step.get("message") or step.get("input") or step.get("prompt")
            resp_any = step.get("response") or step.get("output") or step.get("completion")
            _push(ms_any, resp_any)
    elif isinstance(traj, dict):
        ms_any = traj.get("messages") or traj.get("message") or traj.get("input") or traj.get("prompt")
        resp_any = traj.get("response") or traj.get("output") or traj.get("completion")
        _push(ms_any, resp_any)

    return messages_list, responses

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



def collect_messages_with_responses(root_dir: str, output_json: str) -> Tuple[int, int, int]:
    domains = list_subdirs(root_dir)
    tasks: List[Tuple[str, str, str, str]] = []  # (domain, example, run, traj_json)

    for domain in domains:
        domain_dir = os.path.join(root_dir, domain)
        examples = list_subdirs(domain_dir)
        for ex in examples:
            ex_dir = os.path.join(domain_dir, ex)
            runs = list_numeric_dirs(ex_dir)
            for run in runs:
                traj_json = os.path.join(ex_dir, run, "trajectory.json")
                if os.path.isfile(traj_json):
                    tasks.append((domain, ex, run, traj_json))

    out_index: Dict[Tuple[str, str], int] = {}
    result: List[Dict[str, Any]] = []
    for domain, ex, _run, _ in tasks:
        key = (domain, ex)
        if key not in out_index:
            out_index[key] = len(result)
            result.append({"domain": domain, "example": ex, "trajctory_list": []})

    for domain, ex, run, traj_json in tasks:
        try:
            with open(traj_json, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] failed to open, skip: {traj_json} | reason: {e}")
            continue

        meta_result = (data.get("meta") or {}).get("result")
        traj = data.get("trajectory", [])
        messages_list, responses = normalize_message_and_response_from_trajectory(traj)
        if not messages_list:
            continue

        idx = out_index[(domain, ex)]
        run_id = int(run) if str(run).isdigit() else run
        result[idx]["trajctory_list"].append({
            "run_id": run_id,
            "message": messages_list,
            "response": responses,
            "result": meta_result,
        })

    for item in result:
        sort_traj_list_inplace(item["trajctory_list"])

    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[DONE] messages+responses write: {output_json}")
    return len(domains), len(out_index), len(tasks)


# ----------------- aggregate reward slide -----------------
def _load_json_list(path: str) -> List[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return []
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, dict):
        return [obj]
    else:
        return []

def merge_reward_shards(merge_dir: str, num_nodes: int, out_path: str) -> int:
    shards = []
    for i in range(num_nodes):
        p = os.path.join(merge_dir, f"reward_rollout_results_node_{i}.json")
        if not os.path.isfile(p):
            print(f"[INFO] don't exists, skip: {p}")
            continue
        arr = _load_json_list(p)
        print(f"[INFO] read {i}: {p}, items={len(arr)}")
        shards.extend(arr)

    if not shards:
        print("[WARN] dont find any reward shard to merge.")
        return 0

    merged = OrderedDict()
    for item in shards:
        try:
            d = item["domain"]; e = item["example"]
            tl = item.get("trajctory_list", [])
        except Exception:
            continue
        key = (d, e)
        if key not in merged:
            merged[key] = {"domain": d, "example": e, "trajctory_list": []}
        if isinstance(tl, list):
            merged[key]["trajctory_list"].extend(tl)

    for _, ex_item in merged.items():
        sort_traj_list_inplace(ex_item["trajctory_list"])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(list(merged.values()), f, ensure_ascii=False, indent=2)

    print(f"[DONE] reward merge completed {out_path} | items={len(merged)}")
    return len(merged)



def safe_outcome_reward(result_val: Any) -> int:

    try:
        return 1 if float(result_val) > 0 else -1
    except Exception:
        return -1

def success01(result_val: Any) -> int:
    try:
        return 1 if float(result_val) > 0 else 0
    except Exception:
        return 0

def compute_acc_map_from_rollout(rollout_json: str) -> Dict[Tuple[str, str], float]:

    with open(rollout_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    out: Dict[Tuple[str, str], float] = {}
    for item in data:
        d = item.get("domain"); e = item.get("example")
        if d is None or e is None:
            continue
        tl = item.get("trajctory_list", []) or []
        flags = [success01(t.get("result")) for t in tl]
        out[(d, e)] = (sum(flags) / len(flags)) if flags else 0.0
    return out

def build_id2domain_from_rollout(rollout_json: str) -> Dict[str, str]:
    with open(rollout_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    m = {}
    for item in data:
        d = item.get("domain"); e = item.get("example")
        if isinstance(d, str) and isinstance(e, str):
            m[e] = d
    return m



def list_example_node_files(manifest_dir: str) -> List[str]:
    if not os.path.isdir(manifest_dir):
        return []
    out = []
    for fn in sorted(os.listdir(manifest_dir)):
        if re.fullmatch(r"example_node_\d+\.json", fn):
            out.append(os.path.join(manifest_dir, fn))
    return out

def normalize_manifest_obj(manifest_obj: Any, id2domain: Dict[str, str]) -> Dict[str, List[Dict[str, str]]]:

    def _norm_list(lst: Any) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        if not isinstance(lst, list):
            return out
        for x in lst:
            if isinstance(x, dict) and ("domain" in x) and ("example_id" in x):
                out.append({"domain": str(x["domain"]), "example_id": str(x["example_id"])})
            elif isinstance(x, str):
                d = id2domain.get(x)
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

def aggregate_active_temp_sets(manifest_dir: str, id2domain: Dict[str, str]) -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    node_files = list_example_node_files(manifest_dir)
    active_set: Set[Tuple[str, str]] = set()
    temp_set: Set[Tuple[str, str]] = set()

    for p in node_files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue
        norm = normalize_manifest_obj(obj, id2domain)
        for it in (norm.get("active") or []):
            active_set.add((it["domain"], it["example_id"]))
        for it in (norm.get("temp") or []):
            temp_set.add((it["domain"], it["example_id"]))

    return active_set, temp_set


# ----------------- test temp + if accept update train -----------------
def _read_json(path: str) -> Optional[dict]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None

def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def evaluate_and_promote_temp_tasks(
    base_dir: str,
    project_name: str,
    temp_set: Set[Tuple[str, str]],
    active_set: Set[Tuple[str, str]],
    after_acc_map: Dict[Tuple[str, str], float],
    epoch: int,
    delta: float = 0.0,
) -> Set[Tuple[str, str]]:

    eval_base = os.path.join(base_dir, "OSWorld-main", "evaluation_examples", project_name)
    train_root = os.path.join(eval_base, "train")
    temp_root = os.path.join(eval_base, "temp")

    passed: Set[Tuple[str, str]] = set()

    for (d, e) in sorted(list(temp_set)):
        after_acc = float(after_acc_map.get((d, e), 0.0))

        temp_path = os.path.join(temp_root, d, f"{e}.json")
        temp_obj = _read_json(temp_path)
        if temp_obj is None:
            cprint(f"[WARN] temp example missing, skip validate: {temp_path}", "yellow")
            continue

        pre_acc = temp_obj.get("pre_acc")
        try:
            pre_acc = float(pre_acc)
        except Exception:
            cprint(f"[WARN] temp example has no valid pre_acc, skip: {temp_path}", "yellow")
            continue

        diff = temp_obj.get("difficulty")
        if not isinstance(diff, str) or diff.lower() not in ("harder", "easier"):
            diff = "harder" if pre_acc >= 0.5 else "easier"
        diff = diff.lower()

        ok = False
        if diff == "harder":
            ok = (after_acc < pre_acc - float(delta)) and (after_acc >= 0.2)
        else:
            ok = (after_acc > pre_acc + float(delta)) and (after_acc <= 0.8)
        if after_acc == 0:
            ok = False

        if not ok:
            cprint(f"[INFO] temp NOT passed: {d}/{e} diff={diff} pre={pre_acc:.3f} after={after_acc:.3f} delta={delta}", "yellow")
            continue

        train_path = os.path.join(train_root, d, f"{e}.json")
        train_obj = _read_json(train_path)
        if train_obj is None:
            cprint(f"[WARN] train example missing, cannot promote: {train_path}", "yellow")
            continue

        old_ct = train_obj.get("current_task")
        new_ct = temp_obj.get("current_task")

        if not isinstance(new_ct, dict) or len(new_ct) != 1:
            cprint(f"[WARN] temp current_task invalid, skip promote: {temp_path}", "yellow")
            continue

        hist = train_obj.get("history")
        if not isinstance(hist, list):
            hist = []

        hist.append({
            "epoch": int(epoch),
            "pre_acc": float(pre_acc),
            "after_acc": float(after_acc),
            "difficulty": diff,
            "old_current_task": old_ct,
            "new_current_task": new_ct,
        })
        train_obj["history"] = hist
        train_obj["current_task"] = new_ct

        _write_json(train_path, train_obj)

        passed.add((d, e))
        cprint(f"[PASS] promote temp -> train updated: {d}/{e} diff={diff} pre={pre_acc:.3f} after={after_acc:.3f}", "green")

    return passed


def _stepwise_standardize_inplace(records: List[Dict[str, Any]], require_multi_traj: bool = True) -> None:
    
    by_step = defaultdict(list)
    for i, rec in enumerate(records):
        by_step[int(rec["step"])].append(i)

    for step_k, idxs in by_step.items():
        run_ids = {records[i]["run_id"] for i in idxs}
        vals = [float(records[i]["raw_reward"]) for i in idxs]

        if require_multi_traj and len(run_ids) <= 1:
            for i in idxs:
                records[i]["reward"] = float(records[i]["raw_reward"])
            continue

        if len(vals) == 1:
            i = idxs[0]
            records[i]["reward"] = float(records[i]["raw_reward"])
            continue

        mu = sum(vals) / len(vals)
        var = sum((x - mu) ** 2 for x in vals) / len(vals)
        std = math.sqrt(var)

        if std == 0.0:
            for i in idxs:
                records[i]["reward"] = 0.0
            continue

        for i in idxs:
            x = float(records[i]["raw_reward"])
            records[i]["reward"] = (x - mu) / std


def _reward_within_rollout_standardize_inplace(records: List[Dict[str, Any]]) -> None:

    by_key = defaultdict(list)
    for i, rec in enumerate(records):
        by_key[(rec["run_id"], int(rec["step"]))].append(i)

    for _, idxs in by_key.items():
        vals = [float(records[i]["raw_reward"]) for i in idxs]

        if len(vals) <= 1:
            for i in idxs:
                records[i]["reward"] = 0.0
            continue

        mu = sum(vals) / len(vals)
        var = sum((x - mu) ** 2 for x in vals) / len(vals)
        std = math.sqrt(var)

        if std == 0.0:
            for i in idxs:
                records[i]["reward"] = 0.0
            continue

        for i in idxs:
            x = float(records[i]["raw_reward"])
            records[i]["reward"] = (x - mu) / std


# ----------------- generate RL training data -----------------
def generate_rl_datasets(
    rollout_json: str,
    reward_json: str,
    out_dir: str,
    acc_lower: float = 0.125,
    acc_upper: float = 0.875,
    reward_acc_lower: float = 0.2,   
    reward_acc_upper: float = 0.8,   
    normalize: bool = True,
    policy_out_name: str = "policy_optimization_data.json",
    reward_out_name: str = "reward_optimization_data.json",
    allow_set: Optional[Set[Tuple[str, str]]] = None,
):
    if not os.path.isfile(rollout_json):
        raise FileNotFoundError(f"rollout_json not exist: {rollout_json}")
    if not os.path.isfile(reward_json):
        raise FileNotFoundError(f"reward_json not exist: {reward_json}")

    with open(rollout_json, "r", encoding="utf-8") as f:
        ROLL = json.load(f)
    with open(reward_json, "r", encoding="utf-8") as f:
        REWA = json.load(f)

    def _index_by_de(arr):
        m = {}
        for it in arr:
            d = it.get("domain"); e = it.get("example")
            if d is None or e is None:
                continue
            m[(d, e)] = it
        return m

    idx_roll = _index_by_de(ROLL)
    idx_rewa = _index_by_de(REWA)

    policy_dataset: List[Dict[str, Any]] = []
    reward_dataset: List[Dict[str, Any]] = []

    for key, item_roll in idx_roll.items():
        d, e = key

        if allow_set is not None and key not in allow_set:
            continue

        item_rewa = idx_rewa.get(key)
        if not item_rewa:
            print(f"[WARN] miss reward data, skip example: {key}")
            continue

        traj_roll = item_roll.get("trajctory_list", []) or []
        traj_rewa = item_rewa.get("trajctory_list", []) or []
        sort_traj_list_inplace(traj_roll)
        sort_traj_list_inplace(traj_rewa)

        def _by_run_id(lst):
            m = {}
            for t in lst:
                m[t.get("run_id")] = t
            return m

        roll_by_run = _by_run_id(traj_roll)
        rewa_by_run = _by_run_id(traj_rewa)
        common_run_ids = [rid for rid in roll_by_run.keys() if rid in rewa_by_run]
        if not common_run_ids:
            print(f"[WARN] no mutual run_id, skip example: {key}")
            continue

        results_sign = []
        for rid in common_run_ids:
            res = roll_by_run[rid].get("result")
            results_sign.append(1 if safe_outcome_reward(res) > 0 else 0)
        acc = sum(results_sign) / len(results_sign) if results_sign else 0.0

        keep_policy = True
        if acc_lower is not None and acc_upper is not None:
            keep_policy = (acc >= acc_lower and acc <= acc_upper)

        keep_reward = True
        if reward_acc_lower is not None and reward_acc_upper is not None:
            keep_reward = (acc >= reward_acc_lower and acc <= reward_acc_upper)

        if (not keep_policy) and (not keep_reward):
            continue

        policy_records: List[Dict[str, Any]] = []
        reward_records: List[Dict[str, Any]] = []

        def _rid_key(v):
            if isinstance(v, int):
                return (0, v)
            if str(v).isdigit():
                return (0, int(v))
            return (1, str(v))

        for rid in sorted(common_run_ids, key=_rid_key):
            roll_t = roll_by_run[rid]
            rewa_t = rewa_by_run[rid]

            outcome = float(safe_outcome_reward(roll_t.get("result")))
            proc_list = rewa_t.get("process_reward", []) or []

            pol_msg_list  = roll_t.get("message", []) or []
            pol_resp_list = roll_t.get("response", []) or []

            rew_msg_list   = rewa_t.get("message", pol_msg_list) or []
            rew_resp_list  = rewa_t.get("response", pol_resp_list) or []
            extracted_list = rewa_t.get("extracted_reward", []) or []

            T_pol = min(len(proc_list), len(pol_msg_list), len(pol_resp_list))
            T_rew = min(len(proc_list), len(rew_msg_list), len(rew_resp_list), len(extracted_list))

            # --- policy：---
            if keep_policy:
                for k in range(T_pol):
                    prc = float(proc_list[k])
                    raw = outcome + prc
                    policy_records.append({
                        "domain": d,
                        "example": e,
                        "run_id": rid,
                        "step": k,
                        "prompt_messages": pol_msg_list[k],
                        "response": pol_resp_list[k],
                        "raw_reward": raw,
                    })

            # --- reward：---
            if keep_reward:
                for k in range(T_rew):
                    prc = float(proc_list[k])
                    msgs_k = rew_msg_list[k]
                    resp_list_k = rew_resp_list[k]
                    sign_list_k = extracted_list[k]

                    R = min(len(resp_list_k), len(sign_list_k))
                    for r_idx in range(R):
                        sign = float(sign_list_k[r_idx])
                        raw = (outcome + prc) * sign
                        if sign == 0:
                            raw = -2

                        reward_records.append({
                            "domain": d,
                            "example": e,
                            "run_id": rid,
                            "step": k,
                            "prompt_messages": msgs_k,
                            "response": resp_list_k[r_idx],
                            "raw_reward": raw,
                        })

        if normalize:
            if keep_policy and policy_records:
                _stepwise_standardize_inplace(policy_records, require_multi_traj=True)
            else:
                policy_records = []

            if keep_reward and reward_records:
                _reward_within_rollout_standardize_inplace(reward_records)
            else:
                reward_records = []
        else:
            if keep_policy:
                for rec in policy_records:
                    rec["reward"] = float(rec["raw_reward"])
            else:
                policy_records = []

            if keep_reward:
                for rec in reward_records:
                    rec["reward"] = float(rec["raw_reward"])
            else:
                reward_records = []

        for rec in policy_records:
            if float(rec["reward"]) == 0.0:
                continue
            out = dict(rec)
            out.pop("raw_reward", None)
            policy_dataset.append(out)

        for rec in reward_records:
            if float(rec["reward"]) == 0.0:
                continue
            out = dict(rec)
            out.pop("raw_reward", None)
            reward_dataset.append(out)

    os.makedirs(out_dir or ".", exist_ok=True)
    policy_out = os.path.join(out_dir, policy_out_name)
    reward_out = os.path.join(out_dir, reward_out_name)

    with open(policy_out, "w", encoding="utf-8") as f:
        json.dump(policy_dataset, f, ensure_ascii=False, indent=2)
    with open(reward_out, "w", encoding="utf-8") as f:
        json.dump(reward_dataset, f, ensure_ascii=False, indent=2)

    print(f"[DONE] generate Policy training data {policy_out} | {len(policy_dataset)} samples")
    print(f"[DONE] generate Policy training data {reward_out} | {len(reward_dataset)} samples")
    if reward_acc_lower is not None and reward_acc_upper is not None:
        print(f"[INFO] Reward data comes from acc∈[{reward_acc_lower},{reward_acc_upper}] tasks")


# ----------------- log_policy_accuracy -----------------
def log_policy_accuracy_from_rollout(
    rollout_json: str,
    outputs_result_name: str,
    mode: str,
    allow_set: Optional[Set[Tuple[str, str]]] = None,
):
    if not os.path.isfile(rollout_json):
        print(f"[INFO] rollout_json not exists, acc summary{rollout_json}")
        return

    with open(rollout_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    domain_stat = {}
    total = 0
    correct = 0

    for item in data:
        domain = item.get("domain", "unknown")
        example = item.get("example")

        if allow_set is not None:
            if not isinstance(example, str) or (domain, example) not in allow_set:
                continue

        traj_list = item.get("trajctory_list", [])
        for t in traj_list:
            outcome = safe_outcome_reward(t.get("result"))
            total += 1
            if domain not in domain_stat:
                domain_stat[domain] = [0, 0]
            domain_stat[domain][0] += 1
            if outcome > 0:
                domain_stat[domain][1] += 1
                correct += 1

    if total == 0:
        print(f"[WARN] not effective result, can not summarize acc: {rollout_json}")
        return

    parts = []
    for d in sorted(domain_stat.keys()):
        tot, cor = domain_stat[d]
        if tot == 0:
            continue
        acc = cor / tot
        parts.append(f"{d}:{acc:.4f}")

    overall_acc = correct / total
    parts.append(f"ALL:{overall_acc:.4f}")

    line = f"[{mode}] " + " ".join(parts)

    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    with open(outputs_result_name, "a", encoding="utf-8") as f:
        cprint("\n\n\n" + line, color="green")
        f.write(line + "\n")


# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True, help="domain/example/run/trajectory.json root dir")

    parser.add_argument("--num-nodes", type=int, default=-1,
                        help=">=0 merge reward files: reward_rollout_results_node_{i}.json, i=0..num_nodes")
    parser.add_argument("--merge-dir", type=str,
                        default="/data_storage/wyj/agentic-rl/osworld_multinode_rl/temp_data",
                        help="reward slide dir, (also contains example_node_*.json)")

    # policy acc thresholds
    parser.add_argument("--acc-lower", type=float, default=0.0)
    parser.add_argument("--acc-upper", type=float, default=1.0)

    # reward acc thresholds
    parser.add_argument("--reward-acc-lower", type=float, default=0.2)
    parser.add_argument("--reward-acc-upper", type=float, default=0.8)

    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--policy-out-name", type=str, default="policy_optimization_data.json")
    parser.add_argument("--reward-out-name", type=str, default="reward_optimization_data.json")
    parser.add_argument("--type", type=str, default="train")
    parser.add_argument("--step", type=int, default=1)

    parser.add_argument("--temp-delta", type=float, default=0.0,
                        help="temp pass judge margin")

    args = parser.parse_args()

    # collect message+response
    if args.type == "train":
        output_rollout_result = os.path.join(args.merge_dir, "rl_rollout_results.json")
    else:
        output_rollout_result = os.path.join(args.merge_dir, "eval_rollout_results.json")
    collect_messages_with_responses(args.root_dir, output_rollout_result)

    # get id2domain,after_acc_map, then aggregate active/temp
    id2domain = build_id2domain_from_rollout(output_rollout_result)
    after_acc_map = compute_acc_map_from_rollout(output_rollout_result)

    active_set, temp_set = aggregate_active_temp_sets(args.merge_dir, id2domain)
    cprint(f"[INFO] active_set={len(active_set)} temp_set={len(temp_set)} (from {args.merge_dir}/example_node_*.json)", "green")

    # temp harder/easier validate + promote
    project_name = os.path.basename(os.path.dirname(os.path.abspath(args.merge_dir.rstrip("/"))))
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.merge_dir.rstrip("/"))))

    passed_temp_set: Set[Tuple[str, str]] = set()
    if args.type == "train" and temp_set:
        passed_temp_set = evaluate_and_promote_temp_tasks(
            base_dir=base_dir,
            project_name=project_name,
            temp_set=temp_set,
            active_set=active_set,
            after_acc_map=after_acc_map,
            epoch=int(args.step),
            delta=float(args.temp_delta),
        )
        cprint(f"[INFO] passed_temp_set={len(passed_temp_set)}", "green")

    allow_set: Optional[Set[Tuple[str, str]]] = None
    if args.type == "train":
        allow_set = set(active_set) | set(passed_temp_set)
        cprint(f"[INFO] allow_set={len(allow_set)} (active + passed_temp)", "green")

    # merge reward slides
    reward_merged_path = ""
    if args.type == "train":
        reward_merged_path = os.path.join(args.merge_dir, "rl_reward_rollout_results.json")
        merge_reward_shards(args.merge_dir, args.num_nodes, reward_merged_path)

    # generate RL training data
    if args.type == "train":
        rollout_json = output_rollout_result
        reward_json = reward_merged_path

        lower = args.acc_lower if args.acc_lower is not None and args.acc_lower >= 0 else None
        upper = args.acc_upper if args.acc_upper is not None and args.acc_upper <= 1 else None
        normalize = not args.no_normalize

        generate_rl_datasets(
            rollout_json=rollout_json,
            reward_json=reward_json,
            out_dir=args.merge_dir,
            acc_lower=lower if (lower is not None and upper is not None) else None,
            acc_upper=upper if (lower is not None and upper is not None) else None,
            reward_acc_lower=args.reward_acc_lower,
            reward_acc_upper=args.reward_acc_upper,
            normalize=normalize,
            policy_out_name=args.policy_out_name,
            reward_out_name=args.reward_out_name,
            allow_set=allow_set,
        )

    # acc
    if args.type == "train":
        outputs_result_name = os.path.join("..", project_name, "results", "rl-results.txt")
        mode_str = "TRAIN " + str(args.step)
    else:
        outputs_result_name = os.path.join("..", project_name, "results", "eval-results.txt")
        mode_str = "EVAL " + str(args.step)

    log_policy_accuracy_from_rollout(
        rollout_json=output_rollout_result,
        outputs_result_name=outputs_result_name,
        mode=mode_str,
        allow_set=active_set if args.type == "train" else None,
    )


if __name__ == "__main__":
    main()
