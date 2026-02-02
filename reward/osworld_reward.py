

import os
import json
import argparse
from termcolor import cprint
from typing import Any, Dict, List, Tuple
from collections import OrderedDict


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
            print(f"[WARN] fail to open, skip {traj_json} | reason: {e}")
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

    print(f"[DONE] messages+responses write to: {output_json}")
    return len(domains), len(out_index), len(tasks)



def safe_outcome_reward(result_val: Any) -> int:

    try:
        return 1 if float(result_val) > 0 else -1
    except Exception:
        return -1



def log_policy_accuracy_from_rollout(rollout_json: str, outputs_result_name: str, mode: str):

    if not os.path.isfile(rollout_json):
        print(f"[INFO] rollout_json does not exist, skip {rollout_json}")
        return

    with open(rollout_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    # domain -> [total, correct]
    domain_stat = {}
    total = 0
    correct = 0

    for item in data:
        domain = item.get("domain", "unknown")
        traj_list = item.get("trajctory_list", [])
        for t in traj_list:
            outcome = safe_outcome_reward(t.get("result"))
            if outcome == 0:
                continue
            total += 1
            if domain not in domain_stat:
                domain_stat[domain] = [0, 0]
            domain_stat[domain][0] += 1
            if outcome > 0:
                domain_stat[domain][1] += 1
                correct += 1

    if total == 0:
        print(f"[WARN] not effective result, can not get acc {rollout_json}")
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
        def save_and_print(text):
            cprint("\n\n\n" + text, color="green")
            f.write(text + "\n")
        save_and_print(line)




# ----------------- CLI -----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", required=True, help="domain/example/run/trajectory.json root")

    parser.add_argument("--num-nodes", type=int, default=-1,
                        help=">=0 merge reward slide data, reward_rollout_results_node_{i}.json, i=0..num_nodes")
    parser.add_argument("--merge-dir", type=str,
                        default="/data_storage/wyj/agentic-rl/osworld_multinode_rl/temp_data",
                        help="reward slide dir")

    parser.add_argument("--acc-lower", type=float, default=0.0,
                        help="example acc lower bound")
    parser.add_argument("--acc-upper", type=float, default=1.0,
                        help="example acc upper bound")
    parser.add_argument("--no-normalize", action="store_true",
                        help="no normalization (usually we do normalization)")
    parser.add_argument("--policy-out-name", type=str, default="policy_optimization_data.json")
    parser.add_argument("--reward-out-name", type=str, default="reward_optimization_data.json")
    parser.add_argument("--type", type=str, default="evaluation")
    parser.add_argument("--step", type=int, default=1)

    args = parser.parse_args()

    output_rollout_result = os.path.join(args.merge_dir, "eval_rollout_results.json")
    collect_messages_with_responses(args.root_dir, output_rollout_result)
    
    project_name = os.path.basename(os.path.dirname(os.path.abspath(args.merge_dir.rstrip("/"))))

    outputs_result_name = os.path.join("..", project_name, "results", "eval-results.txt")
    mode_str = "EVAL " + str(args.step)

    log_policy_accuracy_from_rollout(
        rollout_json=output_rollout_result,
        outputs_result_name=outputs_result_name,
        mode=mode_str,
    )

if __name__ == "__main__":
    main()
