# -*- coding: utf-8 -*-
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import json
import random
import argparse
import multiprocessing as mp
from termcolor import cprint

import time
from queue import Empty
from vllm import LLM, SamplingParams

from transformers import AutoTokenizer
from omegaconf import OmegaConf

from alfworld_utils import *  # noqa


############################
# Config
############################
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


############################
# TP GPU group helpers
############################
def make_tp_gpu_groups(tp: int, num_visible_gpus: int):
    """
    Return gpu_groups for tensor parallel engines.

    Example:
      num_visible_gpus=8, tp=4 -> [[0,1,2,3],[4,5,6,7]]
      num_visible_gpus=8, tp=8 -> [[0,1,2,3,4,5,6,7]]
    """
    if tp <= 0:
        raise ValueError(f"tp must be positive, got {tp}")
    if num_visible_gpus <= 0:
        raise RuntimeError("No visible CUDA devices. Check CUDA_VISIBLE_DEVICES / your allocation.")
    if num_visible_gpus % tp != 0:
        raise ValueError(
            f"Visible GPUs ({num_visible_gpus}) must be divisible by tp ({tp}). "
            f"Current CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','<unset>')}"
        )
    return [list(range(i, i + tp)) for i in range(0, num_visible_gpus, tp)]


############################
# vLLM Worker Pool
############################
def worker_fn(pretrained_model, gpu_ids, task_queue, result_queue, max_model_len, max_generation_token, temp):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    print(f"[vLLM] Loading model on GPUs {gpu_ids}: {pretrained_model}")

    try:
        print(f"[worker {gpu_ids}] Loading model...")
        llm = LLM(
            model=pretrained_model,
            dtype="bfloat16",
            tensor_parallel_size=len(gpu_ids),
            gpu_memory_utilization=0.85,
            max_model_len=max_model_len
        )
    except Exception as e:

        result_queue.put(("ERROR", f"init failed on GPUs {gpu_ids}: {repr(e)}"))
        return

    sampling_params = SamplingParams(
        temperature=temp,
        top_p=0.95,
        top_k=-1,
        min_p=0.0,
        max_tokens=max_generation_token,
        stop=["</answer>", "User:", "Human:", "Assistant:", "<|im_end|>", "<|endoftext|>"],
    )

    while True:
        task = task_queue.get()
        if task == "STOP":
            print(f"[worker {gpu_ids}] Stopping worker...")
            break
        task_id, prompts = task
        try:
            outputs = llm.generate(prompts, sampling_params)
            result_texts = [out.outputs[0].text for out in outputs]
            result_queue.put((task_id, result_texts))
        except Exception as e:
            result_queue.put(("ERROR", f"generate failed on GPUs {gpu_ids}: {repr(e)}"))
            break


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

    # 2) join / terminate
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


def parse_binary_reward_from_output(out_text: str) -> int:
    """
    Force extracted_reward to be in {-1, 0, 1}.
    - Return  1 if boxed contains a clear +1/1
    - Return -1 if boxed contains a clear -1
    - Otherwise return 0

    This is robust to cases where the model accidentally includes other numbers
    (e.g., action index, Attempt 1/8, step numbers) inside the boxed content.
    """
    boxed = extract_final_boxed_answer(out_text or "")
    if not boxed:
        return 0

    s = boxed.strip()
    # normalize unicode minus signs
    s = s.replace("−", "-").replace("–", "-")

    # Fast path: exact
    if s in ("1", "+1"):
        return 1
    if s == "-1":
        return -1

    # If it contains a fraction like 1/8, treat as invalid -> 0
    if re.search(r"\d+\s*/\s*\d+", s):
        return 0

    # Extract all integers in the boxed content
    nums = [int(x) for x in re.findall(r"[-+]?\d+", s)]
    if not nums:
        return 0

    # Keep only {-1, 1}; if there's exactly one such signal, use it, else ambiguous -> 0
    good = [v for v in nums if v in (-1, 1)]
    if len(good) == 1:
        return good[0]

    return 0


def extract_block_raw(text: str, head="goal"):
    """
    Return the full raw block including the wrapper:
      (:goal ... )
    """
    start = text.find(f"(:{head}")
    if start < 0:
        return ""
    i, depth = start, 0
    while i < len(text):
        c = text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
        i += 1
    return ""


def split_prompts(prompts, n):
    k, m = divmod(len(prompts), n)
    return [prompts[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def generate_results(
    all_prompts,
    gpu_groups,
    task_queues,
    result_queues,
    desc: str = "policy",
    timeout_s: float = 3600.0,
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
        now = time.time()
        if now - start_time > timeout_s:
            raise RuntimeError(
                f"[{desc}] timeout waiting results; still missing jobs {sorted(remaining)}"
            )

        for i, rq in enumerate(result_queues):
            if i not in remaining:
                continue
            try:
                task_id, result = rq.get(timeout=0.1)
            except Empty:
                continue

            if task_id == "ERROR":
                raise RuntimeError(
                    f"[{desc}] worker {i} reported error: {result}"
                )

            results_by_job[task_id] = result
            remaining.remove(task_id)

    out = []
    for i, prompts in enumerate(chunks):
        if not prompts:
            continue
        if i not in results_by_job:
            raise RuntimeError(
                f"[{desc}] missing result for job {i} (this should not happen)"
            )
        out.extend(results_by_job[i])
    return out


############################
# Trajectory + Prompt Builders
############################
def _as_list(x):
    return x if isinstance(x, (list, tuple)) else [x]


def _cmds_from_info(info_i):
    cmds = info_i.get("admissible_commands", [])
    if isinstance(cmds, (list, tuple, set)):
        cmds = list(cmds)
        if cmds and isinstance(cmds[0], (list, tuple, set)):
            cmds = list(cmds[0])
    return cmds


def _render_traj(traj):
    lines = []
    for t in traj:
        if "obs" in t and t["obs"] is not None:
            lines.append(f"observation: {t['obs']}")
        if t.get("act") is not None:
            lines.append(f"you took action: {t['act']}")
    return "\n".join(lines)


def _render_traj_numbered(traj):
    """Render full trajectory with step numbers for summary prompts."""
    lines = []
    for t_idx, t in enumerate(traj):
        obs = t.get("obs", None)
        act = t.get("act", None)
        if obs is not None:
            lines.append(f"[{t_idx}] observation: {obs}")
        if act is not None:
            lines.append(f"[{t_idx}] action: {act}")
    return "\n".join(lines)


def build_traj_failure_summary_prompt(task: str, traj, max_steps: int, bad_steps=None):
    """
    Ask model to summarize why the rollout failed within max_steps.
    Additionally provide high-probability wrong step indices from reward model.
    Output must be <= 2 sentences inside \\boxed{}.
    """
    traj_text = _render_traj_numbered(traj) or "(empty trajectory)"
    bad_steps = bad_steps or []

    if bad_steps:
        bad_steps_str = ",".join(map(str, bad_steps))
        bad_hint = f"High-risk steps (always -1): [{bad_steps_str}] (match [idx]).\n"
    else:
        bad_hint = ""

    prompt = (
        "<|im_start|>You are a helpful assistant. <|im_end|>\n"
        "<|im_start|>user\n"
        "You are analyzing a failed rollout of a policy in a text-based environment.\n"
        f"The rollout did NOT finish the task within {max_steps} interaction steps.\n"
        f"Task (natural language): {task}\n\n"
        f"{bad_hint}"
        "Full trajectory (observation/action sequence):\n"
        f"{traj_text}\n\n"
        "In at most TWO sentences, explain the most likely reasons the policy failed to finish in time.\n"
        "Be concrete (e.g., wrong exploration, looping, wrong target/location, inconsistent reasoning, hallucination, etc.).\n"
        "Put the final <=2 sentence summary in \\boxed{} and output NOTHING else.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant"
    )
    return prompt


def _split_info_dict_to_batches(info_dict, B):
    out = [dict() for _ in range(B)]
    for k, v in info_dict.items():
        if isinstance(v, (list, tuple)) and len(v) == B:
            for i in range(B):
                out[i][k] = v[i]
        else:
            for i in range(B):
                out[i][k] = v
    return out


def _align_infos(infos, B):
    if isinstance(infos, dict):
        return _split_info_dict_to_batches(infos, B)
    if isinstance(infos, (list, tuple)):
        infos = list(infos)
        if len(infos) == B and (not infos or isinstance(infos[0], dict)):
            return infos
        if len(infos) == 1 and isinstance(infos[0], dict):
            return _split_info_dict_to_batches(infos[0], B)
        infos = [x if isinstance(x, dict) else {} for x in (infos + [{}] * B)[:B]]
        return infos
    return [{} for _ in range(B)]


def _get_uids_and_infos(infos, B):
    infos_b = _align_infos(infos, B)
    uids = [str(infos_b[i].get("extra.uid", "")) for i in range(B)]
    assert all(uids), f"[ENV] missing extra.uid in infos. keys={list(infos_b[0].keys())[:30]}"
    assert len(set(uids)) == B, f"[ENV] uid collision in batch! sample={uids[:3]}"
    return uids, infos_b


def _reorder_by_uid(obs, scores, dones, infos, uid_order, uid2gf):
    obs = _as_list(obs)
    scores = list(scores) if isinstance(scores, (list, tuple)) else [scores]
    dones  = list(dones)  if isinstance(dones,  (list, tuple)) else [dones]
    B = len(obs)

    uids_cur, infos_b = _get_uids_and_infos(infos, B)

    assert set(uids_cur) == set(uid_order), (
        f"[ENV] uid set changed within episode!\n"
        f"order0[:3]={uid_order[:3]}\ncur[:3]={uids_cur[:3]}"
    )

    pos = {u:i for i,u in enumerate(uids_cur)}
    perm = [pos[u] for u in uid_order]

    obs    = [obs[i] for i in perm]
    scores = [scores[i] for i in perm]
    dones  = [dones[i] for i in perm]
    infos_b = [infos_b[i] for i in perm]

    for i, u in enumerate(uid_order):
        gf = os.path.realpath(os.path.normpath(str(infos_b[i]["extra.gamefile"])))
        assert gf == uid2gf[u], f"[ENV] gamefile changed for uid={u}\nold={uid2gf[u]}\nnew={gf}"

    return obs, scores, dones, infos_b


from collections import Counter

def _canon_path(p: str) -> str:
    return os.path.realpath(os.path.normpath(str(p)))

def _infos_gamefiles(infos, B: int):
    infos_b = _align_infos(infos, B)
    gfs = [_canon_path(infos_b[i].get("extra.gamefile", "")) for i in range(B)]
    assert all(gfs), f"[ENV] missing extra.gamefile in infos. keys={list(infos_b[0].keys())[:30]}"
    return gfs

def _assert_multiset_equal(got_list, exp_list, where: str):
    cg, ce = Counter(got_list), Counter(exp_list)
    if cg != ce:
        only_exp = list((ce - cg).elements())[:5]
        only_got = list((cg - ce).elements())[:5]
        raise AssertionError(
            f"[{where}] multiset mismatch between env-returned gamefiles and sub_paths.\n"
            f"only_in_expected={only_exp}\n"
            f"only_in_got={only_got}"
        )


def build_step_prompts(histories, infos, is_think):
    B = len(histories)
    infos_b = _align_infos(infos, B)
    prompts = []
    guide = (
        "You are playing a text game. Your objective is to complete the task as soon as possible.\n"
        "Below is your trajectory so far and current candidate actions.\n"
        "You need to think step by step then put the integer (the index of your chosen action) in \\boxed{}.\n"
    )
    for i in range(B):
        traj = _render_traj(histories[i]) or "(empty)"
        cand = _cmds_from_info(infos_b[i]) or ["look"]
        options = "\n".join(f"{k}) {a}" for k, a in enumerate(cand))
        prompt_i = (
            "<|im_start|>You are a helpful assistant. <|im_end|>\n"
            "<|im_start|>user\n"
            f"{guide}\n\n{traj}\n\n"
            "You need to think step by step then choose one action by number:\n"
            f"{options}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant"
        )
        if is_think:
            prompt_i += "<think>"
        prompts.append(prompt_i)
    return prompts


def extract_final_boxed_answer(s: str):
    tag = r"\boxed{"
    start = s.rfind(tag)
    if start == -1:
        return ""
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
    return "".join(buf) if depth == 0 else ""


def safe_to_index(s: str, default=0) -> int:
    if s is None:
        return default
    m = re.search(r"-?\d+", str(s))
    try:
        return int(m.group(0)) if m else default
    except Exception:
        return default


def map_idxlist_to_actions(indices, infos, B: int):
    infos_b = _align_infos(infos, B)
    idxs = (list(indices) + [0] * B)[:B]
    actions = []
    for i in range(B):
        cand = _cmds_from_info(infos_b[i]) or ["look"]
        j = max(0, min(int(idxs[i]), len(cand) - 1))
        actions.append(cand[j])
    return actions


def _num_actions_so_far(traj):
    return sum(1 for t in traj if t.get("act") is not None)


def update_success_steps(success_steps, infos, histories):
    B = len(histories)
    out = list(success_steps) if success_steps is not None else [-1] * B
    infos_b = _align_infos(infos, B)
    for i in range(B):
        if out[i] != -1:
            continue
        won = bool(infos_b[i].get("won", False))
        if won:
            out[i] = max(0, _num_actions_so_far(histories[i]) - 1)
    return out


def update_histories(histories, actions, obs, alive_mask=None):
    obs_list = list(obs) if isinstance(obs, (list, tuple)) else [obs]
    if alive_mask is None:
        alive_mask = [True] * len(histories)
    for i, act in enumerate(actions):
        if not alive_mask[i]:
            continue
        histories[i][-1]["act"] = act
        histories[i].append({"obs": obs_list[i], "act": None})


############################
# Data partition helpers
############################
def random_select(data_list, random_k):
    return random.sample(data_list, random_k)


def get_random_k(node_index, num_node, total_num):
    if node_index < total_num % num_node:
        return 1 + int(total_num / num_node)
    return int(total_num / num_node)


def get_data_chunk(data, num_nodes, node_idx):
    total = len(data)
    start = (total * node_idx) // num_nodes
    end = (total * (node_idx + 1)) // num_nodes
    return data[start:end]


from typing import List

def compute_acc_from_if_success(if_success_list: List[int]) -> float:
    if not if_success_list:
        return 0.0
    corr = sum(1 for x in if_success_list if int(x) == 1)
    return corr / len(if_success_list)


def save_outputs(project_name, node_index, num_node, outputs_name, data):
    if num_node > 1:
        output_file = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"
    else:
        output_file = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[INFO] wrote outputs: {output_file}", flush=True)
    return output_file


############################
# Main
############################
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  

    config = get_config()

    # -------------------------
    # FORCE TP SETTINGS (as requested):
    #   policy: TP=4
    #   reward/env: TP=8
    # -------------------------
    import torch
    num_visible_gpus = torch.cuda.device_count()

    POLICY_TP = 4
    OTHER_TP  = 8

    policy_gpu_groups = make_tp_gpu_groups(POLICY_TP, num_visible_gpus)
    other_gpu_groups  = make_tp_gpu_groups(OTHER_TP,  num_visible_gpus)

    print(f"[TP CONFIG] visible_gpus={num_visible_gpus}")
    print(f"[TP CONFIG] policy TP={POLICY_TP}, gpu_groups={policy_gpu_groups}")
    print(f"[TP CONFIG] other  TP={OTHER_TP},  gpu_groups={other_gpu_groups}")

    project_name = config.experiment.project
    num_node = int(config.experiment.num_node)
    node_index = int(config.experiment.node_index)
    current_epoch = int(config.experiment.current_epoch)

    # policy model path
    if current_epoch == 1:
        policy_model = config.model.policy_model
    else:
        policy_model = "../" + project_name + "/ckpt/" + config.model.optimized_name

    # load rollout params
    if config.experiment.function == "train":
        gpu_groups = policy_gpu_groups  # <<< FORCE policy TP=4
        max_model_len = config.rollout.policy.model_length
        max_generation_token = config.rollout.policy.max_gen_length
        temp = config.rollout.policy.temperature
        num_rollout_per_trial = config.rollout.policy.num_rollout_per_trial
        max_interaction_step = config.rollout.policy.max_interaction_step
        if_start_with_think = config.rollout.policy.if_start_with_think
        num_trial_total = int(config.rollout.policy.num_trial)
    else:
        gpu_groups = policy_gpu_groups  # <<< FORCE policy TP=4 (evaluation too)
        max_model_len = config.evaluation.policy.model_length
        max_generation_token = config.evaluation.policy.max_gen_length
        temp = config.evaluation.policy.temperature
        num_rollout_per_trial = config.evaluation.policy.num_rollout_per_trial
        max_interaction_step = config.evaluation.policy.max_interaction_step
        if_start_with_think = config.evaluation.policy.if_start_with_think
        num_trial_total = 0

    # AlfWorld env setup
    os.environ["ALFWORLD_DATA"] = config.dataset.environment_data_dir
    os.environ["ALFWORLD_DATA_PROJECT"] = config.experiment.project

    import sys
    repo_root = config.dataset.environment_file_dir
    sys.path.insert(0, repo_root)
    sys.argv = ["", config.dataset.environment_file_dir + "/configs/base_config.yaml"]

    from my_environments import AlfredTWEnv as get_environment
    import alfworld.agents.modules.generic as generic
    env_config = generic.load_config()

    # ===========================
    # Evaluation: keep old behavior
    # ===========================
    if config.experiment.function == "evaluation":
        env = get_environment(env_config, train_eval=config.dataset.alfworld_eval_type)
        env._build_catalog()

        env.subset_and_repeat(
            indices=list(range(len(env.catalog))),
            repeat=num_rollout_per_trial,
            shuffle=False,
        )

        TOTAL = len(env.game_files)
        CAP = min(int(config.rollout.env_max_parallel), TOTAL) if TOTAL > 0 else 0

        task_list = [row["task"] for row in env.catalog]
        trial_list = [row["trial"] for row in env.catalog]
        task_path_list = [row["task_path"] for row in env.catalog]

        key2trial = {}
        global_trial_idx = [0] * TOTAL
        global_rollout_idx = [0] * TOTAL
        first_g_for_trial = []
        per_trial_counts = []

        for g in range(TOTAL):
            key = (task_list[g], trial_list[g], task_path_list[g])
            if key not in key2trial:
                tid = len(per_trial_counts)
                key2trial[key] = tid
                per_trial_counts.append(0)
                first_g_for_trial.append(g)
            tid = key2trial[key]
            global_trial_idx[g] = tid
            r = per_trial_counts[tid]
            global_rollout_idx[g] = r
            per_trial_counts[tid] += 1

        path2trial_eval = {}
        for g in range(TOTAL):
            key = (task_list[g], trial_list[g], task_path_list[g])
            tid = key2trial[key]
            gf = _canon_path(env.game_files[g])
            if gf in path2trial_eval:
                assert path2trial_eval[gf] == tid, f"[EVAL MAP CONFLICT] {gf} maps to multiple trial_idx"
            else:
                path2trial_eval[gf] = tid

        num_unique = len(per_trial_counts)
        data = [{} for _ in range(num_unique)]
        for i in range(num_unique):
            g0 = first_g_for_trial[i]
            data[i]["task"] = task_list[g0]
            data[i]["trial"] = trial_list[g0]
            data[i]["task_path"] = task_path_list[g0]
            data[i]["type"] = "raw"
            data[i]["source"] = "evaluation"
            data[i]["trajectory"] = [[] for _ in range(num_rollout_per_trial)]
            data[i]["prompt"] = [[] for _ in range(num_rollout_per_trial)]
            data[i]["response"] = [[] for _ in range(num_rollout_per_trial)]
            data[i]["if_success"] = [-1 for _ in range(num_rollout_per_trial)]
            data[i]["success_steps"] = [-1 for _ in range(num_rollout_per_trial)]
            data[i]["init_obs"] = [""] * num_rollout_per_trial

        task_queues, result_queues, processes = start_workers(
            policy_model, gpu_groups, max_model_len, max_generation_token, temp
        )

        outputs_name = "eval-" + policy_model.replace("/", ".") + "-" + config.dataset.environment_type + "-" + config.dataset.alfworld_eval_type

        per_trial_seen = [0] * num_unique
        visited_eval = [[False] * num_rollout_per_trial for _ in range(num_unique)]

        for offset in range(0, TOTAL, CAP if CAP > 0 else 1):
            sub_paths = env.game_files[offset: offset + CAP]
            sub_catalog = env.catalog[offset: offset + CAP]
            _saved_paths, _saved_catalog = env.game_files, env.catalog
            env.game_files, env.catalog = sub_paths, sub_catalog

            tw_env = env.init_env(batch_size=len(sub_paths))
            obs, infos = tw_env.reset()
            obs = _as_list(obs)
            B = len(obs)

            uid_order, infos_b0 = _get_uids_and_infos(infos, B)
            uid2gf = {
                uid_order[i]: os.path.realpath(os.path.normpath(str(infos_b0[i]["extra.gamefile"])))
                for i in range(B)
            }
            infos = infos_b0

            slot_gfs0 = _infos_gamefiles(infos, B)
            expected = [_canon_path(p) for p in sub_paths]
            _assert_multiset_equal(slot_gfs0, expected, where="EVAL RESET")

            histories = [[{"obs": obs[i], "act": None}] for i in range(B)]
            success_steps = [-1] * B
            per_batch_prompts = [[] for _ in range(B)]
            per_batch_responses = [[] for _ in range(B)]

            for i_step in range(max_interaction_step):
                if all(s != -1 for s in success_steps):
                    break

                prompts = build_step_prompts(histories, infos, if_start_with_think)
                active_index = [i for i, s in enumerate(success_steps) if s == -1]
                target_prompts = [prompts[i] for i in active_index]

                cprint(f"[policy] eval chunk {offset//max(1,CAP)}/{max(1, (TOTAL-1)//max(1,CAP))} step {i_step}/{max_interaction_step}", "green")

                if target_prompts:
                    shuffled_idx = list(range(len(target_prompts)))
                    random.shuffle(shuffled_idx)
                    shuffled_prompts = [target_prompts[i] for i in shuffled_idx]
                    shuffled_outputs = generate_results(shuffled_prompts, gpu_groups, task_queues, result_queues, desc="policy")
                    restored_outputs = [None] * len(target_prompts)
                    for out, idx in zip(shuffled_outputs, shuffled_idx):
                        restored_outputs[idx] = out

                    reply_indices = [safe_to_index(extract_final_boxed_answer(x or ""), default=0) for x in restored_outputs]
                else:
                    reply_indices = []
                    restored_outputs = []

                final_indices = []
                full_outputs = []
                ptr = 0
                for i in range(B):
                    if success_steps[i] != -1:
                        final_indices.append(0)
                        full_outputs.append("The task is completed.")
                    else:
                        final_indices.append(reply_indices[ptr] if ptr < len(reply_indices) else 0)
                        full_outputs.append(restored_outputs[ptr] if ptr < len(restored_outputs) else "")
                        ptr += 1

                for k in range(B):
                    per_batch_prompts[k].append(prompts[k])
                    per_batch_responses[k].append(full_outputs[k])

                actions = map_idxlist_to_actions(final_indices, infos, B)
                obs, scores, dones, infos = tw_env.step(actions)

                obs, scores, dones, infos = _reorder_by_uid(obs, scores, dones, infos, uid_order, uid2gf)

                alive_mask = [success_steps[i] == -1 for i in range(B)]
                update_histories(histories, actions, obs, alive_mask=alive_mask)
                success_steps = update_success_steps(success_steps, infos, histories)

            for k in range(B):
                gf = _canon_path(infos[k].get("extra.gamefile", ""))
                assert gf in path2trial_eval, f"[EVAL WRITEBACK] unknown gf: {gf}"
                trial_idx = path2trial_eval[gf]

                rollout_idx = per_trial_seen[trial_idx]
                assert rollout_idx < num_rollout_per_trial, (
                    f"[EVAL WRITEBACK] trial_idx={trial_idx} exceeded rollouts {num_rollout_per_trial}"
                )
                per_trial_seen[trial_idx] += 1

                assert not visited_eval[trial_idx][rollout_idx], (
                    f"[EVAL WRITEBACK DUP] trial_idx={trial_idx}, rollout_idx={rollout_idx}"
                )
                visited_eval[trial_idx][rollout_idx] = True

                hist_k = histories[k]
                eff_len = _num_actions_so_far(hist_k)
                data[trial_idx]["init_obs"][rollout_idx] = hist_k[0]["obs"]

                if eff_len == 0:
                    data[trial_idx]["if_success"][rollout_idx] = -1
                    data[trial_idx]["trajectory"][rollout_idx] = []
                    data[trial_idx]["prompt"][rollout_idx] = []
                    data[trial_idx]["response"][rollout_idx] = []
                    data[trial_idx]["success_steps"][rollout_idx] = -1
                    continue

                if success_steps[k] == -1:
                    data[trial_idx]["if_success"][rollout_idx] = -1
                    data[trial_idx]["trajectory"][rollout_idx] = histories[k]
                    data[trial_idx]["prompt"][rollout_idx] = per_batch_prompts[k]
                    data[trial_idx]["response"][rollout_idx] = per_batch_responses[k]
                    data[trial_idx]["success_steps"][rollout_idx] = -1
                else:
                    cut = success_steps[k] + 1
                    data[trial_idx]["if_success"][rollout_idx] = 1
                    data[trial_idx]["trajectory"][rollout_idx] = histories[k]
                    data[trial_idx]["prompt"][rollout_idx] = per_batch_prompts[k][:cut]
                    data[trial_idx]["response"][rollout_idx] = per_batch_responses[k][:cut]
                    data[trial_idx]["success_steps"][rollout_idx] = cut

            try:
                if tw_env is not None and hasattr(tw_env, "close"):
                    tw_env.close()
            except Exception as e:
                print("[WARN] tw_env.close() failed:", e)
            finally:
                del tw_env
                env.game_files, env.catalog = _saved_paths, _saved_catalog

        stop_workers(task_queues, result_queues, processes)

        time.sleep(5)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        output_file = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        raise SystemExit(0)



    # ===========================
    # Train: NEW slot+active+pending dataset
    # ===========================

    env_train = get_environment(env_config, train_eval=config.dataset.alfworld_train_type)
    raw_game_paths = list(env_train.game_files)
    total_raw = len(raw_game_paths)
    all_raw_indices = list(range(total_raw))
    my_raw_indices = get_data_chunk(all_raw_indices, num_node, node_index) if num_node > 1 else all_raw_indices

    env_data_dir = config.dataset.environment_data_dir
    st = load_env_state(env_data_dir, project_name, node_index=node_index)

    if current_epoch == 1 and (not st.get("slots")):
        for idx in my_raw_indices:
            gf = os.path.realpath(os.path.normpath(raw_game_paths[idx]))
            raw_game_id = canonical_game_id(gf, env_data_dir)
            ensure_slot(st, str(idx), raw_game_id)
        save_env_state(env_data_dir, project_name, st, node_index=node_index)

    pending_list = list(st.get("pending", []))
    pending_slots = pending_slot_ids(st)

    slot_keys = sorted(st.get("slots", {}).keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    candidate_slots = [sid for sid in slot_keys if sid not in pending_slots]
    random.shuffle(candidate_slots)

    per_node_quota = get_random_k(node_index, num_node, num_trial_total)
    take_n = min(per_node_quota, len(candidate_slots))
    sampled_slots = candidate_slots[:take_n]

    unique_meta = []

    for p in pending_list:
        sid = str(p.get("slot_id"))
        temp_gid = p.get("temp_game_id") or p.get("game_id")
        if not temp_gid:
            continue
        unique_meta.append({
            "source": "pending",
            "slot_id": sid,
            "goal": p.get("goal"),
            "prev_acc": p.get("prev_acc"),
            "created_epoch": p.get("created_epoch"),
            "raw_game_id": p.get("raw_game_id"),
            "parent_active_game_id": p.get("parent_active_game_id"),
            "game_id": temp_gid,
            "type": "temp",
        })

    for sid in sampled_slots:
        slot = st["slots"][sid]
        gid = slot.get("active_game_id", slot.get("raw_game_id"))
        gtype = slot.get("active_type", game_type_from_game_id(gid, project_name))
        unique_meta.append({
            "source": "active",
            "slot_id": sid,
            "game_id": gid,
            "type": gtype,
            "raw_game_id": slot.get("raw_game_id"),
            "active_game_id": gid,
            "active_epoch": slot.get("active_epoch", 0),
        })

    num_unique = len(unique_meta)

    unique_abs_files = [resolve_game_id(m["game_id"], env_data_dir) for m in unique_meta]

    outputs_name = "rl-" + policy_model.replace("/", ".") + "-" + config.dataset.environment_type

    if len(unique_abs_files) == 0:
        save_outputs(project_name, node_index, num_node, outputs_name, data=[])
        raise SystemExit(0)

    env = get_environment(env_config, train_eval=config.dataset.alfworld_train_type)
    env.game_files = unique_abs_files
    env.num_games = len(unique_abs_files)
    env._build_catalog()

    abs2idx = {os.path.realpath(os.path.normpath(p)): i for i, p in enumerate(env.game_files)}

    ordered_indices = []
    for m in unique_meta:
        p = os.path.realpath(os.path.normpath(resolve_game_id(m["game_id"], env_data_dir)))
        if p not in abs2idx:
            raise RuntimeError(f"[ORDER MISMATCH] cannot find game in env.catalog: {p}")
        ordered_indices.append(abs2idx[p])

    env.subset_and_repeat(indices=ordered_indices, repeat=num_rollout_per_trial, shuffle=False)

    TOTAL = len(env.game_files)
    assert TOTAL == len(unique_meta) * num_rollout_per_trial, \
        (TOTAL, len(unique_meta), num_rollout_per_trial)

    from collections import defaultdict, Counter

    def _canon(p: str) -> str:
        return os.path.realpath(os.path.normpath(p))

    meta_paths = [_canon(resolve_game_id(m["game_id"], env_data_dir)) for m in unique_meta]
    dup_meta_paths = [p for p, c in Counter(meta_paths).items() if c > 1]
    assert len(dup_meta_paths) == 0, (
        f"[DUP META PATH] unique_meta contains duplicate game paths! "
        f"examples={dup_meta_paths[:3]} (showing up to 3)"
    )

    assert len(set(ordered_indices)) == len(ordered_indices), (
        "[ORDERED_INDICES DUP] multiple unique_meta items map to the same env index. "
        "This means abs2idx dict got overwritten (duplicate paths) or env.game_files has duplicates."
    )

    path2trial = {meta_paths[i]: i for i in range(len(unique_meta))}

    global_trial_idx = [-1] * TOTAL
    global_rollout_idx = [-1] * TOTAL
    trial_seen = [0] * len(unique_meta)

    for g in range(TOTAL):
        p = _canon(env.game_files[g])
        assert p in path2trial, f"[MAP ERROR] env.game_files[{g}] not in unique_meta: {p}"
        tid = path2trial[p]
        rid = trial_seen[tid]
        global_trial_idx[g] = tid
        global_rollout_idx[g] = rid
        trial_seen[tid] += 1

    bad = [(i, c) for i, c in enumerate(trial_seen) if c != num_rollout_per_trial]
    assert not bad, f"[REPEAT COUNT BAD] trials with wrong repeat count: {bad[:10]}"

    print("[DEBUG] mapping head:", [(g, global_trial_idx[g], global_rollout_idx[g]) for g in range(min(8, TOTAL))])

    task_list = [row["task"] for row in env.catalog]
    trial_list = [row["trial"] for row in env.catalog]
    task_path_list = [row["task_path"] for row in env.catalog]

    from collections import defaultdict

    trial2keys = defaultdict(set)
    for g in range(TOTAL):
        tid = global_trial_idx[g]
        key = (task_list[g], trial_list[g], task_path_list[g])
        trial2keys[tid].add(key)

    bad_trials = {tid: keys for tid, keys in trial2keys.items() if len(keys) != 1}
    assert not bad_trials, (
        "[TRIAL->TASK MAPPING BUG] some trial_idx map to multiple (task, trial, task_path).\n"
        + "\n".join(
            f"  trial_idx={tid}: {list(keys)[:3]}" for tid, keys in list(bad_trials.items())[:5]
        )
    )

    def _canon(p: str) -> str:
        return os.path.realpath(os.path.normpath(p))

    num_unique = len(unique_meta)

    first_g_for_trial = [None] * num_unique
    per_trial_counts = [0] * num_unique

    for g in range(TOTAL):
        p = _canon(env.game_files[g])
        if p not in path2trial:
            raise RuntimeError(f"[MAPPING ERROR] game file not in unique_meta: {p}")
        tid = path2trial[p]

        if first_g_for_trial[tid] is None:
            first_g_for_trial[tid] = g

        per_trial_counts[tid] += 1

    for tid, cnt in enumerate(per_trial_counts):
        if cnt != num_rollout_per_trial:
            print(f"[WARN] trial {tid} expected {num_rollout_per_trial} rollouts, got {cnt}")

    data = [{} for _ in range(num_unique)]
    for i in range(num_unique):
        meta = unique_meta[i]
        data[i]["source"] = meta["source"]
        data[i]["slot_id"] = meta["slot_id"]
        data[i]["node_index"] = int(node_index)
        data[i]["type"] = meta.get("type", "raw")
        data[i]["game_id"] = meta["game_id"]
        data[i]["raw_game_id"] = meta.get("raw_game_id")
        if meta["source"] == "pending":
            data[i]["goal"] = meta.get("goal")
            data[i]["prev_acc"] = meta.get("prev_acc")
            data[i]["created_epoch"] = meta.get("created_epoch")
            data[i]["parent_active_game_id"] = meta.get("parent_active_game_id")

        g0 = first_g_for_trial[i]
        if g0 is not None:
            data[i]["task"] = task_list[g0]
            data[i]["trial"] = trial_list[g0]
            data[i]["task_path"] = task_path_list[g0]
        else:
            data[i]["task"] = ""
            data[i]["trial"] = ""
            data[i]["task_path"] = ""

        data[i]["trajectory"] = [[] for _ in range(num_rollout_per_trial)]
        data[i]["prompt"] = [[] for _ in range(num_rollout_per_trial)]
        data[i]["response"] = [[] for _ in range(num_rollout_per_trial)]
        data[i]["if_success"] = [-1 for _ in range(num_rollout_per_trial)]
        data[i]["success_steps"] = [-1 for _ in range(num_rollout_per_trial)]
        data[i]["init_obs"] = [""] * num_rollout_per_trial

    cprint("[policy] start generation...", "green")
    task_queues, result_queues, processes = start_workers(policy_model, gpu_groups, max_model_len, max_generation_token, temp)

    outputs_name = "rl-" + policy_model.replace("/", ".") + "-" + config.dataset.environment_type

    TOTAL = len(env.game_files)
    CAP = min(int(config.rollout.env_max_parallel), TOTAL) if TOTAL > 0 else 0

    visited = [[False] * num_rollout_per_trial for _ in range(num_unique)]
    trial_write_ptr = [0] * num_unique

    for offset in range(0, TOTAL, CAP if CAP > 0 else 1):
        sub_paths = env.game_files[offset: offset + CAP]
        sub_catalog = env.catalog[offset: offset + CAP]
        _saved_paths, _saved_catalog = env.game_files, env.catalog
        env.game_files, env.catalog = sub_paths, sub_catalog

        tw_env = env.init_env(batch_size=len(sub_paths))
        obs, infos = tw_env.reset()
        obs = _as_list(obs)
        B = len(obs)

        uid_order, infos_b0 = _get_uids_and_infos(infos, B)
        uid2gf = {
            uid_order[i]: os.path.realpath(os.path.normpath(str(infos_b0[i]["extra.gamefile"])))
            for i in range(B)
        }
        infos = infos_b0

        slot_gfs0 = _infos_gamefiles(infos, B)
        expected = [_canon_path(p) for p in sub_paths]
        _assert_multiset_equal(slot_gfs0, expected, where="TRAIN RESET")

        histories = [[{"obs": obs[i], "act": None}] for i in range(B)]
        success_steps = [-1] * B
        per_batch_prompts = [[] for _ in range(B)]
        per_batch_responses = [[] for _ in range(B)]

        for i_step in range(max_interaction_step):
            if all(s != -1 for s in success_steps):
                break

            prompts = build_step_prompts(histories, infos, if_start_with_think)
            active_index = [i for i, s in enumerate(success_steps) if s == -1]
            target_prompts = [prompts[i] for i in active_index]

            cprint(f"[policy] chunk {offset//max(1,CAP)}/{max(1, (TOTAL-1)//max(1,CAP))} step {i_step}/{max_interaction_step}", "green")

            if target_prompts:
                shuffled_idx = list(range(len(target_prompts)))
                random.shuffle(shuffled_idx)
                shuffled_prompts = [target_prompts[i] for i in shuffled_idx]
                shuffled_outputs = generate_results(shuffled_prompts, gpu_groups, task_queues, result_queues, desc="policy")
                restored_outputs = [None] * len(target_prompts)
                for out, idx in zip(shuffled_outputs, shuffled_idx):
                    restored_outputs[idx] = out

                reply_indices = [safe_to_index(extract_final_boxed_answer(x or ""), default=0) for x in restored_outputs]
            else:
                reply_indices = []
                restored_outputs = []

            final_indices = []
            full_outputs = []
            ptr = 0
            for i in range(B):
                if success_steps[i] != -1:
                    final_indices.append(0)
                    full_outputs.append("The task is completed.")
                else:
                    final_indices.append(reply_indices[ptr] if ptr < len(reply_indices) else 0)
                    full_outputs.append(restored_outputs[ptr] if ptr < len(restored_outputs) else "")
                    ptr += 1

            for k in range(B):
                per_batch_prompts[k].append(prompts[k])
                per_batch_responses[k].append(full_outputs[k])

            actions = map_idxlist_to_actions(final_indices, infos, B)
            obs, scores, dones, infos = tw_env.step(actions)

            obs, scores, dones, infos = _reorder_by_uid(obs, scores, dones, infos, uid_order, uid2gf)

            alive_mask = [success_steps[i] == -1 for i in range(B)]
            update_histories(histories, actions, obs, alive_mask=alive_mask)
            success_steps = update_success_steps(success_steps, infos, histories)

        for k in range(B):
            gf = uid2gf[uid_order[k]]
            assert gf in path2trial, f"[TRAIN WRITEBACK] unknown gf: {gf}"
            trial_idx = path2trial[gf]

            rollout_idx = trial_write_ptr[trial_idx]
            assert rollout_idx < num_rollout_per_trial
            trial_write_ptr[trial_idx] += 1

            assert not visited[trial_idx][rollout_idx]
            visited[trial_idx][rollout_idx] = True

            hist_k = histories[k]

            eff_len = _num_actions_so_far(hist_k)
            data[trial_idx]["init_obs"][rollout_idx] = hist_k[0]["obs"]

            if eff_len == 0:
                data[trial_idx]["if_success"][rollout_idx] = -1
                data[trial_idx]["trajectory"][rollout_idx] = []
                data[trial_idx]["prompt"][rollout_idx] = []
                data[trial_idx]["response"][rollout_idx] = []
                data[trial_idx]["success_steps"][rollout_idx] = -1
                continue

            if success_steps[k] == -1:
                data[trial_idx]["if_success"][rollout_idx] = -1
                data[trial_idx]["trajectory"][rollout_idx] = histories[k]
                data[trial_idx]["prompt"][rollout_idx] = per_batch_prompts[k]
                data[trial_idx]["response"][rollout_idx] = per_batch_responses[k]
                data[trial_idx]["success_steps"][rollout_idx] = -1
            else:
                cut = success_steps[k] + 1
                data[trial_idx]["if_success"][rollout_idx] = 1
                data[trial_idx]["trajectory"][rollout_idx] = histories[k]
                data[trial_idx]["prompt"][rollout_idx] = per_batch_prompts[k][:cut]
                data[trial_idx]["response"][rollout_idx] = per_batch_responses[k][:cut]
                data[trial_idx]["success_steps"][rollout_idx] = cut

        try:
            if tw_env is not None and hasattr(tw_env, "close"):
                tw_env.close()
        except Exception as e:
            print("[WARN] tw_env.close() failed:", e)
        finally:
            del tw_env
            env.game_files, env.catalog = _saved_paths, _saved_catalog

    stop_workers(task_queues, result_queues, processes)

    time.sleep(5)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    cprint("[policy] stop generation...", "green")

    for i in range(len(data)):
        data[i]["acc"] = compute_acc_from_if_success(data[i]["if_success"])


    ############################
    # Reward model inference
    ############################
    if current_epoch == 1:
        reward_model = config.model.reward_model
    else:
        reward_model = "../" + project_name + "/ckpt/" + config.model.optimized_reward_name

    # <<< FORCE reward TP=8
    rgpu_groups = other_gpu_groups
    rmax_model_len = config.rollout.reward.model_length
    rmax_generation_token = config.rollout.reward.max_gen_length
    rtemp = config.rollout.reward.temperature
    num_rollout_per_query = config.rollout.reward.num_rollout_per_query
    start_with_think = config.rollout.reward.if_start_with_think

    task_queues, result_queues, processes = start_workers(
        reward_model, rgpu_groups, rmax_model_len, rmax_generation_token, rtemp
    )

    for i in range(len(data)):
        data[i]["failed_rollout_summaries"] = []
        data[i]["reward_prompt"] = [[] for _ in range(num_rollout_per_trial)]
        data[i]["reward_response"] = [[] for _ in range(num_rollout_per_trial)]
        data[i]["extracted_reward"] = [[] for _ in range(num_rollout_per_trial)]

        for j in range(num_rollout_per_trial):
            traj_j = data[i]["trajectory"][j]
            n_traj = len(traj_j)
            n_prompt = len(data[i]["prompt"][j])
            n_steps = min(n_prompt, max(0, n_traj - 1))

            data[i]["reward_prompt"][j] = [[] for _ in range(n_steps)]
            data[i]["reward_response"][j] = [[] for _ in range(n_steps)]
            data[i]["extracted_reward"][j] = [[] for _ in range(n_steps)]

    reward_prompt_set = []
    reward_items = []

    for i in range(len(data)):
        for j in range(num_rollout_per_trial):
            traj_j = data[i]["trajectory"][j]
            steps_j = len(data[i]["extracted_reward"][j])
            for k in range(steps_j):
                policy_prompt = data[i]["prompt"][j][k]
                policy_response = data[i]["response"][j][k]
                next_obs = traj_j[k + 1]["obs"]

                rp = (
                    "<|im_start|>You are a helpful assistant. <|im_end|>\n"
                    "<|im_start|>user\n"
                    "You are a judge for an agent acting in a text-based environment.\n"
                    "Evaluate ONE step using:\n"
                    " - the agent's prompt (observation + candidate actions),\n"
                    " - its response (reasoning + chosen index), and\n"
                    " - the environment's next observation after executing that action.\n"
                    "Scoring (binary):\n"
                    "• Score 1 if ALL are true:\n"
                    "  (a) The selected action is appropriate for the current observation and task goal (it reasonably explores, progresses or completes the task);\n"
                    "  (b) The reasoning is present, relevant, and not self-contradictory (no hallucinated objects/locations);\n"
                    "  (c) The chosen index exists in the candidate list, and the resulting next observation is consistent with the described action.\n"
                    "• Otherwise score -1. Cases include: no reasoning provided; index out of range; clearly irrelevant; undoes progress; self-contradictory/hallucinated reasoning; or next observation contradicts the action.\n"
                    "Important: think first then put the final score in \\boxed{}.\n\n"
                    f"Agent's prompt:\n{policy_prompt}\n\n"
                    f"Agent's response:\n{policy_response}\n\n"
                    f"Next observation after this action:\n{next_obs}\n"
                    "<|im_end|>\n"
                    "<|im_start|>assistant"
                )
                if start_with_think:
                    rp += "<think>"

                for _rep in range(num_rollout_per_query):
                    reward_prompt_set.append(rp)
                    reward_items.append((i, j, k))

    cprint("[reward] start generation...", "green")
    Np = len(reward_prompt_set)
    shuffled_idx = list(range(Np))
    random.shuffle(shuffled_idx)
    shuffled_prompts = [reward_prompt_set[idx] for idx in shuffled_idx]
    shuffled_outputs = generate_results(shuffled_prompts, rgpu_groups, task_queues, result_queues, desc="reward")
    restored_outputs = [None] * Np
    for out, idx in zip(shuffled_outputs, shuffled_idx):
        restored_outputs[idx] = out
    cprint("[reward] generation done!", "green")

    for t in range(Np):
        i, j, k = reward_items[t]
        out_text = restored_outputs[t] or ""
        score = parse_binary_reward_from_output(out_text)  # <<< KEY CHANGE

        data[i]["reward_prompt"][j][k].append(reward_prompt_set[t])
        data[i]["reward_response"][j][k].append(out_text)
        data[i]["extracted_reward"][j][k].append(int(score))








    ############################
    # Env coevolve
    ############################
    st = load_env_state(env_data_dir, project_name, node_index=node_index)
    pend_slots = pending_slot_ids(st)

    env_model_spec = str(getattr(config.model, "environment_model", "")).strip()

    stop_workers(task_queues, result_queues, processes)

    time.sleep(20)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    env_model = env_model_spec

    # <<< FORCE env TP=8
    egpu_groups = other_gpu_groups
    emax_model_len = config.rollout.environment.model_length
    emax_generation_token = 8192
    etemp = config.rollout.environment.temperature
    estart_with_think = config.rollout.environment.if_start_with_think

    task_queues, result_queues, processes = start_workers(
        env_model, egpu_groups, emax_model_len, emax_generation_token, etemp
    )
    active_gpu_groups = egpu_groups
    start_with_think_env = estart_with_think









    # ============================================================
    # NEW: Trajectory-level failure summarization (use ENV model)
    # Must run BEFORE building env_prompts so env prompt can read it.
    # ============================================================
    # (Optional) clear in case rerun in same process
    for ii in range(len(data)):
        if "failed_rollout_summaries" not in data[ii] or data[ii]["failed_rollout_summaries"] is None:
            data[ii]["failed_rollout_summaries"] = []
        else:
            data[ii]["failed_rollout_summaries"] = []

    traj_sum_prompts = []
    traj_sum_items = []

    for ii in range(len(data)):
        task_ii = str(data[ii].get("task", "")).strip()
        for jj in range(num_rollout_per_trial):
            if int(data[ii]["if_success"][jj]) == 1:
                continue
            traj_j = data[ii]["trajectory"][jj]
            if not traj_j:
                continue
            
            # NEW: bad step indices from reward model
            bad_steps = []
            try:
                step_scores = data[ii]["extracted_reward"][jj]
                for k, reps in enumerate(step_scores):
                    if not reps:
                        continue
                    reps_int = [int(x) for x in reps]
                    if (len(reps_int) == num_rollout_per_query) and all(x == -1 for x in reps_int):
                        bad_steps.append(k)
            except Exception:
                bad_steps = []

            p = build_traj_failure_summary_prompt(
                task=task_ii if task_ii else "(unknown task)",
                traj=traj_j,
                max_steps=max_interaction_step,
                bad_steps=bad_steps,
            )
            if start_with_think_env:
                p += "<think>"

            traj_sum_prompts.append(p)
            traj_sum_items.append((ii, jj))

    if traj_sum_prompts:
        cprint(f"[env] start trajectory failure summarization for {len(traj_sum_prompts)} failed rollouts...", "green")

        Ns = len(traj_sum_prompts)
        sidx = list(range(Ns))
        random.shuffle(sidx)
        shuffled_prompts_s = [traj_sum_prompts[idx] for idx in sidx]

        shuffled_out_s = generate_results(
            shuffled_prompts_s,
            active_gpu_groups,          # <<< ENV TP=8 groups
            task_queues, result_queues, # <<< ENV workers
            desc="traj_summary_env",
        )

        restored_out_s = [None] * Ns
        for out, idx in zip(shuffled_out_s, sidx):
            restored_out_s[idx] = out

        for t in range(Ns):
            ii, jj = traj_sum_items[t]
            raw = restored_out_s[t] or ""
            summ = (extract_final_boxed_answer(raw) or "").strip()
            if not summ:
                summ = ""

            data[ii]["failed_rollout_summaries"].append({
                "rollout_idx": int(jj),
                "summary": summ,
            })

        cprint("[env] trajectory failure summarization done!", "green")
    else:
        cprint("[env] no failed rollouts to summarize.", "green")


















    env_prompts = []
    env_jobs = []  # [(req, attempt_id, prompt_text), ...]

    # how many attempts per active task
    ENV_NUM_TRIES = 8

    for item in data:
        if item.get("source") != "active":
            continue
        sid = str(item.get("slot_id"))
        if sid in pend_slots:
            continue

        acc_before = float(item.get("acc", 0.0))
        goal = direction_from_acc(acc_before)
        if goal is None:
            continue

        raw_gid = item.get("raw_game_id")
        parent_active_gid = item.get("game_id")
        if not raw_gid:
            continue

        raw_abs = resolve_game_id(raw_gid, env_data_dir)
        if not os.path.exists(raw_abs):
            print(f"[WARN] raw game missing for slot {sid}: {raw_abs}")
            continue

        game_json = json.load(open(raw_abs, "r", encoding="utf-8"))
        problem_text = game_json.get("pddl_problem", "")

        summary_text, S = summarize_init_english(problem_text)
        summary_text = (
            "<|im_start|>You are a helpful assistant. <|im_end|>\n"
            "<|im_start|>user\n"
            "Review the following details about an interactive environment. A related task will follow.\n"
            + summary_text
        )

        task = detect_task_type_from_path(raw_abs)
        goal_raw = extract_block_raw(problem_text, "goal")
        goal_obj_types, goal_rec_types = goal_detect_types(goal_raw)

        summary_text += "\n\n---\n"

        fails = item.get("failed_rollout_summaries", [])
        if fails:
            summary_text += "### Failure summaries from recent rollouts (failed rollouts only)\n"
            for rec in sorted(fails, key=lambda x: int(x.get("rollout_idx", 0))):
                rj = rec.get("rollout_idx", 0)
                ss = str(rec.get("summary", "")).strip()
                summary_text += f"- Rollout {rj}: {ss}\n"
            summary_text += "\n"

        summary_text += f"### Your job is to propose a new goal that makes the task **{goal.upper()}**.\n"
        summary_text += f"- The parent rollout accuracy (prev_acc) is {acc_before}.\n"
        summary_text += "- The new goal must be different from the original and follow the instructions.\n"
        summary_text += "- The overall framework of the goal cannot be changed; you may only modify two tokens within this framework.\n"
        summary_text += "Represent the new goal by outputting two tokens, placed inside \\boxed{} and separated by a comma, e.g., \\boxed{TOKEN_A,TOKEN_B}.\n"
        summary_text += "You need to think step by step then provide final result in \\boxed{}.\n"
        summary_text += "\n" + goal_brief_and_instruction(
            task, goal_obj_types, goal_rec_types, S, direction=goal, prev_acc=acc_before
        )

        # base prompt (shared)
        base_prompt = summary_text + " <|im_end|>\n<|im_start|>assistant"
        # NOTE: we add attempt marker to encourage diverse outputs
        for attempt_id in range(ENV_NUM_TRIES):
            p = base_prompt + f"\n\n(Attempt {attempt_id+1}/{ENV_NUM_TRIES})\n"
            if start_with_think_env:
                p += "<think>"

            env_prompts.append(p)
            env_jobs.append((
                {
                    "slot_id": sid,
                    "goal": goal,
                    "prev_acc": acc_before,
                    "raw_game_id": raw_gid,
                    "parent_active_game_id": parent_active_gid,
                    "raw_abs_path": raw_abs,
                    "env_prompt": p,
                    "failed_rollout_summaries": fails,
                    "task": task,
                    "goal_raw": goal_raw,
                    "goal_obj_types": goal_obj_types,
                    "goal_rec_types": goal_rec_types,
                    "S": S,
                    "game_json": game_json,
                },
                attempt_id,
                p,
            ))

    if env_prompts:
        cprint(f"[env] start generation... (tasks={len(env_jobs)//ENV_NUM_TRIES}, tries={ENV_NUM_TRIES})", "green")

        Np2 = len(env_prompts)
        shuffled_idx2 = list(range(Np2))
        random.shuffle(shuffled_idx2)

        shuffled_prompts2 = [env_prompts[idx] for idx in shuffled_idx2]
        shuffled_outputs2 = generate_results(shuffled_prompts2, active_gpu_groups, task_queues, result_queues, desc="env")
        restored_outputs2 = [None] * Np2
        for out, idx in zip(shuffled_outputs2, shuffled_idx2):
            restored_outputs2[idx] = out

        cprint("[env] generation done!", "green")

        # --- create candidates ---
        cand_records = []  # list of dict; one per (slot, attempt)
        for (req, attempt_id, _prompt_used), out_text in zip(env_jobs, restored_outputs2):
            sid = req["slot_id"]
            raw_abs = req["raw_abs_path"]

            try:
                tboxed = extract_final_boxed_answer(out_text or "")
                type1, type2 = parse_two_tokens(tboxed)
            except Exception as e:
                # parsing failed => just skip this attempt
                continue

            task = req["task"]
            goal_raw = req["goal_raw"]
            goal_obj_types = req["goal_obj_types"]
            goal_rec_types = req["goal_rec_types"]
            S = req["S"]
            game_json = req["game_json"]

            try:
                new_types = decide_new_types(task, goal_obj_types, goal_rec_types, type1, type2, S=S)
            except Exception:
                continue

            try:
                _new_task_dir, _new_trial_dir, new_game_path, _new_data = create_synthetic_task(
                    new_types=new_types,
                    templ=None,
                    goal_raw=goal_raw,
                    task=task,
                    S=S,
                    data=game_json,
                    game_path=raw_abs,
                    goal_obj_types=goal_obj_types,
                    goal_rec_types=goal_rec_types,
                    dest_split="temp_train",
                    project=config.experiment.project,
                    step=current_epoch,
                    node_index=node_index,
                    attempt_id=attempt_id,   # <<< IMPORTANT (avoid overwrite)
                )
            except Exception:
                continue

            temp_game_id = canonical_game_id(new_game_path, env_data_dir)
            cand_records.append({
                "slot_id": sid,
                "attempt_id": int(attempt_id),
                "new_game_path": new_game_path,
                "temp_game_id": temp_game_id,
                "req": req,
                "out_text": out_text,
            })

        # --- validate deployability via purge_bad_trials (textworld.start smoke test) ---
        BASE = os.path.join(env_data_dir, "json_2.1.1", config.experiment.project, "temp_train")
        kept_dirs, bad_dirs = purge_bad_trials(BASE, node_index, current_epoch, delete=True)
        kept_game_paths = set(os.path.join(td, "game.tw-pddl") for td in kept_dirs)

        # --- choose ONE deployable candidate per slot; delete the rest ---
        by_slot = {}
        for rec in cand_records:
            by_slot.setdefault(rec["slot_id"], []).append(rec)

        for sid, lst in by_slot.items():
            lst.sort(key=lambda x: x["attempt_id"])

            chosen = None
            for rec in lst:
                gp = rec["new_game_path"]
                if gp in kept_game_paths and os.path.exists(gp):
                    chosen = rec
                    break

            if chosen is None:
                # no deployable env for this slot in this round -> cleanup leftovers
                for rec in lst:
                    try:
                        delete_trial_dir_for_game(env_data_dir, project_name, rec["temp_game_id"])
                    except Exception:
                        pass
                continue

            # add pending (ONE per slot)
            req = chosen["req"]
            add_pending(st, {
                "slot_id": sid,
                "goal": req["goal"],
                "prev_acc": float(req["prev_acc"]),
                "created_epoch": current_epoch,
                "raw_game_id": req["raw_game_id"],
                "parent_active_game_id": req["parent_active_game_id"],
                "temp_game_id": chosen["temp_game_id"],
                "env_prompt": req["env_prompt"],
                "env_full_output": chosen["out_text"],
                "attempt_id": chosen["attempt_id"],
                "failed_rollout_summaries": req.get("failed_rollout_summaries", []),
            })

            # delete other candidates (even if kept) to avoid clutter
            for rec in lst:
                if rec is chosen:
                    continue
                try:
                    delete_trial_dir_for_game(env_data_dir, project_name, rec["temp_game_id"])
                except Exception:
                    pass

        save_env_state(env_data_dir, project_name, st, node_index=node_index)


    stop_workers(task_queues, result_queues, processes)

    time.sleep(5)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    output_file = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
