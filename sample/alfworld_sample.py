import os
import re
import ast
import json
import random
import argparse
from jinja2 import Template
from termcolor import cprint
import multiprocessing as mp
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer







os.environ["TOKENIZERS_PARALLELISM"] = "false" 





####### vllm inference #######

def worker_fn(pretrained_model, gpu_ids, task_queue, result_queue, max_model_len, max_generation_token, temp):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    print(f"Loading model on GPUs {gpu_ids}...")
    llm = LLM(
        model=pretrained_model,
        dtype="bfloat16",
        tensor_parallel_size=len(gpu_ids),
        gpu_memory_utilization=0.85,
        max_model_len=max_model_len
    )

    sampling_params = SamplingParams(
        temperature=temp,
        top_p=0.95,
        top_k=-1,
        min_p=0.0,
        max_tokens=max_generation_token,
        stop=["</answer>", "User:", "Human:", "Assistant:", "<|im_end|>", "<|endoftext|>"]
    )

    while True:
        task = task_queue.get()
        if task == "STOP":
            print("Stopping worker...")
            break
        task_id, prompts = task

        if not prompts:                
            result_queue.put((task_id, []))
            continue

        outputs = llm.generate(prompts, sampling_params)
        result_texts = [out.outputs[0].text for out in outputs]
        result_queue.put((task_id, result_texts))

# To run the worker setup:
def start_workers(pretrained_model, gpu_configs, max_model_len, max_generation_token, temp):
    task_queues = []
    result_queues = []
    processes = []

    for i, gpu_ids in enumerate(gpu_configs):
        task_q = mp.Queue()
        result_q = mp.Queue()
        p = mp.Process(
            target=worker_fn,
            args=(pretrained_model, gpu_ids, task_q, result_q, max_model_len, max_generation_token, temp)
        )
        p.start()
        task_queues.append(task_q)
        result_queues.append(result_q)
        processes.append(p)
    
    return task_queues, result_queues, processes

# Stop workers
def stop_workers(task_queues, processes):
    for q in task_queues:
        q.put("STOP")
    for p in processes:
        p.join()

# Split prompts into N chunks
def split_prompts(prompts, n):
    k, m = divmod(len(prompts), n)
    return [prompts[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def get_token_lengths(strings, tokenizer):
    return [len(tokenizer.encode(s, add_special_tokens=False)) for s in strings]

# vllm inference
def generate_results(all_prompts, gpu_groups, task_queues, result_queues):
    chunks = split_prompts(all_prompts, len(gpu_groups))
    jobs = []  # (qid, chunk_idx)
    for i, (q, prompts) in enumerate(zip(task_queues, chunks)):
        if len(prompts) == 0:
            continue
        q.put((i, prompts))
        jobs.append(i)

    results_by_job = {}
    remaining = set(jobs)
    while remaining:
        for i, rq in enumerate(result_queues):
            if i not in remaining:
                continue
            try:
                task_id, result = rq.get(timeout=0.05)
            except Exception:
                continue
            results_by_job[task_id] = result
            remaining.remove(task_id)

    result_list = []
    for i, prompts in enumerate(chunks):
        if len(prompts) == 0:
            continue
        result_list.extend(results_by_job[i])
    return result_list





######## process environment trajectory ########

def as_list(x):
    return x if isinstance(x, (list, tuple)) else [x]

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
        infos = [x if isinstance(x, dict) else {} for x in (infos + [{}]*B)[:B]]
        return infos
    return [{} for _ in range(B)]

def build_step_prompts(histories, infos, is_think):
    """
    return: List[str]  — one prompt for one batch
    """
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
        prompt_i = f"""<|im_start|>You are a helpful assistant. <|im_end|>\n<|im_start|>user
{guide}

{traj}

You need to think step by step then choose one action by number:
{options}
<|im_end|>\n<|im_start|>assistant"""
        if is_think:
            prompt_i = prompt_i + "<think>"
        prompts.append(prompt_i)
        
    return prompts

from typing import List

def map_idxlist_to_actions(indices: List[int], infos, B: int) -> List[str]:
    infos_b = _align_infos(infos, B)
    idxs = (list(indices) + [0]*B)[:B]  
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
            continue  # already success
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






####### other functions #########

import random 
def random_select(data_list, random_k):
    data_list = random.sample(data_list, random_k)
    return data_list

def get_data_chunk(data, num_nodes, node_idx):
    total = len(data)
    start = (total * node_idx) // num_nodes
    end   = (total * (node_idx + 1)) // num_nodes
    return data[start:end]

from omegaconf import DictConfig, ListConfig, OmegaConf
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf

def extract_final_boxed_answer(s: str):
    tag = r'\boxed{'
    start = s.rfind(tag)          # last \boxed{
    if start == -1:
        return "Can not extract the answer!"

    i = start + len(tag)
    depth = 1                    # we are already inside one '{'
    buf = []

    while i < len(s) and depth:
        ch = s[i]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:       # matching '}' for the opening \boxed{
                break
        buf.append(ch)
        i += 1

    return ''.join(buf) if depth == 0 else "Can not extract the answer!"
    

def safe_to_index(s: str, default=0) -> int:
    if s is None:
        return default
    m = re.search(r"-?\d+", str(s))
    try:
        return int(m.group(0)) if m else default
    except Exception:
        return default

if __name__ == "__main__":

    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    config = get_config()

    pretrained_model = config.model
    gpu_groups = config.rollout.gpu_groups
    max_model_len = config.rollout.model_length
    max_generation_token = config.rollout.max_gen_length
    temp = config.rollout.temperature
    num_node = config.experiment.num_node
    node_index = config.experiment.node_index
    num_rollout_per_trial = config.rollout.num_rollout_per_trial

    import os, sys
    os.environ["ALFWORLD_DATA"] = config.dataset.environment_data_dir
    os.environ["ALFWORLD_DATA_PROJECT"] = config.experiment.project
    repo_root = config.dataset.environment_file_dir
    sys.path.insert(0, repo_root)
    sys.argv = ["", config.dataset.environment_file_dir + "/configs/base_config.yaml"]
    from my_environments import AlfredTWEnv as get_environment
    import alfworld.agents.modules.generic as generic
    env_config = generic.load_config()

    env = get_environment(env_config, train_eval=config.dataset.alfworld_data_type)
    env._build_catalog()
    N = len(env.catalog)
    overall_index = [i for i in range(N)]
    #overall_index = [i for i in range(8)]

    if num_node > 1:
        # random.shuffle(data)
        selected_index = get_data_chunk(overall_index, num_node, node_index)
    else:
        selected_index = overall_index

    num = len(selected_index)

    env.subset_and_repeat(indices=selected_index, repeat=num_rollout_per_trial, shuffle=False)

    task_list = [row["task"] for row in env.catalog]
    trial_list = [row["trial"] for row in env.catalog]
    task_path_list = [row["task_path"] for row in env.catalog]

    data = [{} for _ in range(num)]
    for i in range(num):
        data[i]["task"] = task_list[i * num_rollout_per_trial]
        data[i]["trial"] = trial_list[i * num_rollout_per_trial]
        data[i]["task_path"] = task_path_list[i * num_rollout_per_trial]
        data[i]["trajectory"] = [[] for _ in range(num_rollout_per_trial)]
        data[i]["prompt"] = [[] for _ in range(num_rollout_per_trial)]
        data[i]["response"] = [[] for _ in range(num_rollout_per_trial)]
        data[i]["if_success"] = [-1 for _ in range(num_rollout_per_trial)]
        data[i]["success_steps"] = [-1 for _ in range(num_rollout_per_trial)]
        data[i]["init_obs"] = [""] * num_rollout_per_trial  

    # vLLM worker
    task_queues, result_queues, processes = start_workers(pretrained_model, gpu_groups, max_model_len, max_generation_token, temp)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + config.dataset.environment_type + "-" + config.dataset.alfworld_data_type

    CAP = config.rollout.env_max_parallel
    TOTAL = len(env.game_files)

    for offset in range(0, TOTAL, CAP):

        sub_paths   = env.game_files[offset: offset + CAP]
        sub_catalog = env.catalog   [offset: offset + CAP]

        _saved_paths, _saved_catalog = env.game_files, env.catalog
        env.game_files, env.catalog  = sub_paths, sub_catalog

        tw_env = env.init_env(batch_size=len(sub_paths))

        # reset
        obs, infos = tw_env.reset()
        obs   = _as_list(obs)
        infos = _as_list(infos)
        B = len(obs)

        histories = [[{"obs": obs[i], "act": None}] for i in range(B)]
        success_steps = [-1] * B
        per_batch_prompts   = [[] for _ in range(B)]
        per_batch_responses = [[] for _ in range(B)]

        for i_step in range(config.rollout.max_interaction_step):
            if all(s != -1 for s in success_steps):
                cprint("all episodes in this chunk finished. break.", "green")
                break

            prompts = build_step_prompts(histories, infos, config.rollout.if_start_with_think)

            active_index = [i for i, s in enumerate(success_steps) if s == -1]
            target_prompts = [prompts[i] for i in active_index]

            cprint(f"batch {int(offset/CAP)}/{int(TOTAL/CAP)}, step {i_step}/{config.rollout.max_interaction_step}", "green")

            Np = len(target_prompts)
            indices = list(range(Np))
            shuffled_idx = indices[:]
            random.shuffle(shuffled_idx)
            shuffled_prompts = [target_prompts[i] for i in shuffled_idx]
            shuffled_outputs = generate_results(shuffled_prompts, gpu_groups, task_queues, result_queues)
            restored_outputs = [None] * Np
            for out, idx in zip(shuffled_outputs, shuffled_idx):
                restored_outputs[idx] = out

            reply_indices = [safe_to_index(extract_final_boxed_answer(x), default=0) for x in restored_outputs]

            final_indices = []
            full_restored_outputs = []
            ptr = 0
            for i in range(B):
                if success_steps[i] != -1:
                    final_indices.append(0)
                    full_restored_outputs.append("The task is completed.")
                else:
                    final_indices.append(reply_indices[ptr])
                    full_restored_outputs.append(restored_outputs[ptr])
                    ptr += 1

            for k in range(B):
                per_batch_prompts[k].append(prompts[k])
                per_batch_responses[k].append(full_restored_outputs[k])

            actions = map_idxlist_to_actions(final_indices, infos, B)
            obs, scores, dones, infos = tw_env.step(actions)
            dones_list = list(dones) if isinstance(dones, (list, tuple)) else [dones]
            alive_mask = [(success_steps[i] == -1) and (not dones_list[i]) for i in range(B)]

            update_histories(histories, actions, obs, alive_mask=alive_mask)
            success_steps = update_success_steps(success_steps, infos, histories)

        for k in range(B):
            g = offset + k
            trial_idx   = g // num_rollout_per_trial
            rollout_idx = g %  num_rollout_per_trial
            if not (0 <= trial_idx < len(data)):
                continue

            # init obs
            data[trial_idx]["init_obs"][rollout_idx] = histories[k][0]["obs"]

            if success_steps[k] == -1:
                data[trial_idx]["if_success"][rollout_idx]    = -1
                data[trial_idx]["trajectory"][rollout_idx]    = histories[k]
                data[trial_idx]["prompt"][rollout_idx]        = per_batch_prompts[k]
                data[trial_idx]["response"][rollout_idx]      = per_batch_responses[k]
                data[trial_idx]["success_steps"][rollout_idx] = -1
            else:
                cut = success_steps[k] + 1
                data[trial_idx]["if_success"][rollout_idx]    = 1
                data[trial_idx]["trajectory"][rollout_idx]    = histories[k][:cut]
                data[trial_idx]["prompt"][rollout_idx]        = per_batch_prompts[k][:cut]
                data[trial_idx]["response"][rollout_idx]      = per_batch_responses[k][:cut]
                data[trial_idx]["success_steps"][rollout_idx] = cut

        try:
            if tw_env is not None and hasattr(tw_env, "close"):
                tw_env.close()
        except Exception as e:
            print("[warn] tw_env.close() failed:", e)
        finally:
            del tw_env
            env.game_files, env.catalog = _saved_paths, _saved_catalog

    # stop vLLM worker
    stop_workers(task_queues, processes)

    project_name = config.experiment.project
    if num_node > 1:
        output_file_name = "../" + project_name + f"/temp_data/outputs-{node_index}-" + outputs_name + ".json"
    else:
        output_file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"
    os.makedirs(os.path.dirname(output_file_name), exist_ok=True)
    with open(output_file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


    










