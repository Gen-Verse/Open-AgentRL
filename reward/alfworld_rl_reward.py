import json
import os
from termcolor import cprint
from omegaconf import OmegaConf

import sys
SAMPLE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sample"))
if SAMPLE_DIR not in sys.path:
    sys.path.insert(0, SAMPLE_DIR)

from alfworld_utils import (
    load_env_state, save_env_state,
    move_trial_temp_to_syn, delete_trial_dir_for_game,
    remove_pending_by_slot,
)

def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)
    return conf


def compute_acc(if_success_list):
    if not if_success_list:
        return 0.0
    corr = sum(1 for x in if_success_list if int(x) == 1)
    return corr / len(if_success_list)


def normalize_process_list(policy_r_list, eps: float = 1e-8):
    out = [list(seq) for seq in policy_r_list]
    if not out:
        return out
    max_len = max(len(seq) for seq in out)
    R = len(out)
    for k in range(max_len):
        idxs = [j for j in range(R) if k < len(out[j])]
        if len(idxs) <= 1:
            continue
        vals = [out[j][k] for j in idxs]
        mu = sum(vals) / len(vals)
        var = sum((v - mu) ** 2 for v in vals) / len(vals)
        std = var ** 0.5
        if std <= eps:
            for j in idxs:
                out[j][k] = 0.0
        else:
            for j, v in zip(idxs, vals):
                out[j][k] = (v - mu) / std
    return out


def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


if __name__ == "__main__":
    config = get_config()

    project_name = config.experiment.project
    current_epoch = int(config.experiment.current_epoch)

    # ---- model path ----
    if current_epoch == 1:
        pretrained_model = config.model.policy_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name

    # ---- outputs name ----
    if config.experiment.function == "train":
        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + config.dataset.environment_type
    else:
        outputs_name = (
            "eval-" + pretrained_model.replace("/", ".") + "-" + config.dataset.environment_type
            + "-" + config.dataset.alfworld_eval_type
        )

    file_name = "../" + project_name + "/temp_data/outputs-" + outputs_name + ".json"

    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ---------------- metrics ----------------
    response_list = []
    max_prompt_list = []
    num_all = 0
    num_success = 0
    sum_success = 0
    max_interaction_step = int(config.rollout.policy.max_interaction_step)

    for i in range(len(data)):
        # token stats
        for j in range(len(data[i].get("prompt", []))):
            if not data[i]["prompt"][j]:
                continue
            max_prompt_list.append(data[i]["prompt"][j][-1])
            response_list.extend(data[i]["response"][j])

        # acc stats
        if_success = data[i].get("if_success", [])
        for j in range(len(if_success)):
            num_all += 1
            if int(if_success[j]) == 1:
                num_success += 1
                sum_success += int(data[i]["success_steps"][j])
            else:
                sum_success += max_interaction_step

        data[i]["acc"] = compute_acc(if_success)

    acc = num_success / num_all if num_all > 0 else 0.0
    avg_step = sum_success / num_all if num_all > 0 else 0.0

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    def get_lengths(strings, tokenizer, type):
        if not strings:
            return 0
        token_lens = [len(tokenizer.encode(s, add_special_tokens=False)) for s in strings]
        if type == "mean":
            return sum(token_lens) / len(token_lens)
        elif type == "max":
            return max(token_lens)

    avg_response_length = get_lengths(response_list, tokenizer, "mean")
    max_prompt_length = get_lengths(max_prompt_list, tokenizer, "max")

    # ---------------- evaluation: just dump metrics ----------------
    if config.experiment.function == "evaluation":
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        outputs_result_name = "../" + project_name + "/results/results-" + outputs_name + ".txt"
        os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
        with open(outputs_result_name, "a") as f:
            def save_and_print(text):
                cprint("\n\n\n" + text, color="green")
                f.write(text + "\n")
            save_and_print(
                f"step: {current_epoch}   acc: {acc}   avg step: {avg_step}   "
                f"avg response length: {avg_response_length}   max_prompt_length: {max_prompt_length}"
            )
        raise SystemExit(0)

    env_data_dir = config.dataset.environment_data_dir

    # cache env_state per node
    st_cache = {}  # node_idx -> state
    moved = deleted = missing = skipped = 0

    def get_node_idx(item):

        return _safe_int(item.get("node_index"), 0)

    for item in data:
        if item.get("source") != "pending":
            continue

        node_idx = get_node_idx(item)
        item["node_index"] = node_idx  

        st = st_cache.get(node_idx)
        if st is None:
            st = load_env_state(env_data_dir, project_name, node_index=node_idx)
            st_cache[node_idx] = st

        now_acc = float(item.get("acc", 0.0))
        prev_acc = item.get("prev_acc", None)
        goal = item.get("goal", None)
        temp_game_id = item.get("game_id", None)

        ok = False
        if prev_acc is not None and goal in ("harder", "easier"):
            try:
                prev_acc_f = float(prev_acc)
                if now_acc > 0:
                    if goal == "harder":
                        ok = (now_acc < prev_acc_f) & (now_acc >= 0.2)
                    else:
                        ok = (now_acc > prev_acc_f) & (now_acc <= 0.8)
            except Exception:
                ok = False

        item["temp_accept"] = bool(ok)

        sid = str(item.get("slot_id"))
        if (not sid) or (not temp_game_id):
            skipped += 1
            continue

        if ok:
            try:
                new_syn_game_id = move_trial_temp_to_syn(env_data_dir, project_name, temp_game_id)
            except FileNotFoundError:
                missing += 1
                remove_pending_by_slot(st, sid)
                continue
            except Exception as e:
                print(f"[WARN] move_trial_temp_to_syn failed node {node_idx} slot {sid}: {e}")
                missing += 1
                remove_pending_by_slot(st, sid)
                continue

            slot = st["slots"].get(sid)
            if slot is None:

                remove_pending_by_slot(st, sid)
                moved += 1
                continue

            prev_active = slot.get("active_game_id", slot.get("raw_game_id"))
            slot["active_game_id"] = new_syn_game_id
            slot["active_type"] = "syn"
            slot["active_epoch"] = current_epoch

            slot.setdefault("history", [])
            slot["history"].append({
                "event": "accept",
                "epoch": current_epoch,
                "goal": goal,
                "prev_acc": float(prev_acc),
                "acc_after": float(now_acc),
                "prev_active_game_id": prev_active,
                "temp_game_id": temp_game_id,
                "new_active_game_id": new_syn_game_id,
            })

            remove_pending_by_slot(st, sid)
            moved += 1
        else:
            try:
                delete_trial_dir_for_game(env_data_dir, project_name, temp_game_id)
            except FileNotFoundError:
                missing += 1
            except Exception as e:
                print(f"[WARN] delete_trial_dir_for_game failed node {node_idx} slot {sid}: {e}")
            remove_pending_by_slot(st, sid)
            deleted += 1

    # save all touched env_states
    for node_idx, st in st_cache.items():
        save_env_state(env_data_dir, project_name, st, node_index=node_idx)

    # ============================================================
    # Build RL training data (pending only if accepted)
    # ============================================================
    num_rollout_per_trial = int(config.rollout.policy.num_rollout_per_trial)
    num_rollout_per_query = int(config.rollout.reward.num_rollout_per_query)

    policy_prompt_list = []
    policy_response_list = []
    policy_reward_list = []
    reward_prompt_list = []
    reward_response_list = []
    reward_reward_list = []

    for i in range(len(data)):
        is_pending = (data[i].get("source") == "pending")
        if is_pending and not bool(data[i].get("temp_accept", False)):
            continue

        if_success = data[i].get("if_success", [])
        if not if_success:
            continue
        acc_i = compute_acc(if_success)
        if acc_i < 1/8 or acc_i > 7/8:
            continue

        extracted_reward = data[i].get("extracted_reward", [])
        if not extracted_reward:
            continue

        # policy model step-wise rewards
        policy_r_list = []
        for j in range(num_rollout_per_trial):
            outcome_reward = float(data[i]["if_success"][j])
            policy_r = []
            steps = extracted_reward[j] if j < len(extracted_reward) else []
            for k in range(len(steps)):
                sco = steps[k]
                mean_score = (sum(sco) / len(sco)) if (isinstance(sco, list) and len(sco) > 0) else 0.0
                policy_r.append(outcome_reward + mean_score)
            policy_r_list.append(policy_r)

        policy_r_list = normalize_process_list(policy_r_list)

        for j in range(num_rollout_per_trial):
            steps = extracted_reward[j] if j < len(extracted_reward) else []
            for k in range(len(steps)):
                if k < len(policy_r_list[j]) and policy_r_list[j][k] != 0:
                    policy_prompt_list.append(data[i]["prompt"][j][k])
                    policy_response_list.append(data[i]["response"][j][k])
                    policy_reward_list.append(policy_r_list[j][k])
        

        # ============================================================
        # reward model training data (normalize across (j, l) together per task)
        # ============================================================

        if acc_i < 0.2 or acc_i > 0.8:
            continue

        for j in range(num_rollout_per_trial):
            outcome_reward = float(data[i]["if_success"][j])
            steps = extracted_reward[j] if j < len(extracted_reward) else []

            for k in range(len(steps)):
                step_scores = steps[k]

                if not isinstance(step_scores, list) or len(step_scores) == 0:
                    continue

                mean_score = sum(step_scores) / len(step_scores)

                # raw reward for each l at this (j, k)
                raw_l = []
                for l in range(num_rollout_per_query):
                    sco = step_scores[l] if l < len(step_scores) else 0.0
                    if sco == 1 or sco == -1:
                        raw_l.append((outcome_reward + mean_score) * sco)
                    else:
                        raw_l.append(-2)

                if len(raw_l) <= 1:
                    continue

                mu = sum(raw_l) / len(raw_l)
                var = sum((x - mu) ** 2 for x in raw_l) / len(raw_l)
                if var == 0.0:
                    continue

                std = var ** 0.5
                norm_l = [(x - mu) / std for x in raw_l]

                # write reward training tuples
                for l in range(num_rollout_per_query):
                    r = norm_l[l]
                    if r != 0:
                        reward_prompt_list.append(data[i]["reward_prompt"][j][k][l])
                        reward_response_list.append(data[i]["reward_response"][j][k][l])
                        reward_reward_list.append(r)

    # write RL data
    if config.experiment.function == "train":
        final_data = []
        for i in range(len(policy_prompt_list)):
            final_data.append({
                "prompt": policy_prompt_list[i],
                "response": policy_response_list[i],
                "reward": policy_reward_list[i],
            })
        with open("../" + project_name + "/temp_data/" + config.dataset.optimization_data + ".json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

        final_data = []
        for i in range(len(reward_prompt_list)):
            final_data.append({
                "prompt": reward_prompt_list[i],
                "response": reward_response_list[i],
                "reward": reward_reward_list[i],
            })
        with open("../" + project_name + "/temp_data/" + config.dataset.reward_optimization_data + ".json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)

    # save outputs with accept marks
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    outputs_result_name = "../" + project_name + "/results/results-" + outputs_name + ".txt"
    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    with open(outputs_result_name, "a") as f:
        def save_and_print(text):
            cprint("\n\n\n" + text, color="green")
            f.write(text + "\n")

        save_and_print(
            f"step: {current_epoch}   acc: {acc}   avg step: {avg_step}   "
            f"avg response length: {avg_response_length}   max_prompt_length: {max_prompt_length}   "
            f"moved: {moved}   deleted: {deleted}   missing: {missing}   skipped: {skipped}"
        )
