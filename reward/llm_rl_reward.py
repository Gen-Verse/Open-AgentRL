# -*- coding: utf-8 -*-
import os
import json
import sys
from typing import List, Any, Dict

from termcolor import cprint
from omegaconf import OmegaConf, MISSING


############################
# Config
############################
def get_config():
    cli_conf = OmegaConf.from_cli()
    yaml_conf = OmegaConf.load(cli_conf.config)
    return OmegaConf.merge(yaml_conf, cli_conf)


############################
# Reward helpers
############################
def z_score_normalize(lst: List[float]) -> List[float]:
    if not lst:
        return []
    mean = sum(lst) / len(lst)
    var = sum((x - mean) ** 2 for x in lst) / len(lst)
    std = var ** 0.5
    if std == 0:
        return [0.0 for _ in lst]
    return [(x - mean) / std for x in lst]


def _pass_all(row: Any) -> bool:
    # "pass all" requires row is a non-empty list and all entries are truthy
    return isinstance(row, list) and len(row) > 0 and all(bool(x) for x in row)


def _safe_list(x):
    return x if isinstance(x, list) else []


############################
# Main
############################
if __name__ == "__main__":
    config = get_config()
    project_name = config.experiment.project
    num_node = int(OmegaConf.select(config, "experiment.num_node", default=1))
    node_index = int(OmegaConf.select(config, "experiment.node_index", default=0))

    # resolve policy model name like your old script
    if int(config.experiment.current_epoch) == 1:
        pretrained_model = config.model.policy_model
    else:
        pretrained_model = "../" + project_name + "/ckpt/" + config.model.optimized_name

    fn = str(config.experiment.function)
    is_train = (fn == "train")
    is_eval = (fn in ("eval", "evaluation", "evaluate", "test"))

    if is_train:
        dataset = config.dataset.train_dataset
        outputs_name = "rl-" + pretrained_model.replace("/", ".") + "-" + dataset
    else:
        dataset = (
            config.dataset.eval_dataset
            if OmegaConf.select(config, "dataset.eval_dataset")
            else config.dataset.train_dataset
        )
        outputs_name = "eval-" + pretrained_model.replace("/", ".") + "-" + dataset

    # outputs file (assumed already aggregated + executed by your separate execution script)
    file_name = f"../{project_name}/temp_data/outputs-{outputs_name}.json"

    with open(file_name, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("outputs file must be a list of task dicts")

    # ------------------------------------------------------------
    # 2) Compute acc (code passes all GT UTs)
    #   NOTE: assumes `correctness` + `syn_correctness` already exist.
    # ------------------------------------------------------------
    response_length_list = []
    pass_all_gt_list = []  # per code sample
    for it in data:
        if isinstance(it.get("response_length", None), list):
            response_length_list.extend(it["response_length"])

        corr_gt = _safe_list(it.get("correctness", []))
        for row in corr_gt:
            pass_all_gt_list.append(1.0 if _pass_all(row) else 0.0)

        # store task-level acc over responses (GT strict)
        gt_pass_all = [bool(_pass_all(row)) for row in corr_gt] if corr_gt else []
        it["gt_pass_all"] = gt_pass_all
        it["acc"] = (sum(1 for x in gt_pass_all if x) / len(gt_pass_all)) if gt_pass_all else 0.0

    overall_acc = (sum(pass_all_gt_list) / len(pass_all_gt_list)) if pass_all_gt_list else 0.0
    avg_len = (sum(response_length_list) / len(response_length_list)) if response_length_list else 0.0

    # ------------------------------------------------------------
    # Syn UT metrics (global) -- based ONLY on syn_correctness
    #
    # syn_ut_acc:
    #   UT is correct if it passes ALL gt codes for that task.
    #   Exclude tasks with NO gt code.
    #
    # perfect_ut_rate:
    #   UT is perfect if it passes ALL gt codes AND fails ALL non-gt codes.
    #   Only count tasks that have BOTH gt and non-gt codes.
    # ------------------------------------------------------------
    syn_ut_total_gt = 0
    syn_ut_correct_total = 0

    syn_ut_total_gt_ngt = 0
    syn_ut_perfect_total = 0

    # ------------------------------------------------------------
    # 3) Build training data
    #   - policy: reward = pass ratio
    #   - reward model: NEW spec (your requested definition)
    # ------------------------------------------------------------
    final_data = []          # policy RL data
    reward_final_data = []   # reward-model RL data (UT generator)

    # optional: keep old behavior for UT rewards (z-score) via config
    ut_zscore = bool(OmegaConf.select(config, "reward_model.z_score", default=False))

    for it in data:
        # ----- temp filtering same as your old style -----
        is_temp = (it.get("source") == "temp")
        is_accept_temp = (is_temp and bool(it.get("temp_accept", False)))
        if is_temp and not is_accept_temp:
            continue

        codes = _safe_list(it.get("extracted_output", []))
        full_outputs = _safe_list(it.get("full_output", []))
        prompt = it.get("prompt", "")

        corr_gt = _safe_list(it.get("correctness", []))            # [m_code][m_gt]
        corr_syn = _safe_list(it.get("syn_correctness", []))       # [m_code][m_syn]  (bool or +/-1)
        m_code = len(codes)

        # ----- optional truncation penalty (same spirit as old script) -----
        lengths = _safe_list(it.get("response_length", []))
        if lengths and m_code == len(lengths) and isinstance(corr_gt, list):
            for j in range(min(m_code, len(corr_gt))):
                if not isinstance(corr_gt[j], list):
                    continue
                if OmegaConf.select(config, "rollout.max_gen_length", default=MISSING) is not MISSING:
                    if lengths[j] >= int(config.rollout.max_gen_length) - 5:
                        corr_gt[j] = [False for _ in corr_gt[j]]
                if OmegaConf.select(config, "rollout.max_token", default=MISSING) is not MISSING:
                    if lengths[j] >= int(config.rollout.max_token) - 5:
                        corr_gt[j] = [False for _ in corr_gt[j]]

        # ============================================================
        # syn_ut_acc + perfect_ut_rate  (DO NOT depend on syn_input keys)
        #   - K_syn derived from syn_correctness matrix itself
        #   - exclude tasks with no gt code (gt_code_idx empty)
        # ============================================================
        # gt codes: pass all GT UTs
        gt_code_idx: List[int] = []
        for j in range(min(m_code, len(corr_gt))):
            if _pass_all(corr_gt[j]):
                gt_code_idx.append(j)
        gt_set = set(gt_code_idx)
        non_gt_idx = [j for j in range(m_code) if j not in gt_set]

        # derive K_syn from syn_correctness matrix
        syn_rows = [row for row in corr_syn if isinstance(row, list)]
        K_syn = min((len(row) for row in syn_rows), default=0)  # robust if some rows shorter

        def syn_ok(j: int, k: int) -> bool:
            if j >= len(corr_syn) or not isinstance(corr_syn[j], list) or k >= len(corr_syn[j]):
                return False
            v = corr_syn[j][k]
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return v > 0
            return bool(v)

        if K_syn > 0 and len(gt_code_idx) > 0:
            # syn_ut_acc denominator counts ONLY tasks with >=1 gt code
            syn_ut_total_gt += K_syn
            for k in range(K_syn):
                if all(syn_ok(j, k) for j in gt_code_idx):
                    syn_ut_correct_total += 1

            # perfect_ut_rate denominator counts ONLY tasks with both gt and non-gt codes
            if len(non_gt_idx) > 0:
                syn_ut_total_gt_ngt += K_syn
                for k in range(K_syn):
                    pass_all_gt_codes = all(syn_ok(j, k) for j in gt_code_idx)
                    if not pass_all_gt_codes:
                        continue
                    fail_all_non_gt = all((not syn_ok(j, k)) for j in non_gt_idx)
                    if fail_all_non_gt:
                        syn_ut_perfect_total += 1

        # ============================================================
        # 3.1 policy reward: per response pass ratio
        # ============================================================
        raw_policy_rewards: List[float] = []
        for j in range(m_code):
            gt_row = corr_gt[j] if (j < len(corr_gt) and isinstance(corr_gt[j], list)) else []

            total = len(gt_row)
            passed = sum(1 for x in gt_row if bool(x))
            r = (passed / total) if total > 0 else 0.0
            raw_policy_rewards.append(float(r))

        it["raw_policy_reward"] = raw_policy_rewards
        policy_rewards = z_score_normalize(raw_policy_rewards)
        it["rewards"] = policy_rewards

        # 训练样本过滤：和你旧版一样，避免太容易/太难（可按需删）
        if is_train:
            # strict GT proportion
            gt_pass_all = _safe_list(it.get("gt_pass_all", []))
            proportion = (sum(1 for x in gt_pass_all if x) / len(gt_pass_all)) if gt_pass_all else 0.0
            if proportion > 0.8 or proportion < 0.2:
                continue

            for j in range(min(len(policy_rewards), len(full_outputs))):
                final_data.append(
                    {
                        "prompt": prompt,
                        "reward": float(policy_rewards[j]),
                        "response": full_outputs[j],
                        "raw_reward": float(raw_policy_rewards[j]),
                    }
                )

        # ============================================================
        # 3.2 reward model data (UT generator): YOUR NEW SPEC
        #
        # Definitions:
        #   - GT code: pass ALL GT dataset UTs (we already computed gt_code_idx)
        #   - GT generated UT: pass ALL GT codes (under syn_correctness)
        #
        # For generated UT k:
        #   - if UT is GT:
        #       reward = (# of non-GT codes that FAIL this UT)  [i.e., distinguished]
        #   - else (UT is non-GT):
        #       reward = - (# of non-GT codes that PASS this UT) [i.e., NOT distinguished]
        #   - if there is NO non-GT code:
        #       reward = +1 if UT is GT else -1
        #
        # Optional: per-task z-score controlled by config reward_model.z_score (default False)
        # ============================================================
        syn_inputs = _safe_list(it.get("syn_input", []))
        syn_outputs = _safe_list(it.get("syn_output", []))
        syn_full = _safe_list(it.get("syn_full_output", []))
        syn_prompts = _safe_list(it.get("syn_prompt", []))  # you said you will add in rollout

        # K from text fields (for making training examples)
        K_text = 0
        if syn_full:
            K_text = min(len(syn_inputs), len(syn_outputs), len(syn_full))
        else:
            K_text = min(len(syn_inputs), len(syn_outputs))

        # Must align with syn_correctness width
        K = min(K_text, K_syn)
        if K <= 0:
            continue

        # If no GT code, the "GT UT" concept is not meaningful; skip (same spirit as syn_ut_acc)
        if len(gt_code_idx) == 0:
            continue

        raw_ut_rewards: List[float] = []
        ut_meta: List[Dict[str, Any]] = []

        for k in range(K):
            # basic validity flag (kept for debugging; does NOT affect reward)
            inp = syn_inputs[k] if k < len(syn_inputs) else ""
            out = syn_outputs[k] if k < len(syn_outputs) else ""
            is_valid_ut = isinstance(inp, str) and isinstance(out, str) and inp != "" and out != ""

            # GT UT: passes ALL GT codes
            is_gt_ut = all(syn_ok(j, k) for j in gt_code_idx)

            if len(non_gt_idx) == 0:
                # no non-GT codes: reward determined only by whether UT is GT
                r = 1.0 if is_gt_ut else -1.0
                n_distinguish = 0
                n_not_distinguish = 0
            else:
                # count distinguish / not-distinguish among NON-GT codes
                n_distinguish = sum(1 for j in non_gt_idx if not syn_ok(j, k))  # fail non-gt => distinguished
                n_not_distinguish = sum(1 for j in non_gt_idx if syn_ok(j, k))  # pass non-gt => not distinguished

                if is_gt_ut:
                    r = float(n_distinguish)
                else:
                    r = -float(n_not_distinguish)

            raw_ut_rewards.append(float(r))

            ut_meta.append(
                {
                    "k": int(k),
                    "valid_ut": bool(is_valid_ut),
                    "is_gt_ut": bool(is_gt_ut),
                    "raw_reward": float(r),
                    "n_code": int(m_code),
                    "n_gt_code": int(len(gt_code_idx)),
                    "n_non_gt_code": int(len(non_gt_idx)),
                    "distinguish_non_gt": int(n_distinguish),
                    "not_distinguish_non_gt": int(n_not_distinguish),
                }
            )

        ut_rewards = z_score_normalize(raw_ut_rewards)

        if is_train:
            for k in range(K):
                pmt = syn_prompts[k] if (k < len(syn_prompts) and isinstance(syn_prompts[k], str)) else it.get("prompt", "")
                resp = syn_full[k] if (k < len(syn_full) and isinstance(syn_full[k], str)) else ""

                reward_final_data.append(
                    {
                        "prompt": pmt,
                        "reward": float(ut_rewards[k]),
                        "response": resp,
                        "raw_reward": float(raw_ut_rewards[k]),
                        "meta": ut_meta[k],
                        "evaluated_task_prompt": it.get("prompt", ""),
                        "evaluated_task_question": it.get("question", ""),
                    }
                )

    # ------------------------------------------------------------
    # 4) write back + log + export training data
    # ------------------------------------------------------------
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    outputs_result_name = f"../{project_name}/results/results-{outputs_name}.txt"
    os.makedirs(os.path.dirname(outputs_result_name), exist_ok=True)
    with open(outputs_result_name, "a", encoding="utf-8") as f:
        def save_and_print(text: str):
            cprint("\n" + text, color="green")
            f.write(text + "\n")

        syn_ut_acc = (syn_ut_correct_total / syn_ut_total_gt) if syn_ut_total_gt > 0 else 0.0
        perfect_ut_rate = (syn_ut_perfect_total / syn_ut_total_gt_ngt) if syn_ut_total_gt_ngt > 0 else 0.0

        save_and_print(
            f"train step: {int(config.experiment.current_epoch)}  "
            f"GT-pass-all acc: {overall_acc:.6f}  avg length: {avg_len:.2f}  "
            f"syn_ut_acc: {syn_ut_acc:.6f}  perfect_ut_rate: {perfect_ut_rate:.6f}  "
            f"(syn_ut_den_gt={syn_ut_total_gt}, syn_ut_den_gt_ngt={syn_ut_total_gt_ngt})"
        )

    if is_eval:
        sys.exit(0)

    # policy RL data
    with open(
        f"../{project_name}/temp_data/{config.dataset.optimization_data}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)

    # reward-model RL data (UT generator)
    with open(
        f"../{project_name}/temp_data/{config.dataset.reward_optimization_data}.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(reward_final_data, f, indent=2, ensure_ascii=False)
