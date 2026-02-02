

import os
import torch
from pathlib import Path
from typing import Any, Dict, List, Tuple

from omegaconf import OmegaConf
from train.utils import get_config


def _maybe_reapply_truncation(
    seq: torch.Tensor,
    start_pos: int,
    max_prompt_len: int,
    max_gen_length: int,
    has_image: bool,
) -> Tuple[torch.Tensor, int, bool]:
    L = int(seq.numel())
    prompt_len = int(start_pos)
    resp_len = max(0, L - prompt_len)

    # prompt truncate
    if max_prompt_len > 0 and prompt_len > max_prompt_len:
        if has_image:
            return seq, start_pos, False
        cut = prompt_len - max_prompt_len
        seq = seq[cut:]
        prompt_len = max_prompt_len
        start_pos = prompt_len
        L = int(seq.numel())
        resp_len = max(0, L - prompt_len)

    # response truncate
    if max_gen_length > 0 and resp_len > max_gen_length:
        seq = seq[: prompt_len + max_gen_length]
        L = int(seq.numel())

    if L == 0:
        return seq, start_pos, False
    return seq, start_pos, True


def _vl_family_from_meta(meta: Dict[str, Any]) -> str:
    if "vl_family" in meta:
        return str(meta["vl_family"])
    # backward compat
    mt = str(meta.get("model_type", "") or "").lower()
    if "opencua" in mt:
        return "opencua"
    if mt == "qwen3vl":
        return "qwen3vl"
    if mt == "uitars15":
        return "qwen25vl"
    return "qwen3vl" if bool(meta.get("is_qwen3_vl", False)) else "other"

def merge_shards_to_padded_dataset(project_name: str, optimization_data: str, cfg) -> Dict[str, Any]:
    shard_dir = Path(project_name) / "temp_data"
    shard_paths = sorted(shard_dir.glob(f"{optimization_data}_preproc_node*_of*.pt"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found under {shard_dir} for {optimization_data}")

    packs = [torch.load(p, map_location="cpu") for p in shard_paths]

    # --- meta consistency check ---
    pad_id = int(packs[0]["meta"]["pad_token_id"])
    vl_family = _vl_family_from_meta(packs[0]["meta"])
    total_size = int(packs[0]["meta"]["total_size"])
    max_prompt_len_cap = int(packs[0]["meta"].get("max_prompt_len_cap", 0))
    max_gen_length_cap = int(packs[0]["meta"].get("max_gen_length_cap", 0))

    for pk in packs[1:]:
        assert int(pk["meta"]["pad_token_id"]) == pad_id, "pad_token_id mismatch"
        assert _vl_family_from_meta(pk["meta"]) == vl_family, "vl_family mismatch"
        assert int(pk["meta"]["total_size"]) == total_size, "total_size mismatch"

    # --- get truncation caps from cfg (same as training) ---
    if cfg.training.target == "policy":
        max_prompt_len = int(cfg.training.policy.max_prompt_len)
        max_gen_length = int(cfg.training.policy.max_gen_length)
    else:
        max_prompt_len = int(cfg.training.reward.max_prompt_len)
        max_gen_length = int(cfg.training.reward.max_gen_length)

    if max_prompt_len_cap != 0 and max_prompt_len != max_prompt_len_cap:
        raise RuntimeError(f"max_prompt_len mismatch: cfg={max_prompt_len} shard_meta={max_prompt_len_cap}")
    if max_gen_length_cap != 0 and max_gen_length != max_gen_length_cap:
        raise RuntimeError(f"max_gen_length mismatch: cfg={max_gen_length} shard_meta={max_gen_length_cap}")

    # --- unpack all samples ---
    all_seqs: List[torch.Tensor] = []
    all_start: List[int] = []
    all_adv: List[float] = []
    pixel_values_list: List[Any] = []
    grid_thws_list: List[Any] = []

    skipped_merge = 0

    for pk in packs:
        flat = pk["flat_input_ids"]          # (sum_L,)
        offsets = pk["offsets"]              # (N+1,)
        start_pos = pk["start_pos"]          # (N,)
        adv = pk["advantage"]                # (N,)
        pv_list = pk["pixel_values_list"]    # list len N
        gt_list = pk.get("image_grid_thw_list", None)
        if gt_list is None:
            gt_list = pk.get("grid_thws_list", None)
        if gt_list is None:
            raise KeyError("Shard missing both image_grid_thw_list and grid_thws_list")

        N = int(offsets.numel() - 1)
        assert len(pv_list) == N and len(gt_list) == N, "vision list length mismatch"

        for i in range(N):
            a = int(offsets[i].item())
            b = int(offsets[i + 1].item())
            seq = flat[a:b].clone()

            sp = int(start_pos[i].item())
            rw = float(adv[i].item())

            pv = pv_list[i]
            gt = gt_list[i]
            has_image = (pv is not None) or (gt is not None)

            seq, sp, keep = _maybe_reapply_truncation(
                seq=seq,
                start_pos=sp,
                max_prompt_len=max_prompt_len,
                max_gen_length=max_gen_length,
                has_image=has_image,
            )
            if not keep:
                skipped_merge += 1
                continue

            all_seqs.append(seq)
            all_start.append(sp)
            all_adv.append(rw)
            pixel_values_list.append(pv)
            grid_thws_list.append(gt)

    N = len(all_seqs)
    if N == 0:
        raise ValueError("No samples kept after merging/truncation checks.")

    # --- pad to global max len (prompt+response total) ---
    lengths = torch.tensor([int(s.numel()) for s in all_seqs], dtype=torch.long)
    Lmax = int(lengths.max().item())

    input_ids = torch.full((N, Lmax), pad_id, dtype=torch.long)
    labels   = torch.full((N, Lmax), pad_id, dtype=torch.long)
    p_mask   = torch.zeros((N, Lmax), dtype=torch.bool)
    advantage = torch.tensor(all_adv, dtype=torch.float32)

    for i, seq in enumerate(all_seqs):
        L = int(seq.numel())
        input_ids[i, :L] = seq
        labels[i, :L] = seq
        sp = int(all_start[i])
        if L > sp:
            p_mask[i, sp:L] = True

    meta = dict(packs[0]["meta"])
    meta.update({
        "loaded_shards": [str(p) for p in shard_paths],
        "merged_kept_N": int(N),
        "merged_skipped_extra": int(skipped_merge),
        "merged_global_max_len": int(Lmax),
        "cfg_max_prompt_len": int(max_prompt_len),
        "cfg_max_gen_length": int(max_gen_length),
    })

    return {
        "input_ids": input_ids,
        "labels": labels,
        "p_mask": p_mask,
        "advantage": advantage,
        "pixel_values_list": pixel_values_list,
        "image_grid_thw_list": grid_thws_list,
        "grid_thws_list": grid_thws_list,
        "meta": meta,
    }


def main():
    cfg = get_config()
    project_name = cfg.experiment.project

    # optimization_data name
    if cfg.training.target == "policy":
        optimization_data = "policy_optimization_data"
    elif cfg.training.target == "reward":
        optimization_data = "reward_optimization_data"
    else:
        raise ValueError(f"Unknown training.target = {cfg.training.target}")

    cli = OmegaConf.from_cli()
    save_dir = getattr(cli, "save_dir", None)
    if save_dir is None:
        save_dir = str(Path(project_name) / "temp_data")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    pack = merge_shards_to_padded_dataset(project_name, optimization_data, cfg)

    out_path = save_dir / f"{optimization_data}_preproc_merged.pt"
    torch.save(pack, out_path)
    print(f"[MERGE] saved: {out_path}", flush=True)
    print(f"[MERGE] kept={pack['meta']['merged_kept_N']} "
          f"Lmax={pack['meta']['merged_global_max_len']} "
          f"extra_skipped={pack['meta']['merged_skipped_extra']}", flush=True)


if __name__ == "__main__":
    main()
