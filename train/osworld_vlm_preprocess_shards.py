

import os
import io
import json
import base64
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from omegaconf import OmegaConf
from transformers import AutoTokenizer, AutoImageProcessor, AutoProcessor, AutoConfig


from train.utils import get_config



def collect_images_from_messages(messages: List[Dict[str, Any]]) -> List[Image.Image]:
    pil_images: List[Image.Image] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            for part in content:
                t = part.get("type")
                if t == "image":
                    src = part.get("image")
                    if src:
                        try:
                            pil_images.append(_decode_image(src))
                        except Exception as e:
                            print(f"[WARN] decode image failed: {src} | {e}", flush=True)
                elif t == "image_url":
                    url = (part.get("image_url") or {}).get("url")
                    if url:
                        try:
                            pil_images.append(_decode_image(url))
                        except Exception as e:
                            print(f"[WARN] decode image_url failed: {url} | {e}", flush=True)
    return pil_images


def _decode_image(src: str) -> Image.Image:
    s = src.strip()
    if s.startswith("http://") or s.startswith("https://"):
        import requests
        resp = requests.get(s, timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    if not os.path.exists(s):
        b64 = s.split(",", 1)[1] if "," in s else s
        raw = base64.b64decode(b64, validate=False)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    return Image.open(s).convert("RGB")


def _normalize_qwen_messages(prompt_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:

    qwen_messages: List[Dict[str, Any]] = []
    for m in prompt_messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        new_content: List[Dict[str, Any]] = []

        if isinstance(content, str):
            new_content.append({"type": "text", "text": content})

        elif isinstance(content, list):
            for part in content:
                if isinstance(part, str):
                    new_content.append({"type": "text", "text": part})
                elif isinstance(part, dict):
                    p_type = part.get("type")

                    if p_type in ("text", "paragraph"):
                        new_content.append({"type": "text", "text": part.get("text", "")})

                    elif p_type in ("image", "video"):

                        new_content.append(part)

                    elif p_type == "image_url":

                        url = (part.get("image_url") or {}).get("url") or part.get("url")
                        if url:
                            new_content.append({"type": "image", "image": url})

                    else:

                        txt = part.get("text", None)
                        new_content.append({"type": "text", "text": txt if txt is not None else str(part)})
                else:
                    new_content.append({"type": "text", "text": str(part)})

        elif isinstance(content, dict):

            p_type = content.get("type")
            if p_type in ("text", "paragraph"):
                new_content.append({"type": "text", "text": content.get("text", "")})
            elif p_type in ("image", "video"):
                new_content.append(content)
            elif p_type == "image_url":
                url = (content.get("image_url") or {}).get("url") or content.get("url")
                if url:
                    new_content.append({"type": "image", "image": url})
                else:
                    new_content.append({"type": "text", "text": str(content)})
            else:
                txt = content.get("text", None)
                new_content.append({"type": "text", "text": txt if txt is not None else str(content)})

        else:
            new_content.append({"type": "text", "text": str(content)})

        qwen_messages.append({"role": role, "content": new_content})

    return qwen_messages


def _pack_1d_long_tensors(seqs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    seqs: list of 1D LongTensor with variable length
    returns:
      flat: (sum_L,)
      offsets: (N+1,) int64, offsets[0]=0, offsets[i+1]=offsets[i]+len_i
    """
    if not seqs:
        flat = torch.empty((0,), dtype=torch.long)
        offsets = torch.zeros((1,), dtype=torch.long)
        return flat, offsets

    lengths = torch.tensor([int(s.numel()) for s in seqs], dtype=torch.long)
    offsets = torch.zeros((len(seqs) + 1,), dtype=torch.long)
    offsets[1:] = torch.cumsum(lengths, dim=0)

    flat = torch.empty((int(offsets[-1].item()),), dtype=torch.long)
    cur = 0
    for s in seqs:
        L = int(s.numel())
        flat[cur:cur + L] = s
        cur += L
    return flat, offsets


def main():
    cfg = get_config()
    project_name = cfg.experiment.project

    # -------- choose branch (policy/reward) --------
    if cfg.training.target == "policy":
        if cfg.experiment.current_epoch == 1:
            pretrained_model = cfg.model.policy_model
        else:
            pretrained_model = cfg.system.rl_base_dir + "/" + project_name + "/ckpt/" + cfg.model.optimized_name
        max_prompt_len = int(cfg.training.policy.max_prompt_len)
        max_gen_length = int(cfg.training.policy.max_gen_length)
        optimization_data = "policy_optimization_data"
    elif cfg.training.target == "reward":
        if cfg.experiment.current_epoch == 1:
            pretrained_model = cfg.model.reward_model
        else:
            pretrained_model = cfg.system.rl_base_dir + "/" + project_name + "/ckpt/" + cfg.model.optimized_reward_name
        max_prompt_len = int(cfg.training.reward.max_prompt_len)
        max_gen_length = int(cfg.training.reward.max_gen_length)
        optimization_data = "reward_optimization_data"
    else:
        raise ValueError(f"Unknown training.target = {cfg.training.target}")


    node_idx = int(OmegaConf.select(cfg, "dataset.node_rank", default=os.environ.get("NODE_RANK", 0)))
    num_nodes = int(OmegaConf.select(cfg, "dataset.num_nodes", default=os.environ.get("NUM_NODES", 1)))
    assert num_nodes >= 1
    assert 0 <= node_idx < num_nodes

    cli = OmegaConf.from_cli()
    cast_pixel_fp16 = int(getattr(cli, "cast_pixel_fp16", 1))
    save_dir = getattr(cli, "save_dir", None)
    if save_dir is None:
        save_dir = str(Path(project_name) / "temp_data")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_type = str(getattr(cfg, "model_type", "") or "").lower()

    # ---- HF config based detection (more robust than only cfg.model_type) ----
    try:
        hf_cfg = AutoConfig.from_pretrained(pretrained_model, trust_remote_code=True)
        hf_model_type = str(getattr(hf_cfg, "model_type", "") or "").lower()
        hf_archs = set(getattr(hf_cfg, "architectures", []) or [])
    except Exception as e:
        hf_model_type = ""
        hf_archs = set()
        print(f"[HFConfig] AutoConfig load failed: {e}", flush=True)

    is_hf_qwen3 = (hf_model_type == "qwen3_vl") or any("Qwen3VL" in a for a in hf_archs)
    is_hf_qwen25 = (hf_model_type == "qwen2_5_vl") or any("Qwen2_5" in a for a in hf_archs)
    is_hf_opencua = (hf_model_type == "opencua") or any("OpenCUA" in a for a in hf_archs)

    is_opencua = ("opencua" in model_type) or is_hf_opencua

    if is_opencua:
        vl_family = "opencua"
    elif is_hf_qwen3 or (model_type == "qwen3vl"):
        vl_family = "qwen3vl"
    elif is_hf_qwen25 or (model_type == "uitars15"):
        vl_family = "qwen25vl"   # UI-TARS 1.5 belongs here
    else:
        vl_family = "other"

    # keep old variable name for backward meta meaning:
    # True => "Qwen-VL style processor path" (qwen3vl + qwen25vl)
    is_qwen3_vl = (vl_family in ("qwen3vl", "qwen25vl"))

    # NEW: all three VL families should use processor align
    use_processor_align = (vl_family in ("qwen3vl", "qwen25vl", "opencua"))
    print(f"[ModelDetect] vl_family={vl_family} | use_processor_align={use_processor_align}", flush=True)

    # -------- load and slice --------
    opt_path = Path(project_name) / "temp_data" / f"{optimization_data}.json"
    with opt_path.open("r", encoding="utf-8") as f:
        data_all = json.load(f)
    total_n = len(data_all)
    start = (total_n * node_idx) // num_nodes
    end = (total_n * (node_idx + 1)) // num_nodes
    data = data_all[start:end]

    print(f"[Node {node_idx}/{num_nodes}] range [{start}, {end}) size={len(data)} total={total_n}", flush=True)

    # -------- tokenizer/processor --------
    if use_processor_align:
        if vl_family in ("qwen25vl", "opencua"):
            processor = AutoProcessor.from_pretrained(pretrained_model, trust_remote_code=True, use_fast=False)
        else:
            processor = AutoProcessor.from_pretrained(pretrained_model, trust_remote_code=True)
        tokenizer = processor.tokenizer
        image_processor = getattr(processor, "image_processor", None)
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
        image_processor = AutoImageProcessor.from_pretrained(pretrained_model, trust_remote_code=True, use_fast=False)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    pad_id = int(tokenizer.pad_token_id)

    # -------- preprocess (NO padding) --------
    seq_tensors: List[torch.Tensor] = []
    start_pos_list: List[int] = []
    adv_list: List[float] = []
    pixel_values_list: List[Any] = []
    grid_thws_list: List[Any] = []

    skipped = 0

    for item in data:
        prompt_messages = item["prompt_messages"]
        response_text = item["response"]
        reward = float(item.get("reward", 0.0))

        # ----- prompt ids + vision -----
        pixel_values, grid_thws = None, None

        if use_processor_align:
            qwen_messages = _normalize_qwen_messages(prompt_messages)

            if vl_family == "opencua":
                # OpenCUA MUST match serve_opencua.py:
                #   1) tokenize=False to get raw text with placeholders
                #   2) processor(images=..., text=...) to expand & align
                text = processor.apply_chat_template(
                    qwen_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                pil_images = collect_images_from_messages(prompt_messages)
                if pil_images:
                    enc = processor(images=pil_images, text=text, return_tensors="pt", padding=False)
                else:
                    enc = processor(text=text, return_tensors="pt", padding=False)

                prompt_ids = enc["input_ids"][0].tolist()

                pixel_values = enc.get("pixel_values", None)
                if pixel_values is not None:
                    pixel_values = pixel_values.clone().cpu()

                grid = enc.get("image_grid_thw", None)
                if grid is None:
                    grid = enc.get("grid_thws", None)  # last-resort
                grid_thws = grid.clone().long().cpu() if grid is not None else None

                # fail-fast: if there are images, both must exist
                if pil_images:
                    assert pixel_values is not None, "OpenCUA preproc: images exist but pixel_values is None"
                    assert grid_thws is not None, "OpenCUA preproc: images exist but image_grid_thw is None"

            else:
                # Qwen3VL / UI-TARS 1.5 (Qwen2.5-VL): keep your original safe path
                proc_inputs = processor.apply_chat_template(
                    qwen_messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    truncation=False,
                    max_length=100000,
                )
                prompt_ids = proc_inputs["input_ids"][0].tolist()

                pixel_values = proc_inputs.get("pixel_values", None)
                if pixel_values is not None:
                    pixel_values = pixel_values.clone().cpu()

                grid = proc_inputs.get("image_grid_thw", None)
                grid_thws = grid.clone().long().cpu() if grid is not None else None

        else:
            # legacy path (other)
            prompt_ids = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
            )

            pil_images = collect_images_from_messages(prompt_messages)
            if pil_images:
                info = image_processor.preprocess(images=pil_images)

                pixel = info.get("pixel_values", None) or info.get("pixel_values_videos", None)

                grid = info.get("image_grid_thw", None)
                if grid is None:
                    grid = info.get("images_grid_thw", None)
                if grid is None:
                    grid = info.get("grid_thws", None)

                if pixel is not None:
                    pixel_values = torch.as_tensor(pixel).cpu()
                if grid is not None:
                    grid_thws = torch.as_tensor(grid, dtype=torch.long).cpu()

        # ----- response ids -----
        resp_ids = tokenizer(response_text, add_special_tokens=False)["input_ids"]

        has_image = (pixel_values is not None) or (grid_thws is not None)

        if max_prompt_len > 0 and len(prompt_ids) > max_prompt_len:
            if has_image:
                skipped += 1
                continue
            else:
                prompt_ids = prompt_ids[-max_prompt_len:]

        if max_gen_length > 0 and len(resp_ids) > max_gen_length:
            resp_ids = resp_ids[:max_gen_length]

        input_ids = prompt_ids + resp_ids
        if len(input_ids) == 0:
            skipped += 1
            continue

        start_pos = len(prompt_ids)

        # pixel fp16 optional
        if cast_pixel_fp16 and pixel_values is not None and pixel_values.dtype == torch.float32:
            pixel_values = pixel_values.half().contiguous()
        elif pixel_values is not None:
            pixel_values = pixel_values.contiguous()

        if grid_thws is not None:
            grid_thws = grid_thws.contiguous()

        seq_tensors.append(torch.tensor(input_ids, dtype=torch.long))
        start_pos_list.append(int(start_pos))
        adv_list.append(float(reward))
        pixel_values_list.append(pixel_values)
        grid_thws_list.append(grid_thws)

    N = len(seq_tensors)
    flat_input_ids, offsets = _pack_1d_long_tensors(seq_tensors)

    pack = {
        "flat_input_ids": flat_input_ids,
        "offsets": offsets,
        "start_pos": torch.tensor(start_pos_list, dtype=torch.long),
        "advantage": torch.tensor(adv_list, dtype=torch.float32),
        "pixel_values_list": pixel_values_list,
        "grid_thws_list": grid_thws_list,
        "meta": {
            "project": project_name,
            "optimization_data": optimization_data,
            "pretrained_model": str(pretrained_model),
            "model_type": str(model_type),
            "is_qwen3_vl": bool(is_qwen3_vl),
            "vl_family": str(vl_family),
            "use_processor_align": bool(use_processor_align),
            "pad_token_id": int(pad_id),
            "node_idx": int(node_idx),
            "num_nodes": int(num_nodes),
            "range_start": int(start),
            "range_end": int(end),
            "shard_size": int(N),
            "total_size": int(total_n),
            "skipped": int(skipped),
            "max_prompt_len_cap": int(max_prompt_len),
            "max_gen_length_cap": int(max_gen_length),
            "cast_pixel_fp16": int(cast_pixel_fp16),
        },
    }

    out_path = save_dir / f"{optimization_data}_preproc_node{node_idx}_of{num_nodes}.pt"
    torch.save(pack, out_path)
    print(
        f"[Node {node_idx}/{num_nodes}] saved: {out_path} "
        f"(N={N} skipped={skipped} flat={int(flat_input_ids.numel())})",
        flush=True,
    )


if __name__ == "__main__":
    main()
