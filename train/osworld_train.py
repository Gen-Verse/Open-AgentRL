

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import io
import json
import math
import time
import shutil
import base64
import logging
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from omegaconf import OmegaConf

from accelerate import Accelerator
from accelerate.utils import set_seed

from transformers import (
    AutoTokenizer,
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    AutoConfig,
)

from train.utils import get_config, flatten_omega_conf
from train.lr_schedulers import get_scheduler
from train.log_utils import set_verbosity_info, set_verbosity_error

try:
    import wandb
except ImportError:
    wandb = None

logger = logging.getLogger(__name__)


# ==================== Dataset ====================

class VLMTrainDataset(Dataset):

    def __init__(
        self,
        input_ids: torch.Tensor,
        p_mask: torch.Tensor,
        labels: torch.Tensor,
        advantage: torch.Tensor,
        pixel_values_list: List[Any],
        image_grid_thw_list: List[Any],
    ):
        assert input_ids.shape == p_mask.shape == labels.shape
        assert input_ids.size(0) == advantage.size(0) == len(pixel_values_list) == len(image_grid_thw_list)

        self.input_ids = input_ids
        self.p_mask = p_mask
        self.labels = labels
        self.advantage = advantage

        self.pixel_values_list = pixel_values_list
        self.image_grid_thw_list = image_grid_thw_list

        N, L = input_ids.shape
        self.logp_old_tok = torch.full((N, L), float("-inf"), dtype=torch.float32)

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return (
            int(idx),
            self.input_ids[idx],
            self.p_mask[idx],
            self.labels[idx],
            float(self.advantage[idx].item()),
            self.pixel_values_list[idx],
            self.image_grid_thw_list[idx],
        )


def simple_collate(batch):
    (
        ids,
        input_ids,
        p_mask,
        labels,
        adv,
        pixel_values,
        image_grid_thw,
    ) = zip(*batch)

    return {
        "ids": torch.tensor(ids, dtype=torch.long),
        "input_ids": torch.stack(input_ids),      # (B, L)
        "p_mask": torch.stack(p_mask),            # (B, L)
        "labels": torch.stack(labels),            # (B, L)
        "advantage": torch.tensor(adv, dtype=torch.float32),   # (B,)
        "pixel_values": list(pixel_values),       # len=B
        "image_grid_thw": list(image_grid_thw),   # len=B
    }


# ==================== Utils for images / prompts ====================

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



# ==================== Core train ====================

def make_attention_mask(input_ids: torch.Tensor, pad_id: int):
    return (input_ids != pad_id).to(torch.long)

def make_position_ids(attention_mask: torch.Tensor):
    position_ids = attention_mask.cumsum(dim=1) - 1
    position_ids.masked_fill_(attention_mask == 0, 0)
    return position_ids


@torch.no_grad()
def compute_logp_old_tok_parallel(
    accelerator: Accelerator,
    model,
    dataset: VLMTrainDataset,
    train_dataloader: DataLoader,
    pad_id: int,
    use_processor_align: bool,
    vl_family
):

    model.eval()
    from tqdm.auto import tqdm

    iterator = tqdm(
        train_dataloader,
        desc="Precomputing old token log-probs",
        dynamic_ncols=True,
        disable=not accelerator.is_local_main_process,
        leave=True,
    )

    for batch in iterator:
        ids = batch["ids"]
        input_ids_batch = batch["input_ids"].to(accelerator.device)
        p_mask_batch = batch["p_mask"].to(accelerator.device)
        pixel_values_list = batch["pixel_values"]
        image_grid_thw_list = batch["image_grid_thw"]

        B, L = input_ids_batch.shape
        full_batch = torch.full((B, L), float("-inf"), device=accelerator.device, dtype=torch.float32)

        for i in range(B):
            seq = input_ids_batch[i : i + 1]
            attention_mask = make_attention_mask(seq, pad_id)
            kwargs = dict(input_ids=seq, attention_mask=attention_mask)

            # These three families follow their own prepare_inputs_for_generation; don't pass position_ids manually
            if vl_family == "other":
                kwargs["position_ids"] = make_position_ids(attention_mask)

            # Cast pixel_values to model dtype (saves memory & avoids implicit casts)
            pv = pixel_values_list[i]
            if pv is not None:
                try:
                    param_dtype = next(model.parameters()).dtype
                except Exception:
                    param_dtype = pv.dtype
                kwargs["pixel_values"] = pv.to(accelerator.device, dtype=param_dtype)

            g = image_grid_thw_list[i]
            if g is not None:
                kwargs["image_grid_thw"] = g.to(accelerator.device)   # qwen3vl / qwen25vl / opencua 

            out = model(**kwargs)
            logits = get_logits_strict(out)

            logits_shifted = logits[:, :-1, :]
            labels_shifted = seq[:, 1:]
            log_probs = F.log_softmax(logits_shifted, dim=-1)
            logp_tok = log_probs.gather(dim=-1, index=labels_shifted.unsqueeze(-1)).squeeze(-1)

            full = torch.full((1, L), float("-inf"), device=logp_tok.device)
            full[:, 1:] = logp_tok
            full_batch[i] = full[0].float()

        dataset.logp_old_tok[ids] = full_batch.detach().cpu()

    accelerator.wait_for_everyone()
    model.train()

def get_logits_strict(out):
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits
    raise RuntimeError(f"[STRICT] model forward returned no logits. out={type(out)}")

def get_logits_compat(model, out):
    if hasattr(out, "logits") and out.logits is not None:
        return out.logits

    hs = None
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        hs = out.last_hidden_state
    elif isinstance(out, (tuple, list)) and len(out) > 0 and torch.is_tensor(out[0]):
        hs = out[0]

    if hs is None:
        raise RuntimeError(f"[COMPAT] No logits/hidden states in output: {type(out)}")

    if hasattr(model, "lm_head"):
        return model.lm_head(hs)
    if hasattr(model, "language_model") and hasattr(model.language_model, "lm_head"):
        return model.language_model.lm_head(hs)

    raise RuntimeError(f"[COMPAT] Cannot find lm_head to compute logits. model={type(model)}")


def forward_process_vlm(
    accelerator: Accelerator,
    model,
    batch: Dict[str, Any],
    dataset: VLMTrainDataset,
    tokenizer,
    config,
    use_processor_align: bool,
    vl_family
):
    device = accelerator.device

    ids = batch["ids"].cpu()
    input_ids_batch = batch["input_ids"].to(device)
    p_mask_batch = batch["p_mask"].to(device)
    adv_batch = batch["advantage"].to(device)
    pixel_values_list = batch["pixel_values"]
    image_grid_thw_list = batch["image_grid_thw"]

    old_lp_batch = dataset.logp_old_tok[ids].to(device)

    pad_id = tokenizer.pad_token_id
    B, L = input_ids_batch.shape
    total_loss = torch.tensor(0.0, device=device)

    for i in range(B):
        input_ids = input_ids_batch[i : i + 1]
        p_mask = p_mask_batch[i : i + 1]
        adv = adv_batch[i : i + 1]
        old_lp = old_lp_batch[i : i + 1]

        attention_mask = make_attention_mask(input_ids, pad_id)
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)

        if vl_family == "other":
            kwargs["position_ids"] = make_position_ids(attention_mask)

        pv = pixel_values_list[i]
        if pv is not None:
            try:
                param_dtype = next(model.parameters()).dtype
            except Exception:
                param_dtype = pv.dtype
            kwargs["pixel_values"] = pv.to(device, dtype=param_dtype)

        g = image_grid_thw_list[i]
        if g is not None:
            kwargs["image_grid_thw"] = g.to(device)

        out = model(**kwargs)
        logits = get_logits_strict(out)

        logits_shifted = logits[:, :-1, :]
        labels_shifted = input_ids[:, 1:]

        log_probs = F.log_softmax(logits_shifted, dim=-1)
        logp_new_tok = log_probs.gather(dim=-1, index=labels_shifted.unsqueeze(-1)).squeeze(-1)

        p_mask_shift = p_mask[:, 1:]
        old_lp_shift = old_lp[:, 1:]

        diff = (logp_new_tok - old_lp_shift)
        diff = torch.where(p_mask_shift, diff, torch.zeros_like(diff))
        diff = diff.clamp(-10.0, 10.0)
        ratio = torch.exp(diff)

        clipped = torch.clamp(ratio, 1.0 - config.training.eps, 1.0 + config.training.eps)

        adv_tok = adv.unsqueeze(1)
        surrogate = torch.min(ratio * adv_tok, clipped * adv_tok) * p_mask_shift

        denom = torch.clamp(p_mask_shift.sum(dim=1), min=1.0)
        policy_loss = - (surrogate.sum(dim=1) / denom).mean()

        kl_loss = torch.tensor(0.0, device=device)
        if config.training.beta > 0:
            kl_seq = (logp_new_tok - old_lp_shift)
            kl_seq = torch.where(p_mask_shift, kl_seq, torch.zeros_like(kl_seq))
            if getattr(config.training, "use_kl_estimator_k3", False):
                t = (-kl_seq).clamp(-10.0, 10.0)
                kl_seq = t.exp() - 1.0 + kl_seq
            kl_seq = (kl_seq * p_mask_shift).sum(dim=1) / torch.clamp(denom, min=1.0)
            kl_loss = config.training.beta * kl_seq.mean()

        total_loss = total_loss + (policy_loss + kl_loss)

    return total_loss / max(B, 1)








def save_checkpoint(model, tokenizer, processor, config, accelerator, name: str):
    import json, time, shutil
    from pathlib import Path

    project_name = config.experiment.project
    rl_base_dir = Path(config.system.rl_base_dir)
    save_base = rl_base_dir / project_name / "ckpt"
    save_dir = save_base / name

    model_to_save = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)

    def _is_opencua(tok, proc) -> bool:
        try:
            if tok is not None and "TikTokenV3" in tok.__class__.__name__:
                return True
        except Exception:
            pass
        try:
            if proc is not None:
                m = str(getattr(proc.__class__, "__module__", "")).lower()
                n = str(getattr(proc.__class__, "__name__", "")).lower()
                if "opencua" in m or "opencua" in n:
                    return True
        except Exception:
            pass
        try:
            s = str(getattr(tok, "name_or_path", "")).lower()
            if "opencua" in s:
                return True
        except Exception:
            pass
        return False

    def _find_base_dir(tok, proc) -> Path | None:
        # IMPORTANT: prefer config.model.* (base model) first, then fall back to name_or_path.
        cands = []
        try:
            target = getattr(config.training, "target", None)
            if target == "policy":
                cands.append(str(config.model.policy_model))
            elif target == "reward":
                cands.append(str(config.model.reward_model))
        except Exception:
            pass

        for obj in (proc, tok):
            if obj is None:
                continue
            v = getattr(obj, "name_or_path", None)
            if isinstance(v, str) and v:
                cands.append(v)
            init_kwargs = getattr(obj, "init_kwargs", {}) or {}
            v2 = init_kwargs.get("_name_or_path", None)
            if isinstance(v2, str) and v2:
                cands.append(v2)

        for s in cands:
            try:
                p = Path(s)
                if p.exists() and p.is_dir():
                    return p
            except Exception:
                continue
        return None

    def _load_json(p: Path):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            try:
                return json.loads(p.read_text())
            except Exception:
                return None

    def _dump_json(p: Path, obj):
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

    def _safe_copy2(src: Path, dst: Path):
        if not src.exists():
            return
        try:
            if src.resolve() == dst.resolve():
                return  # same file -> skip
        except Exception:
            pass
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except shutil.SameFileError:
            return

    if accelerator.is_main_process:
        save_dir.mkdir(parents=True, exist_ok=True)

        # 1) save model weights/config
        model_to_save.save_pretrained(
            save_dir,
            save_function=accelerator.save,
            state_dict=state_dict,
            safe_serialization=True,
        )

        is_opencua = _is_opencua(tokenizer, processor)

        # 2) save tokenizer/processor
        if not is_opencua:
            # Qwen / UI-TARS: keep your original behavior (unchanged)
            if processor is not None:
                processor.save_pretrained(str(save_dir))
            else:
                tokenizer.save_pretrained(str(save_dir))
        else:
            # OpenCUA: DO NOT call processor.save_pretrained (it writes a "mutated" tokenizer_config.json)
            if processor is not None and hasattr(processor, "tokenizer"):
                processor.tokenizer.save_pretrained(str(save_dir))
            else:
                tokenizer.save_pretrained(str(save_dir))

            # 3) OpenCUA: copy required remote-code/assets from base snapshot (OVERWRITE OK)
            base_dir = _find_base_dir(tokenizer, processor)
            if base_dir is None:
                raise RuntimeError("[OpenCUA] base_dir not found; cannot copy required opencua files.")

            # if base_dir accidentally equals save_dir (resume training), skip copying
            try:
                if base_dir.resolve() == save_dir.resolve():
                    base_dir = None
            except Exception:
                pass

            must_copy = [
                "tokenization_opencua.py",
                "processing_opencua.py",
                "configuration_opencua.py",
                "modeling_opencua.py",
                "tiktoken.model",
                "processor_config.json",
                "preprocessor_config.json",
                "image_processor_config.json",
                "generation_config.json",
            ]
            if base_dir is not None:
                for fn in must_copy:
                    _safe_copy2(base_dir / fn, save_dir / fn)

            # 4) OpenCUA: fix tokenizer_config.json using base snapshot as source-of-truth
            tp = save_dir / "tokenizer_config.json"
            tj = _load_json(tp) or {}

            bj = {}
            if base_dir is not None:
                bp = base_dir / "tokenizer_config.json"
                bj = _load_json(bp) or {}

            # (a) remove wrong key that causes loading path to go bad
            tj.pop("processor_class", None)

            # (b) restore chat_template from base if missing
            if "chat_template" not in tj and "chat_template" in bj:
                tj["chat_template"] = bj["chat_template"]

            # (c) ensure auto_map exists and set AutoTokenizer exactly as base (preferred)
            tj.setdefault("auto_map", {})
            if "auto_map" in bj and isinstance(bj["auto_map"], dict) and "AutoTokenizer" in bj["auto_map"]:
                tj["auto_map"]["AutoTokenizer"] = bj["auto_map"]["AutoTokenizer"]
            else:
                # fallback: keep a valid mapping
                tj["auto_map"]["AutoTokenizer"] = ["tokenization_opencua.TikTokenV3", None]

            # (d) if tokenizer_class exists and is a plain class name, drop it to avoid overriding auto_map
            if isinstance(tj.get("tokenizer_class"), str) and "." not in tj["tokenizer_class"]:
                tj.pop("tokenizer_class", None)

            _dump_json(tp, tj)

        # 5) metadata
        metadata = {"save_time": time.strftime("%Y-%m-%d %H:%M:%S")}
        with (save_base / "metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved model + tokenizer to {save_dir}")



















def load_merged_preproc_dataset(project_name: str, optimization_data: str):
    path = Path(project_name) / "temp_data" / f"{optimization_data}_preproc_merged.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing merged dataset: {path}. Run merge script first.")
    pk = torch.load(path, map_location="cpu")

    input_ids = pk["input_ids"]
    p_mask = pk["p_mask"]
    labels = pk["labels"]
    advantage = pk["advantage"]
    pixel_values_list = pk["pixel_values_list"]

    # backward compat:
    if "image_grid_thw_list" in pk:
        image_grid_thw_list = pk["image_grid_thw_list"]
    else:
        image_grid_thw_list = pk.get("grid_thws_list", [None] * int(input_ids.size(0)))

    meta = pk.get("meta", {})

    assert input_ids.shape == p_mask.shape == labels.shape
    assert input_ids.size(0) == advantage.size(0) == len(pixel_values_list) == len(image_grid_thw_list)
    return input_ids, p_mask, labels, advantage, pixel_values_list, image_grid_thw_list, meta


# ==================== Main ====================

def main():
    config = get_config()
    project_name = config.experiment.project

    if config.training.target == "policy":
        if config.experiment.current_epoch == 1:
            pretrained_model = config.model.policy_model
        else:
            pretrained_model = config.system.rl_base_dir + "/" + project_name + "/ckpt/" + config.model.optimized_name
        optimized_name = config.model.optimized_name
        optimization_data = "policy_optimization_data"
        update_per_step = config.training.policy.update_per_step
        batch_size_lm = config.training.policy.batch_size_lm
        gradient_checkpointing_enable = config.training.policy.gradient_checkpointing_enable
    elif config.training.target == "reward":
        if config.experiment.current_epoch == 1:
            pretrained_model = config.model.reward_model
        else:
            pretrained_model = config.system.rl_base_dir + "/" + project_name + "/ckpt/" + config.model.optimized_reward_name
        optimized_name = config.model.optimized_reward_name
        optimization_data = "reward_optimization_data"
        update_per_step = config.training.reward.update_per_step
        batch_size_lm = config.training.reward.batch_size_lm
        gradient_checkpointing_enable = config.training.reward.gradient_checkpointing_enable
    else:
        raise ValueError(f"Unknown training.target = {config.training.target}")

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # ===== load merged dataset =====
    input_ids_tensor, p_mask_tensor, labels_tensor, adv_tensor, pixel_values_list, image_grid_thw_list, meta = \
        load_merged_preproc_dataset(project_name, optimization_data)

    dataset_lm = VLMTrainDataset(
        input_ids=input_ids_tensor,
        p_mask=p_mask_tensor,
        labels=labels_tensor,
        advantage=adv_tensor,
        pixel_values_list=pixel_values_list,
        image_grid_thw_list=image_grid_thw_list,
    )
    total_n = int(meta.get("merged_kept_N", len(adv_tensor)))

    ws = int(os.environ.get("WORLD_SIZE", "1"))
    gradient_accumulation_steps = max(1, math.ceil(total_n / (update_per_step * batch_size_lm * ws)))

    # ===== Detect model types =====
    user_model_type = str(getattr(config, "model_type", "") or "").lower()
    try:
        hf_cfg = AutoConfig.from_pretrained(pretrained_model, trust_remote_code=True)
        hf_model_type = str(getattr(hf_cfg, "model_type", "") or "").lower()
        hf_archs = set(getattr(hf_cfg, "architectures", []) or [])
    except Exception as e:
        hf_cfg = None
        hf_model_type = ""
        hf_archs = set()
        logger.warning(f"[HFConfig] AutoConfig load failed: {e}")

    is_hf_qwen3 = (hf_model_type == "qwen3_vl") or any("Qwen3VL" in a for a in hf_archs)
    is_hf_qwen25 = (hf_model_type == "qwen2_5_vl") or any("Qwen2_5" in a for a in hf_archs)
    is_hf_opencua = (hf_model_type == "opencua") or any("OpenCUA" in a for a in hf_archs)

    # NEW: OpenCUA detection
    is_opencua = ("opencua" in user_model_type) or is_hf_opencua or any("OpenCUA" in a for a in hf_archs)

    # ---- Normalize to one family flag ----
    if is_opencua:
        vl_family = "opencua"
    elif is_hf_qwen3 or (user_model_type == "qwen3vl"):
        vl_family = "qwen3vl"
    elif is_hf_qwen25 or (user_model_type == "uitars15"):
        vl_family = "qwen25vl"   # UI-TARS-1.5 belongs here
    else:
        vl_family = "other"

    # These three families all should use processor-align path
    use_processor_align = (vl_family in ("qwen3vl", "qwen25vl", "opencua"))

    logger.info(f"[ModelDetect] vl_family={vl_family} | use_processor_align={use_processor_align}")

    # ===== Tokenizer / Processor =====
    if use_processor_align:
        if vl_family == "qwen25vl":
            processor = AutoProcessor.from_pretrained(pretrained_model, trust_remote_code=True, use_fast=False)
        else:
            processor = AutoProcessor.from_pretrained(pretrained_model, trust_remote_code=True)
        tokenizer = processor.tokenizer
        image_processor = getattr(processor, "image_processor", None)
    else:
        processor = None
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
        image_processor = AutoImageProcessor.from_pretrained(pretrained_model, trust_remote_code=True)

    # ===== Accelerator =====
    config.experiment.logging_dir = str(Path(project_name) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=None,
        project_dir=config.experiment.logging_dir,
        split_batches=True,
    )

    from accelerate.logging import get_logger
    accel_logger = get_logger(__name__, log_level="INFO")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    accel_logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process and wandb is not None:
        resume_wandb_run = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb_run = False
            run_id = wandb.util.generate_id()
            config.wandb.run_id = run_id

        wandb_init_kwargs = dict(
            name=project_name,
            id=run_id,
            resume=resume_wandb_run,
            entity=config.wandb.get("entity", None),
            config_exclude_keys=[],
        )
        wandb_config = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wandb_config.pop("experiment.resume_from_checkpoint", None)

        accelerator.init_trackers(
            project_name,
            config=wandb_config,
            init_kwargs={"wandb": wandb_init_kwargs},
        )

    if accelerator.is_main_process:
        os.makedirs(project_name, exist_ok=True)
        config_path = Path(project_name) / "config.yaml"
        logging.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)

    if config.training.seed is not None:
        set_seed(config.training.seed)

    # ===== Model =====
    logger.info("Loading VLM model")

    def load_vlm_model(pretrained_model: str, vl_family: str):
        # Qwen3-VL: MUST use ForConditionalGeneration (your “first code” standard)
        if vl_family == "qwen3vl":
            from transformers import Qwen3VLForConditionalGeneration
            return Qwen3VLForConditionalGeneration.from_pretrained(
                pretrained_model, trust_remote_code=True, torch_dtype="auto"
            )

        # UI-TARS 1.5 (Qwen2.5-VL): prefer ImageTextToText head (deployment standard)
        if vl_family == "qwen25vl":
            try:
                from transformers import AutoModelForImageTextToText
                return AutoModelForImageTextToText.from_pretrained(
                    pretrained_model, trust_remote_code=True, torch_dtype="auto"
                )
            except Exception:
                # fallback: still prefer a generation model that exposes logits
                try:
                    from transformers import Qwen2_5_VLForConditionalGeneration
                    return Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        pretrained_model, trust_remote_code=True, torch_dtype="auto"
                    )
                except Exception:
                    from transformers import AutoModelForCausalLM
                    return AutoModelForCausalLM.from_pretrained(
                        pretrained_model, trust_remote_code=True, torch_dtype="auto"
                    )

        # OpenCUA: try generation-capable loader first; fallback to AutoModel (some repos only register there)
        if vl_family == "opencua":
            try:
                from transformers import AutoModelForCausalLM
                return AutoModelForCausalLM.from_pretrained(
                    pretrained_model, trust_remote_code=True, torch_dtype="auto"
                )
            except Exception:
                return AutoModel.from_pretrained(
                    pretrained_model, trust_remote_code=True, torch_dtype="auto"
                )

        # other
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            pretrained_model, trust_remote_code=True, torch_dtype="auto"
        )

    model = load_vlm_model(pretrained_model, vl_family)

    def enable_gc(m):
        if hasattr(m, "config") and hasattr(m.config, "use_cache"):
            m.config.use_cache = False
        if hasattr(m, "language_model") and hasattr(m.language_model, "config"):
            if hasattr(m.language_model.config, "use_cache"):
                m.language_model.config.use_cache = False

        if hasattr(m, "language_model") and hasattr(m.language_model, "gradient_checkpointing_enable"):
            m.language_model.gradient_checkpointing_enable()
            for mm in m.language_model.modules():
                if hasattr(mm, "gradient_checkpointing"):
                    mm.gradient_checkpointing = True
            logger.info("[GC] Enabled on model.language_model")
            return

        if hasattr(m, "gradient_checkpointing_enable"):
            try:
                m.gradient_checkpointing_enable()
                for mm in m.modules():
                    if hasattr(mm, "gradient_checkpointing"):
                        mm.gradient_checkpointing = True
                logger.info("[GC] Enabled on model directly")
                return
            except ValueError as e:
                logger.warning(f"[GC] Direct enable failed: {e}")

        any_flag = False
        for mm in m.modules():
            if hasattr(mm, "gradient_checkpointing"):
                mm.gradient_checkpointing = True
                any_flag = True
        if any_flag:
            logger.info("[GC] Enabled by setting gradient_checkpointing=True on submodules")
        else:
            logger.warning("[GC] No supported gradient checkpointing hooks found.")

    if gradient_checkpointing_enable:
        enable_gc(model)

    pad_id = tokenizer.pad_token_id

    optimizer_config = config.optimizer.params
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": optimizer_config.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if config.optimizer.name == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=optimizer_config.learning_rate,
            betas=(optimizer_config.beta1, optimizer_config.beta2),
            weight_decay=optimizer_config.weight_decay,
            eps=optimizer_config.epsilon,
        )
    else:
        raise ValueError(f"Optimizer {config.optimizer.name} not supported")

    total_batch_size_lm = batch_size_lm * ws * gradient_accumulation_steps
    num_update_steps_per_epoch = max(1, math.ceil(total_n / total_batch_size_lm))
    num_train_epochs = config.training.num_train_epochs
    max_train_steps = num_update_steps_per_epoch * num_train_epochs

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=max_train_steps,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps,
        min_lr_scale=config.lr_scheduler.params.min_lr_scale,
    )

    train_dataloader_lm = DataLoader(
        dataset_lm,
        batch_size=batch_size_lm,
        sampler=None,
        collate_fn=simple_collate,
        num_workers=0,
    )

    logger.info("Preparing model, optimizer, scheduler and dataloader")
    model, optimizer, lr_scheduler, train_dataloader_lm = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader_lm
    )

    # ===== Precompute old logp =====
    logger.info("***** Running inference (precompute old logp) *****")
    compute_logp_old_tok_parallel(
        accelerator,
        model,
        dataset_lm,
        train_dataloader_lm,
        pad_id=pad_id,
        use_processor_align=use_processor_align,
        vl_family=vl_family
    )

    # ===== Training =====
    logger.info("***** Running training *****")
    logger.info(f"  Num training data = {len(dataset_lm)}")
    logger.info(f"  Num training steps = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {batch_size_lm}")
    logger.info(f"  Total train batch size = {total_batch_size_lm}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")

    from tqdm.auto import tqdm

    for epoch in range(num_train_epochs):
        model.train()
        progress = tqdm(
            train_dataloader_lm,
            disable=not accelerator.is_local_main_process,
            dynamic_ncols=True,
        )

        for step, batch in enumerate(progress, start=1):
            loss = forward_process_vlm(
                accelerator=accelerator,
                model=model,
                batch=batch,
                dataset=dataset_lm,
                tokenizer=tokenizer,
                config=config,
                use_processor_align=use_processor_align,
                vl_family=vl_family
            )

            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)

            if (step % gradient_accumulation_steps) == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                loss_val = float(accelerator.gather(loss.detach()).mean().item())
                progress.set_postfix(loss=loss_val)

    torch.cuda.empty_cache()
    accelerator.wait_for_everyone()

    save_checkpoint(model, tokenizer, processor, config, accelerator, optimized_name)
    if config.experiment.current_epoch % config.experiment.save_every == 0:
        save_checkpoint(
            model, tokenizer, processor, config, accelerator,
            f"epoch-{config.experiment.current_epoch}-{config.training.target}"
        )

    accelerator.end_training()


if __name__ == "__main__":
    main()
