# serve_ui_tars_1_5_7b.py
import os, io, time, base64, asyncio, argparse, re, copy
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from collections import Counter

import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from transformers import AutoProcessor

# Prefer Image-Text-to-Text for UI-TARS-1.5-7B (HF tag: Image-Text-to-Text)
try:
    from transformers import AutoModelForImageTextToText
except Exception:
    AutoModelForImageTextToText = None

try:
    from qwen_vl_utils import process_vision_info
except Exception:
    process_vision_info = None

try:
    import requests
except Exception:
    requests = None

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTHONUNBUFFERED", "1")


# -------------------- args (aligned with your serve_qwen3vl.py) --------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model path, e.g. ByteDance-Seed/UI-TARS-1.5-7B")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--queue-ms", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--idle-unload-s", type=int, default=0)
    p.add_argument("--offload-mode", type=str, default="none", choices=["cpu", "disk", "none"])
    p.add_argument("--preload", action="store_true")

    # tokenizer fast toggle (UI-TARS space examples often set use_fast=False to avoid breakage)
    p.add_argument("--use-fast", action="store_true", help="Use fast tokenizer if available (default: False)")

    # multi-GPU sharding options (same names as your scripts)
    p.add_argument("--num-gpu-per-model", type=int, default=1, help="Use N visible GPUs per model via device_map")
    p.add_argument("--device-map", type=str, default="auto", choices=["auto", "balanced", "sequential"],
                   help="Transformers device_map when num-gpu-per-model>1")
    p.add_argument("--max-gpu-mem", type=str, default="", help='Per-GPU cap like "70GiB"; default≈90% of total')

    # kept for compatibility; ignored here
    p.add_argument("--max-model-len", type=int, default=0, help="(ignored in transformers server)")
    p.add_argument("--enforce-eager", action="store_true", help="(ignored in transformers server)")
    return p.parse_args()


def _map_dtype(s: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]


def _apply_chat_template_compat(processor, messages: List[Dict[str, Any]]) -> str:
    tok = getattr(processor, "tokenizer", None)
    if hasattr(processor, "apply_chat_template"):
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if tok is not None and hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # last resort: concatenate all text chunks
    texts = []
    for m in messages:
        for c in m.get("content", []):
            if isinstance(c, dict) and c.get("type") == "text":
                texts.append(c.get("text", ""))
            elif isinstance(c, str):
                texts.append(c)
    return "\n".join(texts)


def _batch_decode_compat(processor, token_id_batches, **kw):
    # Qwen2.5-VL processor also provides post_process_image_text_to_text, but we keep generic here.
    tok = getattr(processor, "tokenizer", None)
    if tok is not None and hasattr(tok, "batch_decode"):
        return tok.batch_decode(token_id_batches, **kw)
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(token_id_batches, **kw)
    raise AttributeError("No batch_decode available on processor or tokenizer.")


class UiTarsLocalHFModel:
    """
    Transformers-backed UI-TARS-1.5 server model with the same surface as your serve_qwen3vl.py:
    - ensure_loaded_on_gpu()
    - offload(mode)
    - generate_batch(reqs)->List[str]
    """
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        use_fast: bool = False,
        num_gpu_per_model: int = 1,
        device_map: str = "auto",
        max_gpu_mem: str = "",
    ):
        # We *prefer* qwen-vl-utils for consistent image/video handling (Qwen2.5-VL uses patch_size=14).
        # But we still allow running without it (processor will do its own resizing).
        self.model_path = model_path
        self.device = device
        self.req_dtype_str = dtype
        self.dtype = _map_dtype(dtype)

        self.use_fast = bool(use_fast)

        self.num_gpu_per_model = int(num_gpu_per_model)
        self.device_map = device_map if self.num_gpu_per_model > 1 else None
        self.max_gpu_mem = (max_gpu_mem or "").strip()
        self.sharded = self.num_gpu_per_model > 1

        # Many UI-TARS community examples use use_fast=False to avoid tokenizer breaking changes.
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True, use_fast=self.use_fast
        )

        self.model = None
        self.on_gpu = False
        self.loaded = False
        self.loading_lock = asyncio.Lock()

        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            # optional: higher matmul precision
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

    # ---------- GPU helpers (use logical indices 0..N-1) ----------
    def _visible_cuda_ids(self) -> List[int]:
        if not torch.cuda.is_available():
            return []
        count = torch.cuda.device_count()
        want = max(1, self.num_gpu_per_model)
        return list(range(min(count, want)))

    def _compute_max_memory(self) -> Dict[int, str]:
        ids = self._visible_cuda_ids()
        mm: Dict[int, str] = {}
        if self.max_gpu_mem:
            for logical_i in range(len(ids)):
                mm[logical_i] = self.max_gpu_mem
            return mm
        for logical_i in ids:
            total = torch.cuda.get_device_properties(logical_i).total_memory
            cap = int(total * 0.90)
            mm[logical_i] = f"{cap // (1024**3)}GiB"
        return mm

    async def ensure_loaded_on_gpu(self):
        async with self.loading_lock:
            want_dtype = self.dtype

            # BF16 fallback if any visible GPU lacks SM>=80
            if want_dtype is torch.bfloat16:
                try:
                    caps = [torch.cuda.get_device_capability(i) for i in self._visible_cuda_ids()]
                    if not caps or any(maj < 8 for maj, _ in caps):
                        print("[serve] BF16 not fully supported; falling back to FP16.", flush=True)
                        want_dtype = torch.float16
                except Exception:
                    want_dtype = torch.float16

            print(
                f"[serve] visible CUDA ids = {self._visible_cuda_ids()}, "
                f"sharded={self.sharded}, device_map={self.device_map}, "
                f"max_memory={self._compute_max_memory() if self.sharded else None}",
                flush=True,
            )

            if self.model is None:
                # Prefer AutoModelForImageTextToText for UI-TARS-1.5-7B
                if self.sharded and torch.cuda.is_available():
                    if AutoModelForImageTextToText is not None:
                        self.model = AutoModelForImageTextToText.from_pretrained(
                            self.model_path,
                            torch_dtype=want_dtype,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            device_map=self.device_map,
                            max_memory=self._compute_max_memory(),
                        )
                    else:
                        from transformers import AutoModelForCausalLM
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            torch_dtype=want_dtype,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            device_map=self.device_map,
                            max_memory=self._compute_max_memory(),
                        )
                    self.on_gpu = True
                else:
                    if AutoModelForImageTextToText is not None:
                        self.model = AutoModelForImageTextToText.from_pretrained(
                            self.model_path,
                            torch_dtype=want_dtype,
                            trust_remote_code=True,
                            low_cpu_mem_usage=False,
                        )
                    else:
                        from transformers import AutoModelForCausalLM
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            torch_dtype=want_dtype,
                            trust_remote_code=True,
                            low_cpu_mem_usage=False,
                        )

                    if torch.cuda.is_available():
                        self.model.to(self.device)
                        self.on_gpu = (next(self.model.parameters()).device.type == "cuda")
                    else:
                        self.on_gpu = False

                self.model.eval()
                self.loaded = True

                if hasattr(self.model, "hf_device_map"):
                    ctr = Counter(self.model.hf_device_map.values())
                    print(f"[serve] hf_device_map summary: {dict(ctr)}", flush=True)
                else:
                    print("[serve] no hf_device_map attribute -> model may not be sharded by HF device_map.", flush=True)

            dev_info = "sharded" if self.sharded else str(next(self.model.parameters()).device)
            dtype_info = str(next(self.model.parameters()).dtype)
            print(f"[serve] ensure_loaded_on_gpu -> device={dev_info}, dtype={dtype_info}", flush=True)

    async def offload(self, mode: str):
        async with self.loading_lock:
            if self.model is None:
                return
            if mode == "cpu":
                try:
                    self.model.to("cpu")
                    self.on_gpu = False
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            elif mode == "disk":
                del self.model
                self.model = None
                self.loaded = False
                self.on_gpu = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # -------- image decode --------
    def _decode_image(self, src: str) -> Image.Image:
        s = (src or "").strip()

        # http(s)
        if re.match(r"^https?://", s):
            if requests is None:
                raise RuntimeError("requests not available to fetch http(s) image.")
            resp = requests.get(s, timeout=20)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")

        # data url / base64（优先判断）
        if s.startswith("data:image") or ("," in s and "base64" in s.split(",", 1)[0]):
            b64 = s.split(",", 1)[1]
            raw = base64.b64decode(b64, validate=False)
            return Image.open(io.BytesIO(raw)).convert("RGB")

        # 只有“短字符串”才当路径试一下
        try:
            if len(s) < 512 and os.path.exists(s):
                return Image.open(s).convert("RGB")
        except OSError:
            pass

        # 最后兜底：当作纯 base64
        raw = base64.b64decode(s, validate=False)
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def _normalize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Accept OpenAI-style messages where content can be str or list[parts].
        Normalize to Qwen-style list[parts] with:
          {"type":"text","text":...}
          {"type":"image","image": PIL.Image}
        """
        msgs = copy.deepcopy(messages)
        for m in msgs:
            c = m.get("content")

            # plain string -> one text chunk
            if isinstance(c, str):
                m["content"] = [{"type": "text", "text": c}]
                continue

            if not isinstance(c, list):
                continue

            new_content = []
            for part in c:
                if isinstance(part, str):
                    new_content.append({"type": "text", "text": part})
                    continue
                if not isinstance(part, dict):
                    new_content.append(part)
                    continue

                t = part.get("type")

                if t == "text":
                    new_content.append({"type": "text", "text": part.get("text", "")})
                    continue

                # OpenAI-ish image_url -> image
                if t == "image_url":
                    url = (part.get("image_url") or {}).get("url")
                    if isinstance(url, str) and url.strip():
                        img = self._decode_image(url)
                        new_content.append({"type": "image", "image": img})
                    else:
                        new_content.append(part)
                    continue

                if t == "image":
                    src = part.get("image")
                    if isinstance(src, Image.Image):
                        new_content.append(part)
                    elif isinstance(src, str) and src.strip():
                        img = self._decode_image(src)
                        new_content.append({"type": "image", "image": img})
                    else:
                        new_content.append(part)
                    continue

                # keep other modalities as-is (video/audio)
                new_content.append(part)

            m["content"] = new_content
        return msgs

    def _call_process_vision_info(self, messages: List[Dict[str, Any]], image_patch_size: int):
        """
        Robust compat wrapper for qwen_vl_utils.process_vision_info:
        - try convo = messages and convo = [messages]
        - drop return_video_metadata only if explicitly unsupported
        - support returns of tuple len=2/3 or dict-like
        """
        if process_vision_info is None:
            return None, None, {}

        base_kwargs = dict(
            return_video_kwargs=True,
            image_patch_size=int(image_patch_size),
            return_video_metadata=True,   # may be unsupported
        )

        def _unwrap(img, vid, vkw):
            if isinstance(img, list) and img and isinstance(img[0], list):
                img = img[0]
            if isinstance(vid, list) and vid and isinstance(vid[0], list):
                vid = vid[0]
            if isinstance(vkw, list):
                vkw = vkw[0] if vkw else {}
            return img, vid, (vkw or {})

        def _parse_out(out):
            if isinstance(out, tuple):
                if len(out) == 3:
                    img, vid, vkw = out
                elif len(out) == 2:
                    img, vid = out
                    vkw = {}
                else:
                    raise TypeError(f"Unexpected tuple len from process_vision_info: {len(out)}")
                return _unwrap(img, vid, vkw)

            if isinstance(out, dict):
                img = out.get("image_inputs") or out.get("images") or out.get("img")
                vid = out.get("video_inputs") or out.get("videos") or out.get("vid")
                vkw = out.get("video_kwargs") or out.get("vkw") or {}
                return _unwrap(img, vid, vkw)

            raise TypeError(f"Unexpected return type from process_vision_info: {type(out)}")

        def _is_unexpected_kw(e: Exception, kw_name: str) -> bool:
            s = str(e)
            return (kw_name in s) and ("unexpected keyword" in s or "got an unexpected keyword argument" in s)

        errs = []
        for convo in (messages, [messages]):
            try:
                out = process_vision_info(convo, **base_kwargs)
                return _parse_out(out)
            except TypeError as e:
                errs.append(e)
                if "return_video_metadata" in base_kwargs and _is_unexpected_kw(e, "return_video_metadata"):
                    try:
                        kw2 = dict(base_kwargs)
                        kw2.pop("return_video_metadata", None)
                        out = process_vision_info(convo, **kw2)
                        return _parse_out(out)
                    except Exception as e2:
                        errs.append(e2)
            except Exception as e:
                errs.append(e)

        raise RuntimeError(f"process_vision_info failed; last error: {errs[-1]!r}") from errs[-1]

    def _prepare_inputs_from_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Preferred path (when qwen-vl-utils is available):
          msgs = normalize(messages)
          text = apply_chat_template(..., add_generation_prompt=True)
          image_inputs, video_inputs, video_kwargs = process_vision_info(..., image_patch_size=14 for Qwen2.5-VL)
          inputs = processor(text=[text], images=image_inputs, videos=video_inputs, video_metadata=..., **video_kwargs,
                             do_resize=False, padding=True, return_tensors="pt")

        Fallback path (no qwen-vl-utils):
          processor(text=[text], images=[PIL_images], padding=True, return_tensors="pt")  (processor will resize)
        """
        msgs = self._normalize_messages(messages)
        text = _apply_chat_template_compat(self.processor, msgs)

        # Qwen2.5-VL patch size is typically 14; prefer processor's value if available
        patch_size = getattr(getattr(self.processor, "image_processor", object()), "patch_size", 14)
        if isinstance(patch_size, (tuple, list)):
            patch_size = int(patch_size[0]) if patch_size else 14
        patch_size = int(patch_size) if patch_size else 14

        # If qwen-vl-utils exists, let it handle resizing & metadata.
        if process_vision_info is not None:
            image_inputs, video_inputs, video_kwargs = self._call_process_vision_info(msgs, patch_size)

            video_metadatas = None
            if video_inputs is not None and isinstance(video_inputs, list) and video_inputs:
                # Some versions may return [(video_tensor, video_metadata), ...]
                if isinstance(video_inputs[0], tuple) and len(video_inputs[0]) == 2:
                    vids, metas = zip(*video_inputs)
                    video_inputs = list(vids)
                    video_metadatas = list(metas)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                video_metadata=video_metadatas,
                **(video_kwargs or {}),
                do_resize=False,          # important: qwen-vl-utils already resized
                padding=True,
                return_tensors="pt",
            )
        else:
            # Fallback: collect PIL images from messages and rely on processor resize.
            pil_images: List[Image.Image] = []
            for m in msgs:
                for c in m.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "image" and isinstance(c.get("image"), Image.Image):
                        pil_images.append(c["image"])
            inputs = self.processor(
                text=[text],
                images=pil_images if pil_images else None,
                padding=True,
                return_tensors="pt",
            )

        # Some configs may include token_type_ids; drop it
        if isinstance(inputs, dict):
            inputs.pop("token_type_ids", None)

        # Ensure attention_mask exists
        if "attention_mask" not in inputs and "input_ids" in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        return inputs

    @staticmethod
    def _apply_stop_posthoc(text: str, stop: Optional[Any]) -> str:
        if not text:
            return text
        stops: List[str] = []
        if isinstance(stop, str) and stop:
            stops = [stop]
        elif isinstance(stop, list):
            stops = [str(x) for x in stop if str(x)]
        if not stops:
            return text
        cut = None
        for s in stops:
            j = text.find(s)
            if j != -1:
                cut = j if cut is None else min(cut, j)
        return text[:cut] if cut is not None else text

    def generate_batch(self, reqs: List[Dict[str, Any]]) -> List[str]:
        outs: List[str] = []
        sharded = self.sharded

        with torch.inference_mode():
            for r in reqs:
                inputs = self._prepare_inputs_from_messages(r["messages"])

                # Move inputs
                enc_on: Dict[str, Any] = {}
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor):
                        if sharded:
                            enc_on[k] = v  # keep on CPU; accelerate will move as needed
                        else:
                            dev = next(self.model.parameters()).device
                            dtype = next(self.model.parameters()).dtype
                            if v.dtype.is_floating_point:
                                enc_on[k] = v.to(dev, dtype=dtype)
                            else:
                                enc_on[k] = v.to(dev)
                    else:
                        enc_on[k] = v

                max_tokens = int(r.get("max_tokens", 512))
                temperature = float(r.get("temperature", 0.0))
                top_p = float(r.get("top_p", 0.9))

                do_sample = temperature > 1e-6
                gen_kwargs = dict(
                    max_new_tokens=max_tokens,
                    do_sample=do_sample,
                )
                if do_sample:
                    gen_kwargs.update(dict(temperature=temperature, top_p=top_p))

                generated_ids = self.model.generate(**enc_on, **gen_kwargs)

                # trim prompt
                in_ids = enc_on.get("input_ids", None)
                if in_ids is not None:
                    trimmed = [out_ids[len(in0):] for in0, out_ids in zip(in_ids, generated_ids)]
                else:
                    trimmed = generated_ids

                # Decode
                # If processor provides post_process_image_text_to_text, use it
                if hasattr(self.processor, "post_process_image_text_to_text"):
                    try:
                        text_list = self.processor.post_process_image_text_to_text(
                            trimmed,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        )
                        text = text_list[0] if text_list else ""
                    except Exception:
                        text = _batch_decode_compat(
                            self.processor, trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                        )[0]
                else:
                    text = _batch_decode_compat(
                        self.processor, trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                    )[0]

                # Optional: OpenAI-like stop (post-hoc trim)
                text = self._apply_stop_posthoc(text, r.get("stop", None))
                outs.append(text)

        return outs


@dataclass
class PendingItem:
    payload: Dict[str, Any]
    fut: asyncio.Future


class MicroBatcher:
    def __init__(self, model: UiTarsLocalHFModel, max_batch: int, queue_ms: int):
        self.model = model
        self.max_batch = max_batch
        self.queue_s = queue_ms / 1000.0
        self.q: asyncio.Queue[PendingItem] = asyncio.Queue()
        self.busy = False

    async def submit(self, payload: Dict[str, Any]) -> str:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        await self.q.put(PendingItem(payload, fut))
        return await fut

    async def loop(self):
        while True:
            item = await self.q.get()
            batch = [item]
            t0 = time.time()
            while len(batch) < self.max_batch and (time.time() - t0) < self.queue_s:
                try:
                    more = self.q.get_nowait()
                    batch.append(more)
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.001)

            await self.model.ensure_loaded_on_gpu()
            self.busy = True
            try:
                payloads = [x.payload for x in batch]
                outputs = await asyncio.to_thread(self.model.generate_batch, payloads)

                assert len(outputs) == len(batch)
                for ans, itm in zip(outputs, batch):
                    if not itm.fut.done():
                        itm.fut.set_result(ans)
            except Exception as e:
                for itm in batch:
                    if not itm.fut.done():
                        itm.fut.set_exception(e)
            finally:
                self.busy = False


def create_app(args):
    app = FastAPI()
    mdl = UiTarsLocalHFModel(
        model_path=args.model,
        device=args.device,
        dtype=args.dtype,
        use_fast=args.use_fast,
        num_gpu_per_model=args.num_gpu_per_model,
        device_map=args.device_map,
        max_gpu_mem=args.max_gpu_mem,
    )
    batcher = MicroBatcher(mdl, max_batch=args.max_batch, queue_ms=args.queue_ms)

    class State:
        last_request_ts = time.time()
        idle_unload_s = args.idle_unload_s
        offload_mode = args.offload_mode

    @app.get("/ready")
    async def ready():
        return {"loaded": mdl.loaded, "on_gpu": mdl.on_gpu, "sharded": mdl.sharded}

    @app.on_event("startup")
    async def _startup():
        asyncio.create_task(batcher.loop())
        if args.preload:
            await mdl.ensure_loaded_on_gpu()
            print("[serve] preload done", flush=True)

        async def idle_offloader():
            while True:
                await asyncio.sleep(1.0)
                if State.idle_unload_s <= 0 or State.offload_mode == "none":
                    continue
                idle_for = time.time() - State.last_request_ts
                if idle_for >= State.idle_unload_s and batcher.q.empty() and not batcher.busy:
                    await mdl.offload(State.offload_mode)

        asyncio.create_task(idle_offloader())

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: Request):
        body = await req.json()
        try:
            payload: Dict[str, Any] = {
                "messages": body["messages"],
                "max_tokens": body.get("max_tokens", 512),
                "temperature": body.get("temperature", 0.0),
                "top_p": body.get("top_p", 0.9),
            }
            if "stop" in body:
                payload["stop"] = body["stop"]

            State.last_request_ts = time.time()
            text = await batcher.submit(payload)
            return JSONResponse({
                "choices": [{
                    "message": {"content": text},
                    "finish_reason": "stop"
                }]
            })
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=500)

    return app


if __name__ == "__main__":
    args = get_args()
    uvicorn.run(create_app(args), host=args.host, port=args.port, log_level="info")
