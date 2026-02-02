# serve_opencua.py
import os, io, time, base64, asyncio, argparse, re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

import torch
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from transformers import AutoModel, AutoTokenizer, AutoImageProcessor
from collections import Counter

try:
    import requests
except Exception:
    requests = None


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF model path, e.g. OpenCUA/OpenCUA-7B")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max-batch", type=int, default=8)
    p.add_argument("--queue-ms", type=int, default=20)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--idle-unload-s", type=int, default=0)
    p.add_argument("--offload-mode", type=str, default="none", choices=["cpu", "disk", "none"])
    p.add_argument("--preload", action="store_true")

    # multi-GPU sharding options
    p.add_argument("--num-gpu-per-model", type=int, default=1, help="Use N visible GPUs per model via device_map")
    p.add_argument("--device-map", type=str, default="auto", choices=["auto", "balanced", "sequential"],
                   help="Transformers device_map when num-gpu-per-model>1")
    p.add_argument("--max-gpu-mem", type=str, default="", help='Per-GPU cap like "70GiB"; default≈90% of total')
    return p.parse_args()


def _map_dtype(s: str):
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]


def _first_not_none(d: dict, keys):
    for k in keys:
        v = d.get(k, None)
        if v is not None:
            return v
    return None

from transformers import AutoProcessor

class OpenCUALocalModel:
    def __init__(self, model_path: str, device="cuda", dtype="bfloat16",
                 num_gpu_per_model: int = 1, device_map: str = "auto", max_gpu_mem: str = ""):
        self.model_path = model_path
        self.device = device
        self.req_dtype_str = dtype
        self.dtype = _map_dtype(dtype)

        self.num_gpu_per_model = int(num_gpu_per_model)
        self.device_map = device_map if self.num_gpu_per_model > 1 else None
        self.max_gpu_mem = (max_gpu_mem or "").strip()
        self.sharded = self.num_gpu_per_model > 1

        # processor + tokenizer
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)

        must = any([
            hasattr(self.image_processor, "preprocess"),
            hasattr(self.image_processor, "__call__"),
        ])
        if not must:
            raise RuntimeError(
                "This model snapshot looks text-only (no image processor). "
                "Use a VL snapshot (with image_processor_config.json) or another model like Qwen2.5-VL."
            )

        self.model = None
        self.on_gpu = False
        self.loaded = False
        self.loading_lock = asyncio.Lock()

        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    # ---------- GPU helpers (use logical indices 0..N-1) ----------
    def _visible_cuda_ids(self) -> List[int]:
        """
        Return logical GPU indices in this process (0..count-1).
        With CUDA_VISIBLE_DEVICES=2,3, torch sees 2 GPUs as 0,1 here.
        """
        if not torch.cuda.is_available():
            return []
        count = torch.cuda.device_count()
        want = max(1, self.num_gpu_per_model)
        return list(range(min(count, want)))

    def _compute_max_memory(self) -> Dict[int, str]:
        """
        Build max_memory dict keyed by logical indices (0..N-1).
        """
        ids = self._visible_cuda_ids()
        mm: Dict[int, str] = {}
        if self.max_gpu_mem:
            for logical_i in range(len(ids)):
                mm[logical_i] = self.max_gpu_mem
            return mm
        # default 90% per visible device
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
                flush=True
            )

            if self.model is None:
                if self.sharded and torch.cuda.is_available():
                    # multi-GPU sharded loading (HF/Accelerate)
                    self.model = AutoModel.from_pretrained(
                        self.model_path,
                        torch_dtype=want_dtype,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        device_map=self.device_map,             # "auto"/"balanced"/"sequential"
                        max_memory=self._compute_max_memory(),  # keys: 0..N-1 (logical)
                    )
                    self.on_gpu = True
                else:
                    # single-GPU / CPU
                    self.model = AutoModel.from_pretrained(
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

                # device map summary (if present)
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
                    torch.cuda.empty_cache()
            elif mode == "disk":
                del self.model
                self.model = None
                self.loaded = False
                self.on_gpu = False
                torch.cuda.empty_cache()

    # -------- image decode --------
    def _decode_image(self, src: str) -> Image.Image:
        s = src.strip()
        if re.match(r"^https?://", s):
            if requests is None:
                raise RuntimeError("requests not available to fetch http(s) image.")
            resp = requests.get(s, timeout=10)
            resp.raise_for_status()
            return Image.open(io.BytesIO(resp.content)).convert("RGB")
        b64 = s.split(",", 1)[1] if "," in s else s
        raw = base64.b64decode(b64, validate=False)
        return Image.open(io.BytesIO(raw)).convert("RGB")


    def _prepare_inputs_from_messages(self, messages):
        pil_images = []
        for m in messages:
            content = m.get("content")
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "image" and part.get("image"):
                        pil_images.append(self._decode_image(part["image"]))
                    elif part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url")
                        if url:
                            pil_images.append(self._decode_image(url))

        # 关键：tokenize=False，先拿到带 <|media_placeholder|> 的纯文本
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 关键：由 processor 负责 images->grid，并把 placeholder 扩展成 num_image_tokens 次
        enc = self.processor(
            images=pil_images if pil_images else None,
            text=text,
            return_tensors="pt",
            padding=False,
        )

        out = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc.get("attention_mask", torch.ones_like(enc["input_ids"])),
        }
        if "pixel_values" in enc:
            out["pixel_values"] = enc["pixel_values"]
        if "image_grid_thw" in enc:
            out["image_grid_thw"] = enc["image_grid_thw"]   # 注意这个名字！
        return out


    def generate_batch(self, reqs: List[Dict[str, Any]]) -> List[str]:
        sharded = self.sharded
        param_dtype = next(self.model.parameters()).dtype
        outs: List[str] = []

        with torch.inference_mode():
            for r in reqs:
                prepared = self._prepare_inputs_from_messages(r["messages"])

                # 放设备/设 dtype
                enc_on: Dict[str, Any] = {}
                for k, v in prepared.items():
                    if isinstance(v, torch.Tensor):
                        if sharded:
                            # 分布式/多卡：留在 CPU，由 HF/Accelerate 分发
                            enc_on[k] = v
                        else:
                            dev = next(self.model.parameters()).device
                            enc_on[k] = (v.to(dev, dtype=param_dtype)
                                         if v.dtype.is_floating_point else v.to(dev))
                    else:
                        enc_on[k] = v

                # === 关键：确保 attention_mask 一起传 ===
                # enc_on 里现在至少包含 input_ids/attention_mask，若有图像则还包含 pixel_values/grid_thws
                gen = self.model.generate(
                    **enc_on,
                    max_new_tokens=int(r.get("max_tokens", 512)),
                    temperature=float(r.get("temperature", 0.0)),
                    top_p=float(r.get("top_p", 0.9)),
                )

                # 解码
                seq = gen.sequences if hasattr(gen, "sequences") else gen
                prompt_len = enc_on["input_ids"].shape[1]
                gen_ids = seq[:, prompt_len:] if seq.shape[1] > prompt_len else seq
                txt = self.tokenizer.batch_decode(
                    gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                outs.append(txt)
        return outs



@dataclass
class PendingItem:
    payload: Dict[str, Any]
    fut: asyncio.Future


class MicroBatcher:
    def __init__(self, model: OpenCUALocalModel, max_batch: int, queue_ms: int):
        self.model = model
        self.max_batch = max_batch
        self.queue_ms = queue_ms / 1000.0
        self.q: asyncio.Queue[PendingItem] = asyncio.Queue()
        self.busy = False

    async def submit(self, payload: Dict[str, Any]) -> str:
        fut = asyncio.get_event_loop().create_future()
        await self.q.put(PendingItem(payload, fut))
        return await fut

    async def loop(self):
        while True:
            item = await self.q.get()
            batch = [item]
            t0 = time.time()
            while len(batch) < self.max_batch and (time.time() - t0) < self.queue_ms:
                try:
                    more = self.q.get_nowait()
                    batch.append(more)
                except asyncio.QueueEmpty:
                    await asyncio.sleep(0.001)

            await self.model.ensure_loaded_on_gpu()
            self.busy = True
            try:
                outputs = self.model.generate_batch([x.payload for x in batch])
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
    mdl = OpenCUALocalModel(
        args.model, device=args.device, dtype=args.dtype,
        num_gpu_per_model=args.num_gpu_per_model,
        device_map=args.device_map, max_gpu_mem=args.max_gpu_mem
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
            payload = {
                "messages": body["messages"],
                "max_tokens": body.get("max_tokens", 512),
                "temperature": body.get("temperature", 0.0),
                "top_p": body.get("top_p", 0.9),
            }
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
