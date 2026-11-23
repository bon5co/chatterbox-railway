#!/usr/bin/env python3
"""Chatterbox text-to-speech API for Railway, CPU only.

resemble-ai/chatterbox (MIT) behind one authenticated, OpenAI-compatible HTTP API:
POST /v1/audio/speech takes text and returns speech as WAV, MP3, FLAC or raw PCM.

What this file adds around the model, because upstream ships a Python package and a Gradio demo, not a server:

  1. Bearer API_KEY on every route except GET /health. An empty API_KEY refuses to start.
  2. GET /health answers 503 until the default model is loaded and warmed, then 200, unauthenticated.
  3. Three models behind one `model` field: chatterbox-turbo (default), chatterbox-nano and
     chatterbox-multilingual (23 languages). Weights are baked into the image; nothing downloads at boot.
     Only MAX_LOADED_MODELS stay in memory; asking for another one loads it and drops the least recent.
  4. Long input is split at sentence boundaries and synthesised piece by piece, so a wall-clock budget can be
     enforced between pieces and a request answers 504 before Railway's roughly 5 minute edge timeout.
  5. POST /v1/audio/clone: one-shot voice cloning from an uploaded reference clip, never stored.
  6. One synthesis at a time (CPU bound), a short queue, 429 when it is full. Text and audio are held in
     memory for the request only and are never logged.
"""

import asyncio
import io
import os
import re
import socket
import sys
import tempfile
import threading
import time
from collections import OrderedDict

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

API_KEY = os.environ.get("API_KEY") or ""
THREADS = int(os.environ.get("TORCH_THREADS") or os.cpu_count() or 1)
MODEL_ROOT = os.environ.get("MODEL_ROOT") or "/models"
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL") or "chatterbox-turbo"
MAX_LOADED_MODELS = max(1, int(os.environ.get("MAX_LOADED_MODELS") or 1))
MAX_INPUT_CHARS = int(os.environ.get("MAX_INPUT_CHARS") or 2000)
MAX_CHUNK_CHARS = int(os.environ.get("MAX_CHUNK_CHARS") or 280)
MAX_SECONDS = float(os.environ.get("MAX_SECONDS") or 240)
MAX_PENDING = int(os.environ.get("MAX_PENDING") or 3)
MAX_REFERENCE_BYTES = int(os.environ.get("MAX_REFERENCE_BYTES") or 10 * 1024 * 1024)
PUBLIC_PATHS = {"/health"}
GAP_SECONDS = 0.12

# name -> (directory under MODEL_ROOT, description, languages)
MODELS = {
    "chatterbox-turbo": ("turbo", "350M, English, paralinguistic tags such as [laugh] and [cough]"),
    "chatterbox-nano": ("nano", "110M, English, the smallest and quickest, paralinguistic tags"),
    "chatterbox-multilingual": ("multilingual", "23 languages, exaggeration and cfg_weight controls"),
}
ALIASES = {"turbo": "chatterbox-turbo", "nano": "chatterbox-nano", "multilingual": "chatterbox-multilingual",
           "tts-1": DEFAULT_MODEL, "tts-1-hd": DEFAULT_MODEL, "chatterbox": DEFAULT_MODEL}
FORMATS = {"wav": ("audio/wav", "WAV", "PCM_16"), "mp3": ("audio/mpeg", "MP3", None),
           "flac": ("audio/flac", "FLAC", "PCM_16"), "pcm": ("audio/L16;rate=24000;channels=1", None, None)}

_state = {"ready": False, "error": None, "loaded_in": None}
_models: "OrderedDict[str, object]" = OrderedDict()
_run_lock = threading.Lock()      # one synthesis at a time
_load_lock = threading.Lock()
_pending = 0
_pending_lock = threading.Lock()
_languages: dict = {}

app = FastAPI(title="Chatterbox TTS API", docs_url=None, redoc_url=None, openapi_url=None)


class _Deadline(Exception):
    pass


def _load_model(name: str):
    """Load (or return the cached) model object; evicts the least recently used past MAX_LOADED_MODELS."""
    with _load_lock:
        if name in _models:
            _models.move_to_end(name)
            return _models[name]
        while len(_models) >= MAX_LOADED_MODELS:
            _models.popitem(last=False)
        import gc
        gc.collect()
        path = os.path.join(MODEL_ROOT, MODELS[name][0])
        t0 = time.time()
        if name == "chatterbox-multilingual":
            from chatterbox.mtl_tts import SUPPORTED_LANGUAGES, ChatterboxMultilingualTTS
            _languages.update(SUPPORTED_LANGUAGES)
            model = ChatterboxMultilingualTTS.from_local(path, "cpu")
        else:
            from chatterbox.tts_turbo import ChatterboxTurboTTS
            model = ChatterboxTurboTTS.from_local(path, "cpu", nano=(name == "chatterbox-nano"))
        _models[name] = model
        print(f"[chatterbox] loaded {name} in {time.time() - t0:.1f}s", flush=True)
        return model


def _boot() -> None:
    t0 = time.time()
    try:
        torch.set_num_threads(THREADS)
        model = _load_model(DEFAULT_MODEL)
        with torch.inference_mode():
            if DEFAULT_MODEL == "chatterbox-multilingual":
                model.generate("Hello.", language_id="en")
            else:
                model.generate("Hello.")
        _state["ready"] = True
        _state["loaded_in"] = round(time.time() - t0, 1)
        print(f"[chatterbox] ready model={DEFAULT_MODEL} threads={THREADS} in {_state['loaded_in']}s", flush=True)
    except Exception as exc:  # noqa: BLE001 - reported over /health, never fatal
        _state["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[chatterbox] LOAD FAILED: {_state['error']}", flush=True)


def _split(text: str) -> list[str]:
    """Sentence-aware pieces of at most MAX_CHUNK_CHARS characters."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？…])\s+|\n+", text) if s.strip()]
    pieces: list[str] = []
    for s in sentences:
        while len(s) > MAX_CHUNK_CHARS:  # no sentence end in reach: cut at the last comma or space
            cut = max(s.rfind(c, 0, MAX_CHUNK_CHARS) for c in (", ", "; ", " "))
            cut = cut if cut > MAX_CHUNK_CHARS // 3 else MAX_CHUNK_CHARS
            pieces.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        if not s:
            continue
        if pieces and len(pieces[-1]) + 1 + len(s) <= MAX_CHUNK_CHARS and not pieces[-1].endswith(("...", "…")):
            pieces[-1] = pieces[-1] + " " + s
        else:
            pieces.append(s)
    return pieces


def _synthesise(name: str, text: str, language: str | None, ref_path: str | None, temperature: float,
                exaggeration: float, cfg_weight: float, seed: int | None, deadline: float):
    model = _load_model(name)
    if seed is not None:
        torch.manual_seed(seed)
    saved_conds = model.conds
    try:
        waves = []
        for i, piece in enumerate(_split(text)):
            if time.time() > deadline:
                raise _Deadline()
            kwargs = {"temperature": temperature}
            if ref_path and i == 0:
                kwargs["audio_prompt_path"] = ref_path
            if name == "chatterbox-multilingual":
                kwargs.update(language_id=language or "en", exaggeration=exaggeration, cfg_weight=cfg_weight)
            with torch.inference_mode():
                wav = model.generate(piece, **kwargs)
            waves.append(wav.squeeze(0).cpu().numpy().astype(np.float32))
            if i == 0 and ref_path:
                ref_path = None  # conditionals are prepared once and kept on the model for the later pieces
        if not waves:
            raise HTTPException(422, "input has no speakable text")
        gap = np.zeros(int(model.sr * GAP_SECONDS), dtype=np.float32)
        out = waves[0]
        for w in waves[1:]:
            out = np.concatenate([out, gap, w])
        return out, model.sr, len(waves)
    finally:
        model.conds = saved_conds  # a cloned voice never leaks into the next request


def _encode(audio: np.ndarray, sr: int, fmt: str) -> bytes:
    import soundfile as sf
    if fmt == "pcm":
        return (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    _mime, sf_format, subtype = FORMATS[fmt]
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format=sf_format, subtype=subtype)
    return buf.getvalue()


def _status_kb(field: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return 0


def _resolve(params: dict) -> tuple[str, str | None, str]:
    raw = (params.get("model") or DEFAULT_MODEL).strip().lower()
    name = ALIASES.get(raw, raw)
    if name not in MODELS:
        raise HTTPException(422, f"unknown model '{raw}'; one of {sorted(MODELS)}")
    fmt = (params.get("response_format") or "wav").strip().lower()
    if fmt not in FORMATS:
        raise HTTPException(422, f"unknown response_format '{fmt}'; one of {sorted(FORMATS)}")
    language = (params.get("language") or "").strip().lower() or None
    if name == "chatterbox-multilingual":
        from chatterbox.mtl_tts import SUPPORTED_LANGUAGES
        if language and language not in SUPPORTED_LANGUAGES:
            raise HTTPException(422, f"unknown language '{language}'; one of {sorted(SUPPORTED_LANGUAGES)}")
    elif language not in (None, "en"):
        raise HTTPException(422, f"{name} speaks English only; use model chatterbox-multilingual for '{language}'")
    return name, language, fmt


def _check_text(text: str) -> str:
    text = (text or "").strip()
    if not text:
        raise HTTPException(422, "`input` is empty")
    if len(text) > MAX_INPUT_CHARS:
        raise HTTPException(413, f"`input` is {len(text)} characters; the limit is {MAX_INPUT_CHARS}")
    return text


async def _run(name, text, language, ref_path, fmt, temperature, exaggeration, cfg_weight, seed):
    global _pending
    if not _state["ready"]:
        raise HTTPException(503, "models are still loading")
    with _pending_lock:
        if _pending >= MAX_PENDING:
            return JSONResponse({"error": "busy: too many requests are running or waiting"}, status_code=429,
                                headers={"Retry-After": "30"})
        _pending += 1
    arrived = time.time()
    deadline = arrived + MAX_SECONDS

    def job():
        if not _run_lock.acquire(timeout=max(1.0, deadline - time.time())):
            raise _Deadline()
        try:
            t_load = time.time()
            _load_model(name)  # a cold model is loaded here so the speed figures below exclude it
            load_s = time.time() - t_load
            t_gen = time.time()
            audio, sr, pieces = _synthesise(name, text, language, ref_path, temperature, exaggeration,
                                            cfg_weight, seed, deadline)
            return audio, sr, pieces, time.time() - t_gen, load_s
        finally:
            _run_lock.release()

    try:
        try:
            audio, sr, pieces, t0, load_s = await asyncio.to_thread(job)
        except _Deadline:
            return JSONResponse({"error": f"could not finish within {MAX_SECONDS:.0f} s; send less text"},
                                status_code=504)
        except AssertionError as exc:
            raise HTTPException(422, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        body = _encode(audio, sr, fmt)
        seconds = len(audio) / sr
        headers = {"X-Processing-Seconds": f"{t0:.2f}", "X-Audio-Seconds": f"{seconds:.2f}",
                   "X-Realtime-Factor": f"{t0 / seconds:.2f}" if seconds else "0", "X-Model": name,
                   "X-Sample-Rate": str(sr), "X-Pieces": str(pieces), "X-Load-Seconds": f"{load_s:.2f}"}
        return Response(body, media_type=FORMATS[fmt][0], headers=headers)
    finally:
        with _pending_lock:
            _pending -= 1


@app.middleware("http")
async def guards(request: Request, call_next):
    path = request.url.path.rstrip("/") or "/"
    if path in PUBLIC_PATHS:
        return await call_next(request)
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    if not (scheme.lower() == "bearer" and API_KEY and token == API_KEY):
        return JSONResponse({"error": "Missing or invalid Authorization: Bearer <API_KEY>."},
                            status_code=401, headers={"WWW-Authenticate": "Bearer"})
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_REFERENCE_BYTES + 65536:
        return JSONResponse({"error": "request body is too large"}, status_code=413)
    return await call_next(request)


@app.get("/health", include_in_schema=False)
def health():
    if not _state["ready"]:
        return JSONResponse({"status": "loading" if not _state["error"] else "failed",
                             "error": _state["error"]}, status_code=503)
    return {"status": "ok", "loaded_in_s": _state["loaded_in"], "threads": THREADS,
            "loaded_models": list(_models), "rss_mb": _status_kb("VmRSS") // 1024,
            "peak_rss_mb": _status_kb("VmHWM") // 1024}


@app.get("/v1/models")
def models():
    from chatterbox.mtl_tts import SUPPORTED_LANGUAGES
    return {"object": "list", "data": [
        {"id": name, "object": "model", "description": desc, "default": name == DEFAULT_MODEL, "license": "MIT",
         "loaded": name in _models,
         **({"languages": SUPPORTED_LANGUAGES} if name == "chatterbox-multilingual" else {"languages": {"en": "English"}})}
        for name, (_d, desc) in MODELS.items()],
        "limits": {"max_input_chars": MAX_INPUT_CHARS, "max_seconds": MAX_SECONDS, "max_pending": MAX_PENDING,
                   "formats": sorted(FORMATS), "max_reference_bytes": MAX_REFERENCE_BYTES,
                   "sample_rate": 24000, "max_loaded_models": MAX_LOADED_MODELS}}


class SpeechRequest(BaseModel):
    input: str
    model: str | None = None
    voice: str | None = Field(default=None, description="accepted for OpenAI clients; the built-in voice is used")
    response_format: str | None = "wav"
    speed: float | None = None
    language: str | None = None
    temperature: float = Field(default=0.8, ge=0.05, le=2.0)
    exaggeration: float = Field(default=0.5, ge=0.0, le=2.0)
    cfg_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    seed: int | None = None


@app.post("/v1/audio/speech")
async def speech(req: SpeechRequest):
    if req.speed not in (None, 1.0):
        raise HTTPException(422, "`speed` is not supported; omit it or send 1.0")
    text = _check_text(req.input)
    name, language, fmt = _resolve(req.model_dump())
    return await _run(name, text, language, None, fmt, req.temperature, req.exaggeration, req.cfg_weight, req.seed)


@app.post("/v1/audio/clone")
async def clone(request: Request):
    ctype = request.headers.get("content-type", "")
    if not ctype.startswith("multipart/form-data"):
        raise HTTPException(415, "send multipart/form-data with a `reference` audio file and an `input` field")
    form = await request.form(max_files=1, max_fields=16)
    ref = form.get("reference")
    if ref is None or not hasattr(ref, "read"):
        raise HTTPException(422, "attach the reference clip as multipart file field `reference`")
    data = await ref.read(MAX_REFERENCE_BYTES + 1)
    if len(data) > MAX_REFERENCE_BYTES:
        raise HTTPException(413, f"`reference` is larger than {MAX_REFERENCE_BYTES} bytes")
    if not data:
        raise HTTPException(422, "`reference` is empty")
    fields = {k: v for k, v in form.items() if isinstance(v, str)}
    text = _check_text(fields.get("input", ""))
    name, language, fmt = _resolve(fields)
    try:
        temperature = float(fields.get("temperature", 0.8))
        exaggeration = float(fields.get("exaggeration", 0.5))
        cfg_weight = float(fields.get("cfg_weight", 0.5))
        seed = int(fields["seed"]) if fields.get("seed") else None
    except ValueError as exc:
        raise HTTPException(422, "temperature, exaggeration, cfg_weight must be numbers and seed an integer") from exc
    if not (0.05 <= temperature <= 2.0 and 0.0 <= exaggeration <= 2.0 and 0.0 <= cfg_weight <= 1.0):
        raise HTTPException(422, "temperature 0.05-2, exaggeration 0-2, cfg_weight 0-1")
    fd, path = tempfile.mkstemp(suffix=".audio")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return await _run(name, text, language, path, fmt, temperature, exaggeration, cfg_weight, seed)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def main() -> int:
    if not API_KEY:
        print("[chatterbox] FATAL: API_KEY is empty; refusing to publish the API to anyone who finds the URL.",
              file=sys.stderr, flush=True)
        return 1
    if DEFAULT_MODEL not in MODELS:
        print(f"[chatterbox] FATAL: DEFAULT_MODEL '{DEFAULT_MODEL}' is not one of {sorted(MODELS)}",
              file=sys.stderr, flush=True)
        return 1
    threading.Thread(target=_boot, name="load", daemon=True).start()
    # Railway injects PORT and its healthcheck dials that injected port. asyncio sets
    # IPV6_V6ONLY on a "::" listener, so one socket per family: IPv4 for the edge and
    # healthcheck, IPv6 for the private network (<service>.railway.internal).
    port = int(os.environ.get("PORT") or 8080)
    socks = []
    for family, addr in ((socket.AF_INET, "0.0.0.0"), (socket.AF_INET6, "::")):
        try:
            sock = socket.socket(family, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind((addr, port))
            sock.set_inheritable(True)
            socks.append(sock)
        except OSError as exc:
            print(f"[chatterbox] cannot bind {addr}:{port}: {exc}", flush=True)
    if not socks:
        return 1
    uvicorn.Server(uvicorn.Config(app, log_level=os.environ.get("LOG_LEVEL", "warning"))).run(sockets=socks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
