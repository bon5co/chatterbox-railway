# Chatterbox TTS API (resemble-ai/chatterbox) — environment

Paste into Railway's variables panel. Everything else lives in the image, because a template variable holding a literal is dropped at generate time and republishes as a blank required field.

```
API_KEY="${{secret(32)}}"
```

## Optional

| Variable | Default | Values |
|---|---|---|
| `DEFAULT_MODEL` | `chatterbox-turbo` | Model used when a request names none, and the one loaded at boot (`chatterbox-turbo`, `chatterbox-nano`, `chatterbox-multilingual`) |
| `MAX_LOADED_MODELS` | `1` | Models kept in memory; asking for another loads it and drops the least recently used |
| `MAX_INPUT_CHARS` | `2000` | Longest `input` accepted (413 above it) |
| `MAX_CHUNK_CHARS` | `280` | Longest piece synthesised in one pass; longer input is split at sentence ends |
| `MAX_SECONDS` | `240` | Wall-clock budget from request arrival; past it the request answers 504 (Railway's edge cuts a connection at about 5 minutes) |
| `MAX_PENDING` | `3` | Requests running or waiting; another answers 429 with `Retry-After: 30` |
| `MAX_REFERENCE_BYTES` | `10485760` | Largest reference clip for `/v1/audio/clone` |

## Set by the entrypoint at boot

`OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and `TORCH_THREADS` come from `/sys/fs/cgroup/cpu.max`, because PyTorch sizes its thread pool from the host's cores and oversubscribes the container.

`PORT` is left to Railway: the healthcheck dials the injected port, so the app honours it.
