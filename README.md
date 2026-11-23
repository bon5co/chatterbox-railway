# chatterbox-railway

Chatterbox text-to-speech API for [Railway](https://railway.com), CPU only. Serves [resemble-ai/chatterbox](https://github.com/resemble-ai/chatterbox) (MIT) behind one authenticated, OpenAI-compatible HTTP API.

- `POST /v1/audio/speech` (JSON): `input`, `model` (`chatterbox-turbo` default, `chatterbox-nano`, `chatterbox-multilingual`), `response_format` (`wav` default, `mp3`, `flac`, `pcm`), `language` (multilingual only), `temperature`, `exaggeration` and `cfg_weight` (multilingual only), `seed`. `voice` is accepted for OpenAI clients and the built-in voice is used.
- `POST /v1/audio/clone` (multipart): the same fields plus a `reference` audio file; speech in that voice, nothing stored.
- `GET /v1/models`: the three models, the 23 languages and the limits in force.
- `GET /health` (503 until the default model is loaded and warm, then 200).
- Auth: `Authorization: Bearer $API_KEY` on every route except `/health`. An empty `API_KEY` refuses to start.
- Env: see `env.md`; `API_KEY` is the only required one.

Weights are baked into the image from three MIT-licensed, ungated Hugging Face repositories pinned to commit SHAs. Text and audio are held in memory for one request and never logged or stored. Every file the models produce carries the upstream Perth watermark. Wrapper code is Apache-2.0 (`LICENSE`); see `NOTICE`, including the upstream retirement notice.

Image: `ghcr.io/bon5co/chatterbox-railway`.
