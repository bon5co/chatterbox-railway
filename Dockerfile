# Chatterbox text-to-speech API (Turbo, Nano, Multilingual), Bearer API_KEY, OpenAI-compatible /v1/audio/speech.
# Source: https://github.com/bon5co/chatterbox-railway
# CPU only (torch), weights baked in (about 7 GB), stateless (no volume needed).
# upstream: resemble-ai/chatterbox is installed from commit ab55923c27757c90cfcc407baa57dc25a7d25217 (main, 2026-11-04) rather than PyPI chatterbox-tts 0.1.7, because 0.1.7 predates Chatterbox-Nano (its ChatterboxTurboTTS.from_local takes no `nano` argument) while main carries it under the same 0.1.7 version string; the newest GitHub release is v0.1.2 and the README says the project was retired in August 2026 (checked 2026-11-24). Weights: three Hugging Face repositories pinned to commit SHAs in fetch_weights.py. spacy-pkuseg segmenter model: sha256 pinned by the library itself.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements.txt /tmp/requirements.txt
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements.txt \
    && pip install --no-deps https://github.com/resemble-ai/chatterbox/archive/ab55923c27757c90cfcc407baa57dc25a7d25217.tar.gz \
    && pip list 2>/dev/null | grep -iE "^(torch|torchaudio|numpy|transformers|chatterbox-tts|resemble-perth|soundfile|fastapi) "

# Weights: three MIT-licensed, ungated Hugging Face repositories, pinned to commit SHAs (see fetch_weights.py).
ENV MODEL_ROOT=/models
COPY fetch_weights.py /tmp/fetch_weights.py
# The multilingual tokenizer's Chinese segmenter (spacy-pkuseg) downloads its model from GitHub on first use;
# fetch it here (the library checks the archive against its own pinned sha256) so nothing is fetched at run time.
ENV PKUSEG_HOME=/opt/pkuseg
RUN python -c "from spacy_pkuseg import pkuseg; print(pkuseg().cut('你好，世界'))" && ls -R /opt/pkuseg | head
RUN python /tmp/fetch_weights.py && find /models -name .cache -type d -prune -exec rm -rf {} + && rm -rf /root/.cache && du -sh /models

# MIT licence text of resemble-ai/chatterbox (LICENSE at commit ab55923c27757c90cfcc407baa57dc25a7d25217).
COPY LICENSE-chatterbox /licenses/chatterbox-LICENSE

RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin tts \
    && chmod -R a+rX /models /licenses
WORKDIR /app
COPY app.py entrypoint.sh NOTICE LICENSE /app/
RUN chmod +x /app/entrypoint.sh
# Every model file is local; any attempt to reach Hugging Face at run time is a bug, not a download.
ENV MODEL_ROOT=/models HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/tmp/hf
USER 1000
EXPOSE 8080
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "/app/app.py"]
