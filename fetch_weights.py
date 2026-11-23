"""Download the Chatterbox weights into /models at image build, pinned to Hugging Face commit SHAs.

A download at boot would sit inside Railway's healthcheck window, so the weights are part of the image.
Each revision below is a commit of the model repository; the files fetched are content-addressed by it.
"""
import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(os.environ.get("MODEL_ROOT", "/models"))
TURBO_REV = "749d1c1a46eb10492095d68fbcf55691ccf137cd"  # ResembleAI/chatterbox-turbo, 2025-12-15, license: mit
NANO_REV = "71ccd1d0081b430592cea481f4307e764e07bc64"  # ResembleAI/chatterbox-nano, 2026-07-21, license: mit
MTL_REV = "5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18"  # ResembleAI/chatterbox, 2026-06-10, license: mit

TOKENIZER = ["added_tokens.json", "merges.txt", "special_tokens_map.json", "tokenizer_config.json", "vocab.json"]

turbo = ROOT / "turbo"
snapshot_download("ResembleAI/chatterbox-turbo", revision=TURBO_REV, local_dir=turbo,
                  allow_patterns=["t3_turbo_v1.safetensors", "s3gen_meanflow.safetensors", "ve.safetensors",
                                  "conds.pt", *TOKENIZER])

# Nano shares the voice encoder, the S3Gen flow weights and the tokenizer with Turbo; only the T3 checkpoint
# and the built-in voice differ. Files already in the Turbo directory are linked, not stored twice.
nano = ROOT / "nano"
snapshot_download("ResembleAI/chatterbox-nano", revision=NANO_REV, local_dir=nano,
                  allow_patterns=["t3_nano_v1.safetensors", "conds.pt"])
for name in ["s3gen_meanflow.safetensors", "ve.safetensors", *TOKENIZER]:
    link = nano / name
    if not link.exists():
        link.symlink_to(Path("..") / "turbo" / name)

mtl = ROOT / "multilingual"
snapshot_download("ResembleAI/chatterbox", revision=MTL_REV, local_dir=mtl,
                  allow_patterns=["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt",
                                  "grapheme_mtl_merged_expanded_v1.json", "conds.pt"])
# The tokenizer reads the Cangjie table through hf_hub_download(cache_dir=<model dir>); cache it there so
# HF_HUB_OFFLINE=1 can resolve it.
hf_hub_download("ResembleAI/chatterbox", "Cangjie5_TC.json", revision=MTL_REV, cache_dir=mtl)
# The code asks for the default revision ("main"), so mirror the commit under refs/main.
refs = mtl / "models--ResembleAI--chatterbox" / "refs"
refs.mkdir(parents=True, exist_ok=True)
(refs / "main").write_text(MTL_REV)

for p in sorted(ROOT.rglob("*")):
    if p.is_file() and not p.is_symlink():
        print(f"{p.stat().st_size:>12} {p}")
