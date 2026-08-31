#!/usr/bin/env bash
# Bootstrap Hydra on a fresh vast.ai instance (standard pytorch image).
# Idempotent: safe to re-run on restart (also usable as the instance onstart script).
set -euo pipefail

REPO="https://github.com/Lukasz-G/Hydra.git"
DATA_GDRIVE_ID="1oX9EHozkfPX5EGKwLNCVPXcYiJyBCe05"   # hydra_mhd_corpus.zip
WORK=/workspace

cd "$WORK"
if [ ! -d hydra/.git ]; then
    git clone "$REPO" hydra
fi
cd hydra
git pull --ff-only
pip install -q -e . gdown

mkdir -p "$WORK/data"
if [ ! -f "$WORK/data/.corpus_ok" ]; then
    gdown "$DATA_GDRIVE_ID" -O "$WORK/data/corpus.zip"
    rm -rf "$WORK/data/MHD"
    python - <<'EOF'
import zipfile
zipfile.ZipFile("/workspace/data/corpus.zip").extractall("/workspace/data/MHD")
EOF
    n=$(ls "$WORK/data/MHD" | wc -l)
    echo "extracted $n corpus files"
    [ "$n" -gt 300 ] && touch "$WORK/data/.corpus_ok"
fi

python -m pytest tests -q -x
echo "setup complete — run: bash tools/train_remote.sh runs/<name> [--set key=value ...]"
