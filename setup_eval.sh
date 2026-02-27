#!/bin/bash
set -e

# ============================================================
# BehaveSHINE GPT4Tools Eval — Vast.ai Setup
# ============================================================
# Prerequisites: 
#   1. git clone your repo and cd into it
#   2. huggingface-cli login (for private model access)
#   3. Then run: bash setup_eval.sh
# ============================================================

echo "═══════════════════════════════════════════════════════"
echo "  BehaveSHINE GPT4Tools Eval Setup"
echo "═══════════════════════════════════════════════════════"

# ── 1. Create conda env ──
echo "[1/4] Creating conda environment..."
conda create -n behaveshine python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate behaveshine

# ── 2. Install dependencies ──
echo "[2/4] Installing pip packages..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install \
    huggingface==0.0.1 \
    modelscope==1.31.0 \
    transformers==4.57.1 \
    datasets==4.4.1 \
    scikit-learn==1.7.2 \
    hydra-core==1.3.2 \
    tensorboard==2.20.0 \
    openai==2.6.1 \
    rouge==1.0.1 \
    seaborn==0.13.2 \
    matplotlib==3.10.7 \
    multiprocess==0.70.16

# ── 3 & 4. Download model + checkpoint concurrently ──
echo "[3/4] Downloading Qwen3-8B and SHINE checkpoint (concurrent)..."
mkdir -p models
mkdir -p ckpts

# Download both in parallel
huggingface-cli download Qwen/Qwen3-8B \
    --local-dir models/Qwen3-8B &
PID_MODEL=$!

huggingface-cli download Nitai99/BehaveSHINE-v2-checkpoints \
    --include "checkpoint-step-1250/*" \
    --local-dir ckpts &
PID_CKPT=$!

echo "  Waiting for downloads (model PID=$PID_MODEL, ckpt PID=$PID_CKPT)..."
wait $PID_MODEL
echo "  ✓ Qwen3-8B downloaded"
wait $PID_CKPT
echo "  ✓ SHINE checkpoint downloaded"

# ── Verify ──
echo ""
echo "[4/4] Verifying..."
ls models/Qwen3-8B/config.json >/dev/null 2>&1 && echo "  ✓ models/Qwen3-8B/config.json exists" || echo "  ✗ Qwen3-8B missing!"
ls ckpts/checkpoint-step-1250/ >/dev/null 2>&1 && echo "  ✓ ckpts/checkpoint-step-1250/ exists" || echo "  ✗ Checkpoint missing!"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Setup complete! Activate env with:"
echo "    conda activate behaveshine"
echo ""
echo "  Then run eval with:"
echo "    python eval_gpt4tools.py \\"
echo "      --gpt4tools_path <PATH_TO_gpt4tools_test_unseen.json> \\"
echo "      --checkpoints step-1250:./ckpts/checkpoint-step-1250 \\"
echo "      --n 300 --seed 42 --scale 0.001"
echo "═══════════════════════════════════════════════════════"
