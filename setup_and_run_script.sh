#!/bin/bash
set -e

# ============================================================
# BehaveSHINE — One-Shot Setup & Eval for Vast.ai
# ============================================================
# Run from anywhere:
#   curl -sL <raw_url> | bash
# Or:
#   git clone ... && cd Behave_SHINE && bash setup_eval.sh
# ============================================================

echo "═══════════════════════════════════════════════════════"
echo "  BehaveSHINE — Full Setup & Eval"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── 0. Clone repo if not already in it ──
if [ ! -f "metanetwork_family.py" ]; then
    echo "[0/5] Cloning repo..."
    git clone https://github.com/Maayan99/Behave_SHINE.git
    cd Behave_SHINE
else
    echo "[0/5] Already in Behave_SHINE repo, skipping clone."
fi

# ── 1. HuggingFace login ──
echo ""
echo "[1/5] HuggingFace authentication"
echo "    Paste your HF token (input is hidden):"
read -s -p "    Token: " HF_TOKEN
echo ""
huggingface-cli login --token "$HF_TOKEN"
echo "  ✓ Logged in"

# ── 2. Create conda env ──
echo ""
echo "[2/5] Creating conda environment..."
conda create -n behaveshine python=3.11 -y
eval "$(conda shell.bash hook)"
conda activate behaveshine

# ── 3. Install dependencies ──
echo ""
echo "[3/5] Installing pip packages..."
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

# ── 4. Download model + checkpoint concurrently ──
echo ""
echo "[4/5] Downloading Qwen3-8B + SHINE checkpoint (concurrent)..."
mkdir -p models ckpts

huggingface-cli download Qwen/Qwen3-8B \
    --local-dir models/Qwen3-8B &
PID_MODEL=$!

huggingface-cli download Nitai99/BehaveSHINE-v2-checkpoints \
    --include "checkpoint-step-1250/*" \
    --local-dir ckpts &
PID_CKPT=$!

echo "  Waiting for downloads..."
wait $PID_MODEL && echo "  ✓ Qwen3-8B ready" || echo "  ✗ Qwen3-8B failed!"
wait $PID_CKPT && echo "  ✓ Checkpoint ready" || echo "  ✗ Checkpoint failed!"

# ── 5. Verify ──
echo ""
echo "[5/5] Verifying..."
ls models/Qwen3-8B/config.json >/dev/null 2>&1 && echo "  ✓ models/Qwen3-8B/config.json" || echo "  ✗ Qwen3-8B missing!"
ls ckpts/checkpoint-step-1250/ >/dev/null 2>&1 && echo "  ✓ ckpts/checkpoint-step-1250/" || echo "  ✗ Checkpoint missing!"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Setup complete!"
echo "═══════════════════════════════════════════════════════"
echo ""

# ── Prompt to run eval ──
read -p "Run an eval script now? [Y/n]: " RUN_EVAL
RUN_EVAL=${RUN_EVAL:-Y}

if [[ "$RUN_EVAL" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Enter the full command (python ... ):"
    echo "  Example: python gpt4tools_random_eval.py --gpt4tools_path gpt4tools_test_unseen_tools.json --checkpoints step-1250:./ckpts/checkpoint-step-1250 --n 300 --seed 42 --scale 0.001"
    echo ""
    read -p "  > " EVAL_CMD
    echo ""
    echo "Running: $EVAL_CMD"
    echo "═══════════════════════════════════════════════════════"
    eval "$EVAL_CMD"
else
    echo "Done. Run your eval manually when ready."
fi
