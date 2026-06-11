#!/bin/bash
# Setup script for running benchmarks on local GPU (WSL2)
# Run this from the project root: bash scripts/setup_bench.sh

set -e

echo "=== TensorPath Benchmark Setup ==="

# check GPU
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# check python
PYTHON=${PYTHON:-python3}
echo "Python: $($PYTHON --version)"

# create venv if needed
if [ ! -d ".venv-bench" ]; then
    echo "Creating benchmark venv..."
    $PYTHON -m venv .venv-bench
fi

source .venv-bench/bin/activate

# install deps
echo "Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
pip install -r benchmarks/requirements.txt -q

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run benchmarks:"
echo "  source .venv-bench/bin/activate"
echo ""
echo "  # offline runner (quick, approximate latency)"
echo "  python benchmarks/runners/bench.py --model llama3.2-3b --quantization fp16"
echo "  python benchmarks/runners/bench.py --model qwen2.5-7b --quantization awq"
echo "  python benchmarks/runners/bench.py --model llama3.1-8b --quantization awq"
echo ""
echo "  # server runner (slower setup, accurate streaming latency)"
echo "  python benchmarks/runners/bench_server.py --model llama3.2-3b --quantization fp16"
echo ""
echo "Models that fit on 12GB VRAM:"
echo "  llama3.2-3b  fp16  (~6.5 GB)  <-- start here"
echo "  llama3.2-3b  awq   (~2.8 GB)"
echo "  qwen2.5-7b   awq   (~5.8 GB)"
echo "  llama3.1-8b  awq   (~6.2 GB)"
echo ""
echo "Models that WON'T fit:"
echo "  qwen2.5-7b   fp16  (~14.5 GB)  too big"
echo "  llama3.1-8b  fp16  (~16 GB)    too big"
