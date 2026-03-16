#!/bin/bash
set -e

echo '========================================'
echo ' Octo 环境重装脚本'
echo '========================================'

# ── 1. 删除旧环境 ─────────────────────────────
echo '[1/6] 删除旧环境 jcc ...'
conda deactivate 2>/dev/null || true
conda env remove -n jcc -y 2>/dev/null || true

# ── 2. 创建新环境 ─────────────────────────────
echo '[2/6] 创建新环境 jcc (python 3.10) ...'
conda create -n jcc python=3.10 -y

# ── 3. 激活环境 ───────────────────────────────
echo '[3/6] 激活环境 ...'
source ~/miniconda3/etc/profile.d/conda.sh
conda activate jcc

# ── 4. 安装 JAX 0.4.20 + CUDA 12（官方推荐方式）─
echo '[4/6] 安装 jax 0.4.20 + cuda12 ...'
pip install --upgrade pip
pip install "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 验证 GPU
echo '[*] 验证 JAX GPU ...'
python -c "import jax; print('JAX devices:', jax.devices())"

# ── 5. 安装 octo 依赖 ─────────────────────────
echo '[5/6] 安装 octo 及依赖 ...'
cd ~/octo
pip install -e .
pip install -r requirements.txt

# ── 6. 验证 ───────────────────────────────────
echo '[6/6] 最终验证 ...'
python -c "
import jax
import numpy as np
print('JAX version:', jax.__version__)
print('JAX devices:', jax.devices())
from octo.model.octo_model import OctoModel
print('Octo import OK')
"

echo ''
echo '========================================'
echo ' 安装完成！运行推理：'
echo '   conda activate jcc'
echo '   cd ~/octo && python examples/01_inference.py'
echo '========================================'
