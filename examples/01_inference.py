import os
import sys

import numpy as _np

# ── scipy / jax 兼容性补丁（必须在所有其他 import 之前）────────────────────────
import scipy.linalg as _sl

if not hasattr(_sl, "tril"):
    _sl.tril = _np.tril
if not hasattr(_sl, "triu"):
    _sl.triu = _np.triu

import jax
import jax.tree_util as _jtu

if not hasattr(jax, "tree"):

    class _TreeShim:
        def __getattr__(self, name):
            fn = getattr(_jtu, "tree_" + name, None) or getattr(_jtu, name)
            return fn

    jax.tree = _TreeShim()

import numpy as np

# ── 配置 ──────────────────────────────────────────────────────────────────────
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

OCTO_ROOT = os.path.expanduser("~/octo")
sys.path.append(OCTO_ROOT)
PRETRAINED_PATH = os.path.join(OCTO_ROOT, "models/octo-base-1.5")

from octo.model.octo_model import OctoModel


# ══════════════════════════════════════════════════════════════════════════════
# Step 1: 单张图像推理（最简示例）
# ══════════════════════════════════════════════════════════════════════════════
def step1_minimal_inference(model):
    print("\n── Step 1: Minimal Inference ─────────────────────────────────────")

    # 模拟一张 256x256 图像（实际使用时替换为真实图像）
    img = np.zeros((256, 256, 3), dtype=np.uint8)

    # 构造观察字典：(batch=1, time=1, H, W, C)
    observation = {
        "image_primary": img[np.newaxis, np.newaxis],
        "timestep_pad_mask": np.array([[True]]),
    }

    task = model.create_tasks(texts=["pick up the fork"])
    action = model.sample_actions(
        observation,
        task,
        unnormalization_statistics=model.dataset_statistics["bridge_dataset"]["action"],
        rng=jax.random.PRNGKey(0),
    )
    print(f"[+] action shape (batch, horizon, dim): {action.shape}")
    print(f"    first action (7-DoF): {action[0, 0]}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2: 滑动窗口轨迹推理
# ══════════════════════════════════════════════════════════════════════════════
def step2_trajectory_inference(model):
    print("\n── Step 2: Trajectory Inference ──────────────────────────────────")
    WINDOW_SIZE = 2
    NUM_STEPS = 5

    # 模拟 NUM_STEPS+1 帧图像（实际使用时替换为真实轨迹帧）
    images = [
        np.zeros((256, 256, 3), dtype=np.uint8)
        for _ in range(NUM_STEPS + WINDOW_SIZE - 1)
    ]
    language_instruction = "pick up the fork"

    task = model.create_tasks(texts=[language_instruction])

    pred_actions = []
    for step in range(NUM_STEPS):
        input_images = np.stack(images[step : step + WINDOW_SIZE])[
            np.newaxis
        ]  # (1, W, H, W, C)
        observation = {
            "image_primary": input_images,
            "timestep_pad_mask": np.full((1, input_images.shape[1]), True, dtype=bool),
        }
        actions = model.sample_actions(
            observation,
            task,
            unnormalization_statistics=model.dataset_statistics["bridge_dataset"][
                "action"
            ],
            rng=jax.random.PRNGKey(0),
        )
        pred_actions.append(actions[0])  # remove batch dim

    pred_actions = np.array(pred_actions)  # (steps, horizon, dim)
    print(f"[+] pred_actions shape: {pred_actions.shape}")  # (5, 4, 7)
    print(f"    step 0, first action: {pred_actions[0, 0]}")


# ══════════════════════════════════════════════════════════════════════════════
def main():
    print(f"[*] JAX devices: {jax.devices()}")
    print(f"[*] Loading model from: {PRETRAINED_PATH}")
    model = OctoModel.load_pretrained(PRETRAINED_PATH)
    print("[+] Model loaded!")

    step1_minimal_inference(model)
    step2_trajectory_inference(model)

    print("\n[+] All done!")


if __name__ == "__main__":
    main()
