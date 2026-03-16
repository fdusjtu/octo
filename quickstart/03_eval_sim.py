"""
ALOHA 仿真评估脚本（本地修改版）

基于 examples/03_eval_finetuned.py，做了以下改动：
  - 添加 scipy/jax 兼容性补丁（支持 jax==0.4.20 + scipy>=1.13）
  - 去掉 wandb，改为本地保存 mp4 视频到 rollout_videos/
  - 固定 ACT 仿真环境路径为 /home/cjt/act
  - 设置离线模式环境变量

运行方式：
    cd ~/octo/quickstart
    python 03_eval_sim.py --finetuned_path=/home/cjt/octo/models/octo-base-1.5

依赖安装：
    pip install mujoco==2.3.3 dm_control==1.0.9 imageio[ffmpeg]

无头服务器需要虚拟显示：
    Xvfb :1 -screen 0 1024x768x16 &
    export DISPLAY=:1
"""
from functools import partial
import os
import sys

# ── scipy / jax 兼容性补丁（必须在所有其他 import 之前）────────────────────────
import scipy.linalg as _sl
import numpy as _np
if not hasattr(_sl, 'tril'):
    _sl.tril = _np.tril
if not hasattr(_sl, 'triu'):
    _sl.triu = _np.triu

import jax
import jax.tree_util as _jtu
if not hasattr(jax, 'tree'):
    class _TreeShim:
        def __getattr__(self, name):
            fn = getattr(_jtu, 'tree_' + name, None) or getattr(_jtu, name)
            return fn
    jax.tree = _TreeShim()

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')

from absl import app, flags, logging
import gym
import matplotlib
matplotlib.use('TkAgg' if 'DISPLAY' in os.environ else 'Agg')
import matplotlib.pyplot as plt
import numpy as np
import imageio

sys.path.append("/home/cjt/act")
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'examples'))

# keep this to register ALOHA sim env
from envs.aloha_sim_env import AlohaGymEnv  # noqa

sys.path.insert(0, os.path.expanduser('~/octo'))
from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper, NormalizeProprio, RHCWrapper
from octo.utils.train_callbacks import supply_rng

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "finetuned_path", None, "Path to finetuned Octo checkpoint directory."
)


def main(_):
    # setup output directory for saving videos
    os.makedirs("rollout_videos", exist_ok=True)

    # load model
    logging.info("Loading model...")
    model = OctoModel.load_pretrained(FLAGS.finetuned_path)
    logging.info(f"JAX devices: {jax.devices()}")
    logging.info(f"dataset_statistics keys: {list(model.dataset_statistics.keys())}")

    # make gym environment
    env = gym.make("aloha-sim-cube-v0")

    # 注意：NormalizeProprio 需要微调模型的 proprio 统计数据，预训练模型跳过
    # env = NormalizeProprio(env, model.dataset_statistics)

    # add wrappers for history and receding horizon control
    # exec_horizon must be <= model's action_horizon (octo-base outputs 4 steps)
    model_action_horizon = 4  # octo-base-1.5 default
    env = HistoryWrapper(env, horizon=1)
    env = RHCWrapper(env, exec_horizon=model_action_horizon)

    # 预训练模型用 bridge_dataset 的统计数据，微调模型直接用 "action"
    if "action" in model.dataset_statistics:
        unnorm_stats = model.dataset_statistics["action"]
    elif "bridge_dataset" in model.dataset_statistics:
        unnorm_stats = model.dataset_statistics["bridge_dataset"]["action"]
        logging.info("Using unnormalization stats from: bridge_dataset")
    else:
        first_dataset = next(iter(model.dataset_statistics))
        unnorm_stats = model.dataset_statistics[first_dataset]["action"]
        logging.info(f"Using unnormalization stats from dataset: {first_dataset}")

    # supply_rng wrapper provides a new random key to sample_actions each call
    policy_fn = supply_rng(
        partial(
            model.sample_actions,
            unnormalization_statistics=unnorm_stats,
        ),
    )

    # 设置 matplotlib 实时显示（有 DISPLAY 才弹窗，否则只保存视频）
    has_display = 'DISPLAY' in os.environ
    if has_display:
        plt.ion()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axis('off')
    plt_img = None

    # run 3 episodes
    for ep in range(3):
        obs, info = env.reset()

        language_instruction = env.get_task()["language_instruction"]
        task = model.create_tasks(texts=language_instruction)

        images = [obs["image_primary"][0]]
        episode_return = 0.0

        # 显示第一帧
        if plt_img is None:
            plt_img = ax.imshow(images[0])
        else:
            plt_img.set_data(images[0])
        ax.set_title(f'Episode {ep} | Step 0 | Return: 0.00')
        if has_display:
            plt.pause(0.001)

        while len(images) < 400:
            # 去掉 proprio（octo-base 不认识该键），只保留图像输入
            obs_for_model = {k: v for k, v in obs.items() if k != 'proprio'}
            actions = policy_fn(jax.tree_util.tree_map(lambda x: x[None], obs_for_model), task)
            logging.info(f'raw actions shape: {np.array(actions).shape}')
            actions = np.array(actions[0])  # (action_horizon, action_dim) e.g. (4, 7)
            logging.info(f'actions[0] shape: {actions.shape}')

            # ALOHA 环境需要 14-DoF 动作，octo-base 输出 7-DoF
            # 将 7-DoF 复制为左右臂各一份（仅用于演示，不保证语义正确）
            if actions.ndim == 1:
                # actions 是 (7,)，需要先变成 (1, 7) 再 pad
                actions = actions[None]  # (1, 7)
            if actions.shape[-1] != 14:
                actions = np.concatenate([actions, actions], axis=-1)  # (N, 14)
            logging.info(f'final actions shape: {actions.shape}')

            obs, reward, done, trunc, info = env.step(actions)
            new_imgs = [o["image_primary"][0] for o in info["observations"]]
            images.extend(new_imgs)
            episode_return += reward

            # 实时更新画面（显示最新帧）
            plt_img.set_data(new_imgs[-1])
            ax.set_title(f'Episode {ep} | Step {len(images)} | Return: {episode_return:.2f}')
            if has_display:
                plt.pause(0.001)

            if done or trunc:
                break

        print(f"Episode {ep} return: {episode_return}")

        # save video locally (subsampled 2x)
        video_path = f"rollout_videos/episode_{ep}.mp4"
        imageio.mimwrite(video_path, np.array(images)[::2], fps=10)
        print(f"Video saved to {video_path}")

    if has_display:
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    app.run(main)
