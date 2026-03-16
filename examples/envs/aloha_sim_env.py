import copy
from typing import List

import dlimp as dl
import gym
import jax.numpy as jnp
import numpy as np

# need to put https://github.com/tonyzhaozh/act in your PATH for this import to work
from sim_env import BOX_POSE, make_sim_env


class AlohaGymEnv(gym.Env):
    def __init__(
        self,
        env: gym.Env,
        camera_names: List[str],
        im_size: int = 256,
        seed: int = 1234,
        task_name: str = "sim_transfer_cube",
    ):
        self._env = env
        self.observation_space = gym.spaces.Dict(
            {
                **{
                    f"image_{i}": gym.spaces.Box(
                        low=np.zeros((im_size, im_size, 3)),
                        high=255 * np.ones((im_size, im_size, 3)),
                        dtype=np.uint8,
                    )
                    for i in ["primary", "wrist"][: len(camera_names)]
                },
                "proprio": gym.spaces.Box(
                    low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((14,)) * -1, high=np.ones((14,)), dtype=np.float32
        )
        self.camera_names = camera_names
        self._im_size = im_size
        self._rng = np.random.default_rng(seed)
        self._task_name = task_name

    def step(self, action):
        ts = self._env.step(action)
        obs, images = self.get_obs(ts)
        reward = ts.reward
        info = {"images": images}

        if reward == self._env.task.max_reward:
            self._episode_is_success = 1

        return obs, reward, False, False, info

    def reset(self, **kwargs):
        if self._task_name == "sim_transfer_cube":
            x_range = [0.0, 0.2]
            y_range = [0.4, 0.6]
            z_range = [0.05, 0.05]
            ranges = np.vstack([x_range, y_range, z_range])
            cube_position = self._rng.uniform(ranges[:, 0], ranges[:, 1])
            cube_quat = np.array([1, 0, 0, 0])
            BOX_POSE[0] = np.concatenate([cube_position, cube_quat])
        elif self._task_name == "sim_insertion":
            peg_ranges = np.vstack([[0.1, 0.2], [0.4, 0.6], [0.05, 0.05]])
            peg_position = self._rng.uniform(peg_ranges[:, 0], peg_ranges[:, 1])
            peg_pose = np.concatenate([peg_position, np.array([1, 0, 0, 0])])

            socket_ranges = np.vstack([[-0.2, -0.1], [0.4, 0.6], [0.05, 0.05]])
            socket_position = self._rng.uniform(
                socket_ranges[:, 0], socket_ranges[:, 1]
            )
            socket_pose = np.concatenate([socket_position, np.array([1, 0, 0, 0])])
            BOX_POSE[0] = np.concatenate([peg_pose, socket_pose])
        else:
            raise ValueError(f"Unsupported task: {self._task_name}")

        ts = self._env.reset(**kwargs)
        obs, images = self.get_obs(ts)
        info = {"images": images}
        self._episode_is_success = 0

        return obs, info

    def get_obs(self, ts):
        curr_obs = {}
        vis_images = []

        obs_img_names = ["primary", "wrist"]
        for i, cam_name in enumerate(self.camera_names):
            curr_image = ts.observation["images"][cam_name]
            vis_images.append(copy.deepcopy(curr_image))
            curr_image = jnp.array(curr_image)
            curr_obs[f"image_{obs_img_names[i]}"] = curr_image
        curr_obs = dl.transforms.resize_images(
            curr_obs, match=curr_obs.keys(), size=(self._im_size, self._im_size)
        )

        qpos_numpy = np.array(ts.observation["qpos"])
        qpos = jnp.array(qpos_numpy)
        curr_obs["proprio"] = qpos

        return curr_obs, np.concatenate(vis_images, axis=-2)

    def get_task(self):
        if self._task_name == "sim_insertion":
            instruction = "insert the peg into the socket"
        else:
            instruction = "pick up the cube and hand it over"
        return {
            "language_instruction": [instruction],
        }

    def get_episode_metrics(self):
        return {
            "success_rate": self._episode_is_success,
        }


# register gym environments
gym.register(
    "aloha-sim-cube-v0",
    entry_point=lambda: AlohaGymEnv(
        make_sim_env("sim_transfer_cube"),
        camera_names=["top"],
        task_name="sim_transfer_cube",
    ),
)

gym.register(
    "aloha-sim-insertion-v0",
    entry_point=lambda: AlohaGymEnv(
        make_sim_env("sim_insertion"),
        camera_names=["top"],
        task_name="sim_insertion",
    ),
)
