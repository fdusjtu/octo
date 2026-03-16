"""
This script demonstrates how to finetune Octo to a new observation space (single camera + proprio)
and new action space (bimanual) using a simulated ALOHA cube handover dataset (https://tonyzhaozh.github.io/aloha/).

To run this example, first download and extract the dataset from here: https://rail.eecs.berkeley.edu/datasets/example_sim_data.zip

python examples/02_finetune_new_observation_action.py --pretrained_path=hf://rail-berkeley/octo-small-1.5 --data_dir=...

python 02_finetune_new_observation_action.py   --pretrained_path=/home/cjt/octo/models/octo-base-1.5   --data_dir=/home/cjt/octo/aloha_sim_dataset   --save_dir=/home/cjt/octo/checkpoints/aloha_ckpts   --batch_size=4   --num_steps=8000
"""
from absl import app, flags, logging
import flax
import jax
import optax
import tensorflow as tf
import tqdm
import wandb

from octo.data.dataset import make_single_dataset
from octo.model.components.action_heads import L1ActionHead
from octo.model.components.tokenizers import LowdimObsTokenizer
from octo.model.octo_model import OctoModel
from octo.utils.jax_utils import initialize_compilation_cache
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import (
    freeze_weights,
    merge_params,
    process_text,
    TrainState,
)

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "pretrained_path", None, "Path to pre-trained Octo checkpoint directory."
)
flags.DEFINE_string("data_dir", None, "Path to finetuning dataset, in RLDS format.")
flags.DEFINE_string("save_dir", None, "Directory for saving finetuning checkpoints.")
flags.DEFINE_integer("batch_size", 128, "Batch size for finetuning.")
flags.DEFINE_integer("num_steps", 4000, "Number of finetuning steps.")
flags.DEFINE_integer(
    "save_interval",
    100,
    "Save a checkpoint every N steps.",
)

flags.DEFINE_bool(
    "freeze_transformer",
    False,
    "Whether pre-trained transformer weights should be frozen.",
)
flags.DEFINE_bool(
    "overwrite_save_dir",
    True,
    "Whether to delete existing step checkpoints in save_dir before training.",
)
flags.DEFINE_bool(
    "save_best_only",
    True,
    "Whether to keep only the checkpoint with the lowest training loss.",
)


def _clear_existing_checkpoints(save_dir):
    if not save_dir or not tf.io.gfile.exists(save_dir):
        return

    for entry in tf.io.gfile.listdir(save_dir):
        full_path = tf.io.gfile.join(save_dir, entry)
        if not tf.io.gfile.isdir(full_path):
            continue
        if entry.isdigit() or ".orbax-checkpoint-tmp-" in entry:
            logging.info("Removing existing checkpoint directory: %s", full_path)
            tf.io.gfile.rmtree(full_path)


def _clear_step_checkpoints(save_dir):
    if not save_dir or not tf.io.gfile.exists(save_dir):
        return

    for entry in tf.io.gfile.listdir(save_dir):
        full_path = tf.io.gfile.join(save_dir, entry)
        if not tf.io.gfile.isdir(full_path):
            continue
        if entry.isdigit() or ".orbax-checkpoint-tmp-" in entry:
            tf.io.gfile.rmtree(full_path)


def main(_):
    assert (
        FLAGS.batch_size % jax.device_count() == 0
    ), "Batch size must be divisible by device count."

    initialize_compilation_cache()
    # prevent tensorflow from using GPU memory since it's only used for data loading
    tf.config.set_visible_devices([], "GPU")

    if FLAGS.save_dir and FLAGS.overwrite_save_dir:
        _clear_existing_checkpoints(FLAGS.save_dir)

    # setup wandb for logging
    wandb.init(name="finetune_aloha", project="octo")

    # load pre-trained model
    logging.info("Loading pre-trained model...")
    pretrained_model = OctoModel.load_pretrained(FLAGS.pretrained_path)

    # make finetuning dataset
    # apply Gaussian normalization, load chunks of 50 actions since we'll train with action chunking
    # delete goal images in the data loader since we will train a language-conditioned-only policy
    # TODO: directly load this from raw data to make it less opaque?
    logging.info("Loading finetuning dataset...")
    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name="aloha_sim_cube_scripted_dataset",
            data_dir=FLAGS.data_dir,
            image_obs_keys={"primary": "top"},
            proprio_obs_key="state",
            language_key="language_instruction",
        ),
        traj_transform_kwargs=dict(
            window_size=1,
            action_horizon=50,
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256)},
        ),
        train=True,
    )
    train_data_iter = (
        dataset.repeat()
        .unbatch()
        .shuffle(10000)  # can reduce this if RAM consumption too high
        .batch(FLAGS.batch_size)
        .iterator()
    )

    # run text tokenizer over batch (this needs to happen before training / sharding) + delete unused keys
    text_processor = pretrained_model.text_processor

    def process_batch(batch):
        batch = process_text(batch, text_processor)
        del batch["dataset_name"]
        return batch

    train_data_iter = map(process_batch, train_data_iter)
    example_batch = next(train_data_iter)

    # load pre-training config and modify --> remove wrist cam, add proprio input, change action head
    # following Zhao et al. we use "action chunks" of length 50 and L1 loss for ALOHA
    config = pretrained_model.config
    del config["model"]["observation_tokenizers"]["wrist"]
    ###
    config["model"]["observation_tokenizers"]["proprio"] = ModuleSpec.create(
        LowdimObsTokenizer,
        n_bins=256,
        bin_type="normal",
        low=-2.0,
        high=2.0,
        obs_keys=["proprio"],
    )
    # Fully override the old action head with a new one (for smaller changes, you can use update_config)
    config["model"]["heads"]["action"] = ModuleSpec.create(
        L1ActionHead,
        action_horizon=50,
        action_dim=14,
        readout_key="readout_action",
    )

    # initialize weights for modified Octo model, then merge in all applicable pre-trained weights
    # new position encodings for proprio inputs & weights for new action head will remain "from scratch"
    logging.info("Updating model for new observation & action space...")
    model = OctoModel.from_config(
        config,
        example_batch,
        text_processor,
        verbose=True,
        dataset_statistics=dataset.dataset_statistics,
    )
    merged_params = merge_params(model.params, pretrained_model.params)
    # can perform any additional parameter surgery here...
    # ...
    model = model.replace(params=merged_params)
    del pretrained_model

    # create optimizer & train_state, optionally freeze keys for pre-trained transformer
    # train_state bundles parameters & optimizers
    learning_rate = optax.join_schedules(
        [optax.linear_schedule(0, 3e-5, 100), optax.constant_schedule(3e-5)], [100]
    )
    tx = optax.adamw(learning_rate)
    frozen_keys = model.config["optimizer"]["frozen_keys"]
    if FLAGS.freeze_transformer:
        frozen_keys.append("BlockTransformer_0")
    tx = freeze_weights(tx, model.params, frozen_keys)
    train_state = TrainState.create(
        rng=jax.random.PRNGKey(1234),
        model=model,
        tx=tx,
    )

    # define loss function and train step
    def loss_fn(params, batch, rng, train=True):
        bound_module = model.module.bind({"params": params}, rngs={"dropout": rng})
        transformer_embeddings = bound_module.octo_transformer(
            batch["observation"],
            batch["task"],
            batch["observation"]["timestep_pad_mask"],
            train=train,
        )
        action_loss, action_metrics = bound_module.heads["action"].loss(
            transformer_embeddings,  # Action head knows to pull out the action readout_key
            batch["action"],
            batch["observation"]["timestep_pad_mask"],
            batch["action_pad_mask"],
            train=train,
        )
        return action_loss, action_metrics

    @jax.jit
    def train_step(state, batch):
        rng, dropout_rng = jax.random.split(state.rng)
        (loss, info), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.model.params, batch, dropout_rng, train=True
        )
        new_state = state.apply_gradients(grads=grads, rng=rng)
        return new_state, info

    # run finetuning loop
    logging.info("Starting finetuning...")
    best_loss = float("inf")
    best_step = None
    for i in tqdm.tqdm(
        range(FLAGS.num_steps), total=FLAGS.num_steps, dynamic_ncols=True
    ):
        batch = next(train_data_iter)
        train_state, update_info = train_step(train_state, batch)
        if (i + 1) % 100 == 0:
            update_info = jax.device_get(update_info)
            wandb.log(
                flax.traverse_util.flatten_dict({"training": update_info}, sep="/"),
                step=i,
            )
        if FLAGS.save_dir and (i + 1) % FLAGS.save_interval == 0:
            current_loss = float(jax.device_get(update_info["loss"]))
            if FLAGS.save_best_only:
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_step = i + 1
                    _clear_step_checkpoints(FLAGS.save_dir)
                    train_state.model.save_pretrained(
                        step=i + 1, checkpoint_path=FLAGS.save_dir
                    )
                    logging.info(
                        "Saved new best checkpoint at step %d with loss %.6f",
                        i + 1,
                        current_loss,
                    )
            else:
                train_state.model.save_pretrained(
                    step=i + 1, checkpoint_path=FLAGS.save_dir
                )

    if FLAGS.save_best_only and best_step is not None:
        logging.info("Best checkpoint: step=%d loss=%.6f", best_step, best_loss)


if __name__ == "__main__":
    app.run(main)
