import glob
import os
import pickle as pkl
import random
import jax
from jax import numpy as jnp
import flax.linen as nn
from flax.training import checkpoints
import numpy as np
import optax
from tqdm import tqdm
from absl import app, flags

import local_imports  # noqa: F401
from serl_launcher.data.data_store import ReplayBuffer
from serl_launcher.utils.train_utils import concat_batches
from serl_launcher.vision.data_augmentations import batched_random_crop
from serl_launcher.networks.reward_classifier import create_classifier

from experiments.mappings import CONFIG_MAPPING


FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", None, "Name of experiment corresponding to folder.")
flags.DEFINE_integer("num_epochs", 200, "Number of training epochs.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_float("val_ratio", 0.2, "Ratio of validation data.")


def load_and_split_data(paths, label, env, val_ratio=0.2):
    """加载 pkl 文件并打乱，按比例切分为训练集和验证集列表"""
    all_transitions = []
    for path in paths:
        data = pkl.load(open(path, "rb"))
        for trans in data:
            if "images" in trans['observations'].keys():
                continue
            trans["labels"] = label
            trans['actions'] = env.action_space.sample()
            all_transitions.append(trans)
    
    # 打乱数据以保证切分均匀
    random.seed(42)
    random.shuffle(all_transitions)
    
    val_size = int(len(all_transitions) * val_ratio)
    val_data = all_transitions[:val_size]
    train_data = all_transitions[val_size:]
    
    return train_data, val_data


def populate_buffer(buffer, transitions):
    """将数据填充至 ReplayBuffer"""
    for trans in transitions:
        buffer.insert(trans)


def main(_):
    assert FLAGS.exp_name in CONFIG_MAPPING, 'Experiment folder not found.'
    config = CONFIG_MAPPING[FLAGS.exp_name]()
    env = config.get_environment(fake_env=True, save_video=False, classifier=False)

    devices = jax.local_devices()
    sharding = jax.sharding.PositionalSharding(devices)

    # ------------------ 1. 加载并切分数据集 ------------------
    success_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*success*.pkl"))
    failure_paths = glob.glob(os.path.join(os.getcwd(), "classifier_data", "*failure*.pkl"))

    pos_train, pos_val = load_and_split_data(success_paths, label=1, env=env, val_ratio=FLAGS.val_ratio)
    neg_train, neg_val = load_and_split_data(failure_paths, label=0, env=env, val_ratio=FLAGS.val_ratio)

    print(f"Positives - Train: {len(pos_train)}, Val: {len(pos_val)}")
    print(f"Negatives - Train: {len(neg_train)}, Val: {len(neg_val)}")

    # ------------------ 2. 创建 Train 与 Val Buffer ------------------
    # Train Buffers
    pos_train_buffer = ReplayBuffer(env.observation_space, env.action_space, capacity=20000, include_label=True)
    neg_train_buffer = ReplayBuffer(env.observation_space, env.action_space, capacity=50000, include_label=True)
    populate_buffer(pos_train_buffer, pos_train)
    populate_buffer(neg_train_buffer, neg_train)

    # Val Buffers
    pos_val_buffer = ReplayBuffer(env.observation_space, env.action_space, capacity=20000, include_label=True)
    neg_val_buffer = ReplayBuffer(env.observation_space, env.action_space, capacity=50000, include_label=True)
    populate_buffer(pos_val_buffer, pos_val)
    populate_buffer(neg_val_buffer, neg_val)

    # Iterators
    batch_half = FLAGS.batch_size // 2
    pos_train_iter = pos_train_buffer.get_iterator(sample_args={"batch_size": batch_half}, device=sharding.replicate())
    neg_train_iter = neg_train_buffer.get_iterator(sample_args={"batch_size": batch_half}, device=sharding.replicate())

    # 验证集使用全量数据（按 Batch 采样，取最接近训练总批次的大小或标准 Batch）
    val_batch_size = min(len(pos_val), len(neg_val), batch_half)
    pos_val_iter = pos_val_buffer.get_iterator(sample_args={"batch_size": val_batch_size}, device=sharding.replicate())
    neg_val_iter = neg_val_buffer.get_iterator(sample_args={"batch_size": val_batch_size}, device=sharding.replicate())

    # ------------------ 3. 模型初始化与增强 ------------------
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    sample = concat_batches(next(pos_train_iter), next(neg_train_iter), axis=0)

    classifier = create_classifier(key, sample["observations"], config.classifier_keys)

    def data_augmentation_fn(rng, observations):
        for pixel_key in config.classifier_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    # ------------------ 4. 训练与评估 Step 定义 ------------------
    @jax.jit
    def train_step(state, batch, key):
        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params}, batch["observations"], rngs={"dropout": key}, train=True
            )
            return optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        
        logits = state.apply_fn(
            {"params": state.params}, batch["observations"], train=False, rngs={"dropout": key}
        )
        preds = (nn.sigmoid(logits) >= 0.5)
        train_accuracy = jnp.mean(preds == batch["labels"])

        return state.apply_gradients(grads=grads), loss, train_accuracy

    @jax.jit
    def eval_step(state, batch):
        """验证集评估计算（不计算梯度，包含 Precision 和 Recall）"""
        logits = state.apply_fn(
            {"params": state.params}, batch["observations"], train=False
        )
        loss = optax.sigmoid_binary_cross_entropy(logits, batch["labels"]).mean()
        
        preds = (nn.sigmoid(logits) >= 0.5)
        labels = (batch["labels"] == 1)

        accuracy = jnp.mean(preds == labels)
        
        tp = jnp.sum((preds == True) & (labels == True))
        fp = jnp.sum((preds == True) & (labels == False))
        fn = jnp.sum((preds == False) & (labels == True))

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)

        return loss, accuracy, precision, recall

    # ------------------ 5. 主训练循环 ------------------
    best_val_loss = float('inf')

    for epoch in tqdm(range(FLAGS.num_epochs)):
        # --- Train Phase ---
        pos_sample = next(pos_train_iter)
        neg_sample = next(neg_train_iter)
        batch = concat_batches(pos_sample, neg_sample, axis=0)

        rng, key = jax.random.split(rng)
        obs = data_augmentation_fn(key, batch["observations"])
        batch = batch.copy(
            add_or_replace={
                "observations": obs,
                "labels": batch["labels"][..., None],
            }
        )

        rng, key = jax.random.split(rng)
        classifier, train_loss, train_accuracy = train_step(classifier, batch, key)

        # --- Validation Phase ---
        pos_v_sample = next(pos_val_iter)
        neg_v_sample = next(neg_val_iter)
        val_batch = concat_batches(pos_v_sample, neg_v_sample, axis=0)
        val_batch = val_batch.copy(
            add_or_replace={"labels": val_batch["labels"][..., None]}
        )

        val_loss, val_accuracy, val_precision, val_recall = eval_step(classifier, val_batch)

        print(
            f"\nEpoch: {epoch+1:03d} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, "
            f"Val Prec: {val_precision:.4f}, Val Rec: {val_recall:.4f}"
        )

    # ------------------ 6. 保存 Checkpoint ------------------
    checkpoints.save_checkpoint(
        os.path.join(os.getcwd(), "classifier_ckpt/"),
        classifier,
        step=FLAGS.num_epochs,
        overwrite=True,
    )


if __name__ == "__main__":
    app.run(main)