"""
Mixture of Experts (MoE) on CINIC-10 with 6 visual experts

This script extends the CIFAR-10 MoE demo to support CINIC-10 (270k images),
adds a 6-expert routing/mapping aligned to visual similarity, and generates
specialization metrics and visualizations suitable for an explainer video.

Highlights
- Dataset switch: --dataset cinic10 (default) or cifar10 (fallback)
- Visual grouping into 6 experts (fur small/large, bird, frog, sleek metal, boxy industrial)
- Vectorized guidance targets (no tf.py_function), load-balancing, routing metrics
- Artifacts: CSVs (competence/affinity), PNGs (utilization, routing accuracy, entropy), model.h5
- Colab-friendly: works with !python and supports --subset_samples for quick runs

Author: Educational ML Demo
License: MIT
"""

import os
import sys
import math
import time
import json
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Optional: plotting (only used for saving figures)
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

# GPU memory growth
_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    try:
        for _g in _gpus:
            tf.config.experimental.set_memory_growth(_g, True)
    except Exception as e:
        print(f"[warn] GPU memory growth: {e}")

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
CLASS_NAME_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}
NUM_CLASSES = 10
IMG_SIZE = 32
CHANNELS = 3

# Default training (overridable by CLI)
DEFAULT_EPOCHS = 20
DEFAULT_BATCH = 64
INITIAL_LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
WARMUP_EPOCHS = 5
EMERGENCE_END_EPOCH = 15
WARMUP_GUIDE_WEIGHT = 1.0
WARMUP_LB_COEFF = 0.1
NORMAL_LB_COEFF = 0.01
ROUTER_NOISE_STD = 0.1

# ------------------------------
# Visual 6-expert mapping
# ------------------------------
# Expert 0: small domestic fur  -> cat, dog
# Expert 1: large wild fur      -> horse, deer
# Expert 2: feathered flier     -> bird
# Expert 3: smooth amphibian    -> frog
# Expert 4: sleek metal         -> airplane, automobile
# Expert 5: boxy industrial     -> truck, ship

VISUAL6_MAPPING = {
    CLASS_NAME_TO_ID['cat']: 0,
    CLASS_NAME_TO_ID['dog']: 0,
    CLASS_NAME_TO_ID['horse']: 1,
    CLASS_NAME_TO_ID['deer']: 1,
    CLASS_NAME_TO_ID['bird']: 2,
    CLASS_NAME_TO_ID['frog']: 3,
    CLASS_NAME_TO_ID['airplane']: 4,
    CLASS_NAME_TO_ID['automobile']: 4,
    CLASS_NAME_TO_ID['truck']: 5,
    CLASS_NAME_TO_ID['ship']: 5,
}

EXPERT_6_DOMAINS = {
    0: {'name': 'SmallFur', 'classes': [CLASS_NAME_TO_ID['cat'], CLASS_NAME_TO_ID['dog']]},
    1: {'name': 'LargeFur', 'classes': [CLASS_NAME_TO_ID['horse'], CLASS_NAME_TO_ID['deer']]},
    2: {'name': 'Feather', 'classes': [CLASS_NAME_TO_ID['bird']]},
    3: {'name': 'Amphib', 'classes': [CLASS_NAME_TO_ID['frog']]},
    4: {'name': 'SleekMetal', 'classes': [CLASS_NAME_TO_ID['airplane'], CLASS_NAME_TO_ID['automobile']]},
    5: {'name': 'BoxyInd', 'classes': [CLASS_NAME_TO_ID['truck'], CLASS_NAME_TO_ID['ship']]},
}


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# =============================================================================
# Dataset loaders
# =============================================================================

def _ds_to_arrays(ds: tf.data.Dataset, limit: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    count = 0
    for x, y in ds:
        xs.append(x.numpy())
        ys.append(y.numpy())
        count += x.shape[0]
        if limit is not None and count >= limit:
            break
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if limit is not None:
        X = X[:limit]
        y = y[:limit]
    return X, y


def load_cifar10(batch_size: int, subset_per_split: Optional[int] = None):
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    if subset_per_split is not None:
        x_train = x_train[:subset_per_split]
        y_train = y_train[:subset_per_split]
        x_test = x_test[:subset_per_split // 4]
        y_test = y_test[:subset_per_split // 4]

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000, seed=SEED).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = max(1, len(x_train) // batch_size)
    val_steps = max(1, len(x_test) // batch_size)
    return train_ds, val_ds, steps_per_epoch, val_steps, (x_test, y_test)


def load_cinic10(batch_size: int,
                 cinic10_dir: Optional[str] = None,
                 subset_per_split: Optional[int] = None):
    """Load CINIC-10 using TFDS if available or from directory.

    Directory format expected if cinic10_dir is provided:
      cinic10_dir/
        train/<class>/*.png
        valid/<class>/*.png
        test/<class>/*.png
    """
    x_val = y_val = None

    # Try TFDS first
    used_tfds = False
    try:
        import tensorflow_datasets as tfds  # type: ignore
        if 'cinic10' in tfds.list_builders():
            used_tfds = True
            splits = {
                'train': tfds.Split.TRAIN,
                'valid': tfds.Split.VALIDATION if hasattr(tfds.Split, 'VALIDATION') else 'validation',
                'test': tfds.Split.TEST,
            }
            def _prep(example):
                image = tf.image.convert_image_dtype(example['image'], tf.float32)
                image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
                label = tf.cast(example['label'], tf.int32)
                return image, label
            train_ds = tfds.load('cinic10', split=splits['train'], as_supervised=False)
            val_ds = tfds.load('cinic10', split=splits['test'], as_supervised=False)  # use test as val
            train_ds = train_ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
            val_ds = val_ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE)
            if subset_per_split is not None:
                train_ds = train_ds.take(subset_per_split)
                val_ds = val_ds.take(max(1, subset_per_split // 4))
            train_ds = train_ds.shuffle(10000, seed=SEED).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
            val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
            # Materialize a validation subset for metrics (capped)
            x_val, y_val = _ds_to_arrays(val_ds.unbatch().batch(1024), limit=min(10000, subset_per_split or 10000))
            steps_per_epoch = 100000 // batch_size if subset_per_split is None else max(1, subset_per_split // batch_size)
            val_steps = max(1, len(y_val) // batch_size)
            return train_ds, val_ds, steps_per_epoch, val_steps, (x_val, y_val)
    except Exception as e:
        print(f"[info] TFDS CINIC-10 not available ({e}); falling back to directory loader.")

    # Directory fallback
    if cinic10_dir is None or not os.path.isdir(cinic10_dir):
        raise RuntimeError("CINIC-10 directory not found. Provide --cinic10_dir pointing to cinic-10 dataset, or install TFDS with cinic10.")

    def _load_split(split_name: str):
        split_path = os.path.join(cinic10_dir, split_name)
        if not os.path.isdir(split_path):
            raise RuntimeError(f"Missing split folder: {split_path}")
        ds = tf.keras.utils.image_dataset_from_directory(
            split_path,
            labels='inferred',
            label_mode='int',
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=None,
            shuffle=True,
            seed=SEED)
        ds = ds.map(lambda x, y: (tf.image.convert_image_dtype(x, tf.float32), tf.cast(y, tf.int32)),
                    num_parallel_calls=tf.data.AUTOTUNE)
        return ds

    train_raw = _load_split('train')
    val_raw = _load_split('valid') if os.path.isdir(os.path.join(cinic10_dir, 'valid')) else _load_split('test')

    if subset_per_split is not None:
        train_raw = train_raw.take(subset_per_split)
        val_raw = val_raw.take(max(1, subset_per_split // 4))

    train_ds = train_raw.shuffle(20000, seed=SEED).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    val_ds = val_raw.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    x_val, y_val = _ds_to_arrays(val_raw.batch(1024), limit=min(10000, subset_per_split or 10000))
    steps_per_epoch = 100000 // batch_size if subset_per_split is None else max(1, subset_per_split // batch_size)
    val_steps = max(1, len(y_val) // batch_size)
    return train_ds, val_ds, steps_per_epoch, val_steps, (x_val, y_val)


# =============================================================================
# Model components
# =============================================================================

def build_shared_trunk() -> keras.Model:
    return keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
    ], name='trunk')


def build_router(num_experts: int) -> keras.Model:
    return keras.Sequential([
        layers.Input(shape=(128,)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_experts),
    ], name='router')


def build_expert(expert_id: int) -> keras.Model:
    return keras.Sequential([
        layers.Input(shape=(128,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(NUM_CLASSES),
    ], name=f'expert_{expert_id}')


def compute_load_balancing_loss(router_probs: tf.Tensor, expert_mask: tf.Tensor) -> tf.Tensor:
    num_experts = tf.cast(tf.shape(router_probs)[1], tf.float32)
    fraction_per_expert = tf.reduce_mean(expert_mask, axis=0)
    mean_prob_per_expert = tf.reduce_mean(router_probs, axis=0)
    lb_loss = num_experts * tf.reduce_sum(fraction_per_expert * mean_prob_per_expert)
    return lb_loss


def create_soft_targets(labels: tf.Tensor, mapping: Dict[int, int], num_experts: int, confidence: float = 0.9) -> tf.Tensor:
    class_to_expert = [0] * NUM_CLASSES
    for c in range(NUM_CLASSES):
        class_to_expert[c] = mapping.get(c, 0)
    class_to_expert = tf.constant(class_to_expert, dtype=tf.int32)
    labels = tf.cast(labels, tf.int32)
    correct_expert = tf.gather(class_to_expert, labels)
    other = (1.0 - confidence) / tf.cast(num_experts - 1, tf.float32)
    base = tf.fill([tf.shape(labels)[0], num_experts], other)
    hot = tf.one_hot(correct_expert, depth=num_experts, on_value=confidence - other, off_value=0.0)
    soft_targets = tf.cast(base, tf.float32) + tf.cast(hot, tf.float32)
    return soft_targets


class MixtureOfExpertsModel(keras.Model):
    def __init__(self, num_experts: int, mapping: Dict[int, int]):
        super().__init__()
        self.num_experts = num_experts
        self.mapping = mapping
        self.trunk = build_shared_trunk()
        self.router = build_router(num_experts)
        self.experts = [build_expert(i) for i in range(num_experts)]
        self.cls_loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.kld = keras.losses.KLDivergence()
        self.current_epoch = tf.Variable(0, trainable=False, dtype=tf.int32)
        self.guide_weight = tf.Variable(WARMUP_GUIDE_WEIGHT, trainable=False, dtype=tf.float32)
        self.lb_coeff = tf.Variable(WARMUP_LB_COEFF, trainable=False, dtype=tf.float32)
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.cls_loss_tracker = keras.metrics.Mean(name='cls_loss')
        self.lb_loss_tracker = keras.metrics.Mean(name='lb_loss')
        self.guide_loss_tracker = keras.metrics.Mean(name='guide_loss')
        self.acc_tracker = keras.metrics.SparseCategoricalAccuracy(name='accuracy')

    @property
    def metrics(self):
        return [self.loss_tracker, self.cls_loss_tracker, self.lb_loss_tracker, self.guide_loss_tracker, self.acc_tracker]

    def update_phase(self, epoch: int):
        self.current_epoch.assign(epoch)
        if epoch < WARMUP_EPOCHS:
            self.guide_weight.assign(WARMUP_GUIDE_WEIGHT)
            self.lb_coeff.assign(WARMUP_LB_COEFF)
        elif epoch < EMERGENCE_END_EPOCH:
            progress = (epoch - WARMUP_EPOCHS) / float(EMERGENCE_END_EPOCH - WARMUP_EPOCHS)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            self.guide_weight.assign(WARMUP_GUIDE_WEIGHT * cosine)
            self.lb_coeff.assign(NORMAL_LB_COEFF)
        else:
            self.guide_weight.assign(0.0)
            self.lb_coeff.assign(NORMAL_LB_COEFF)

    def call(self, inputs, training=False):
        feats = self.trunk(inputs, training=training)
        router_logits = self.router(feats, training=training)
        router_probs = tf.nn.softmax(router_logits, axis=-1)
        expert_logits = [e(feats, training=training) for e in self.experts]
        expert_stack = tf.stack(expert_logits, axis=1)  # [B, E, C]
        out = tf.reduce_sum(expert_stack * tf.expand_dims(router_probs, -1), axis=1)
        return out

    def train_step(self, data):
        x, y = data
        y = tf.cast(y, tf.int32)
        with tf.GradientTape() as tape:
            feats = self.trunk(x, training=True)
            router_logits = self.router(feats, training=True)
            noisy_router_logits = router_logits + tf.random.normal(tf.shape(router_logits), stddev=ROUTER_NOISE_STD)
            router_probs = tf.nn.softmax(noisy_router_logits, axis=-1)
            expert_indices = tf.argmax(router_probs, axis=-1, output_type=tf.int32)
            expert_mask = tf.one_hot(expert_indices, self.num_experts)
            expert_logits = [e(feats, training=True) for e in self.experts]
            expert_stack = tf.stack(expert_logits, axis=1)
            final_logits = tf.reduce_sum(expert_stack * tf.expand_dims(router_probs, -1), axis=1)
            cls_loss = self.cls_loss_fn(y, final_logits)
            lb_loss = compute_load_balancing_loss(router_probs, expert_mask)
            soft_targets = create_soft_targets(y, self.mapping, self.num_experts)
            guide_loss = self.kld(soft_targets, router_probs)
            total = cls_loss + (self.lb_coeff * lb_loss) + (self.guide_weight * guide_loss)
        vars = self.trainable_variables
        grads = tape.gradient(total, vars)
        grads, _ = tf.clip_by_global_norm(grads, GRAD_CLIP)
        self.optimizer.apply_gradients(zip(grads, vars))
        self.loss_tracker.update_state(total)
        self.cls_loss_tracker.update_state(cls_loss)
        self.lb_loss_tracker.update_state(lb_loss)
        self.guide_loss_tracker.update_state(guide_loss)
        self.acc_tracker.update_state(y, final_logits)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        logits = self(x, training=False)
        loss = self.cls_loss_fn(y, logits)
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(y, logits)
        return {m.name: m.result() for m in [self.loss_tracker, self.acc_tracker]}


# =============================================================================
# Dashboard Callback: metrics + artifacts
# =============================================================================

class SpecializationCallback(keras.callbacks.Callback):
    def __init__(self, val_array: Tuple[np.ndarray, np.ndarray], num_experts: int, mapping: Dict[int, int], out_dir: str, batch: int = 1024):
        super().__init__()
        self.x_val, self.y_val = val_array
        self.num_experts = num_experts
        self.mapping = mapping
        self.out_dir = out_dir
        self.batch = batch
        self.epoch_times = []
        self.utilization_history = []  # list of [E] per epoch
        self.routing_acc = []
        self.entropy = []
        ensure_dir(out_dir)

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()
        self.model.update_phase(epoch)

    def _batched_predict(self, model_or_fn, X):
        outs = []
        for i in range(0, len(X), self.batch):
            xb = X[i:i+self.batch]
            out = model_or_fn.predict(xb, verbose=0) if hasattr(model_or_fn, 'predict') else model_or_fn(xb, training=False)
            if isinstance(out, tf.Tensor):
                out = out.numpy()
            outs.append(out)
        return np.concatenate(outs, axis=0)

    def on_epoch_end(self, epoch, logs=None):
        t = time.time() - self._t0
        # Features and router
        feats = self._batched_predict(self.model.trunk, self.x_val)
        router_logits = self._batched_predict(self.model.router, feats)
        router_probs = tf.nn.softmax(router_logits, axis=-1).numpy()
        expert_choices = np.argmax(router_probs, axis=-1)

        # Utilization
        util = [(expert_choices == e).mean() for e in range(self.num_experts)]
        self.utilization_history.append(util)

        # Routing accuracy vs mapping
        class_to_expert = np.array([self.mapping.get(c, 0) for c in range(NUM_CLASSES)])
        correct_e = class_to_expert[self.y_val]
        r_acc = (expert_choices == correct_e).mean()
        self.routing_acc.append(r_acc)

        # Entropy
        eps = 1e-10
        ent = (-router_probs * np.log(router_probs + eps)).sum(axis=-1).mean()
        self.entropy.append(ent)

        # Progress print
        lr = self.model.optimizer.learning_rate
        lr_val = lr(self.model.optimizer.iterations).numpy() if isinstance(lr, keras.optimizers.schedules.LearningRateSchedule) else float(lr.numpy())
        print("\n" + "="*80)
        print(f"EPOCH {epoch+1} | Time: {t:.1f}s | LR: {lr_val:.6f} | Routing Acc: {100*r_acc:4.1f}% | Entropy: {ent:.3f}")
        print("Utilization:", [f"E{e}:{u*100:4.1f}%" for e, u in enumerate(util)])
        print("="*80)

        # Save training log line
        with open(os.path.join(self.out_dir, 'training_log.txt'), 'a') as f:
            f.write(json.dumps({
                'epoch': epoch+1,
                'time_sec': t,
                'lr': float(lr_val),
                'routing_acc': float(r_acc),
                'entropy': float(ent),
                'utilization': [float(x) for x in util],
                'metrics': {k: float(v) for k, v in (logs or {}).items()}
            }) + "\n")

        # At last epoch, compute matrices and save artifacts
        if epoch + 1 == self.params.get('epochs', epoch+1):
            self._save_artifacts(router_probs)

    def _save_artifacts(self, router_probs: np.ndarray):
        # Expert competence matrix
        comp = np.zeros((NUM_CLASSES, self.num_experts), dtype=np.float32)
        # Affinity matrix from router probs
        affinity = np.zeros_like(comp)

        # Precompute feats per class
        for c in range(NUM_CLASSES):
            mask = self.y_val == c
            if not np.any(mask):
                continue
            Xc = self.x_val[mask]
            feats_c = self._batched_predict(self.model.trunk, Xc)
            for e in range(self.num_experts):
                logits_e = self._batched_predict(self.model.experts[e], feats_c)
                preds = logits_e.argmax(axis=-1)
                comp[c, e] = (preds == c).mean() * 100.0
            affinity[c] = router_probs[mask].mean(axis=0)

        # Save CSVs
        np.savetxt(os.path.join(self.out_dir, 'expert_competence_matrix.csv'), comp, delimiter=',', fmt='%.3f')
        np.savetxt(os.path.join(self.out_dir, 'router_affinity_matrix.csv'), affinity, delimiter=',', fmt='%.6f')

        # Plots: utilization over epochs
        util_arr = np.array(self.utilization_history)  # [epochs, E]
        plt.figure(figsize=(8,4))
        for e in range(self.num_experts):
            plt.plot(util_arr[:, e]*100, label=f'Expert {e}')
        plt.xlabel('Epoch'); plt.ylabel('Utilization %'); plt.title('Expert Utilization Over Time'); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'expert_utilization.png'))
        plt.close()

        # Routing accuracy over epochs
        plt.figure(figsize=(6,4))
        plt.plot(np.array(self.routing_acc)*100)
        plt.xlabel('Epoch'); plt.ylabel('Routing Accuracy %'); plt.title('Routing Accuracy Over Time'); plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'routing_accuracy.png'))
        plt.close()

        # Entropy over epochs
        plt.figure(figsize=(6,4))
        plt.plot(self.entropy)
        plt.xlabel('Epoch'); plt.ylabel('Entropy'); plt.title('Router Entropy Over Time'); plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, 'router_entropy.png'))
        plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cinic10', 'cifar10'], default='cinic10', help='Which dataset to use')
    parser.add_argument('--cinic10_dir', type=str, default=None, help='Path to CINIC-10 folder (train/valid/test) if not using TFDS')
    parser.add_argument('--num_experts', type=int, default=6, help='Number of experts (default 6 for visual grouping)')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH)
    parser.add_argument('--subset_samples', type=int, default=None, help='Limit samples per split for quick runs (e.g., 50000)')
    parser.add_argument('--metrics_subset', type=int, default=10000, help='Cap number of validation samples used for metrics')
    parser.add_argument('--output_dir', type=str, default='moe_outputs', help='Where to save artifacts')
    args = parser.parse_args()

    out_dir = args.output_dir
    ensure_dir(out_dir)

    # Mapping (currently fixed to visual-6; could be extended via flag)
    mapping = VISUAL6_MAPPING

    print("\n" + "="*80)
    print(" "*20 + "MoE CINIC-10 Educational Demo")
    print(" "*8 + f"6-Expert Visual Specialization | Dataset: {args.dataset}")
    print("="*80)

    # Load data
    if args.dataset == 'cinic10':
        train_ds, val_ds, steps_per_epoch, val_steps, (x_val, y_val) = load_cinic10(
            batch_size=args.batch_size,
            cinic10_dir=args.cinic10_dir,
            subset_per_split=args.subset_samples,
        )
    else:
        train_ds, val_ds, steps_per_epoch, val_steps, (x_val, y_val) = load_cifar10(
            batch_size=args.batch_size,
            subset_per_split=args.subset_samples,
        )
        # cap validation arrays for metrics
        if len(y_val) > args.metrics_subset:
            x_val = x_val[:args.metrics_subset]
            y_val = y_val[:args.metrics_subset]

    # Build model
    model = MixtureOfExpertsModel(num_experts=args.num_experts, mapping=mapping)

    # LR schedule with warmup
    total_steps = args.epochs * steps_per_epoch
    warm_steps = max(1, min(3*steps_per_epoch, total_steps-1))
    lr_sched = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=INITIAL_LR,
        decay_steps=max(1, total_steps - warm_steps),
        alpha=0.1,
    )
    opt = keras.optimizers.AdamW(learning_rate=lr_sched, weight_decay=WEIGHT_DECAY)
    model.compile(optimizer=opt)

    # Callback
    dash = SpecializationCallback((x_val, y_val), num_experts=args.num_experts, mapping=mapping, out_dir=out_dir)

    # Train
    model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=[dash],
        verbose=2,
    )

    # Save model
    model.save(os.path.join(out_dir, 'model_final.h5'))

    # Final evaluation on validation set
    res = model.evaluate(val_ds, steps=val_steps, verbose=0)
    print(f"\nFinal Metrics -> loss: {res[0]:.4f}, accuracy: {res[-1]*100:.2f}%")

    # Print locations
    print(f"Artifacts saved under: {os.path.abspath(out_dir)}")
    print(" - expert_competence_matrix.csv, router_affinity_matrix.csv")
    print(" - expert_utilization.png, routing_accuracy.png, router_entropy.png")
    print(" - training_log.txt, model_final.h5")


if __name__ == '__main__':
    main()
