# Inspired by https://medium.com/@lou1swang/lets-reproduce-nanogpt-with-jax-part-1-95bec4630eb4
import time
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
import optax
import tiktoken
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from flax.training.train_state import TrainState


@dataclass
class ModelConfig:
    block_size: int = 1024  # Max sequence length
    vocab_size: int = (
        50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding dimension


class CausalSelfAttention(nn.Module):

    config: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic=True):

        assert len(x.shape) == 3

        b, l, d = x.shape

        q = nn.Dense(
            self.config.n_embd, kernel_init=nn.initializers.normal(stddev=0.02)
        )(x)
        k = nn.Dense(
            self.config.n_embd, kernel_init=nn.initializers.normal(stddev=0.02)
        )(x)
        v = nn.Dense(
            self.config.n_embd, kernel_init=nn.initializers.normal(stddev=0.02)
        )(x)
        # q*k / sqrt(dim) -> softmax -> @v
        q = jnp.reshape(q, (b, l, d // self.config.n_head, self.config.n_head))
        k = jnp.reshape(k, (b, l, d // self.config.n_head, self.config.n_head))
        v = jnp.reshape(v, (b, l, d // self.config.n_head, self.config.n_head))
        norm = jnp.sqrt(list(jnp.shape(k))[-1])
        attn = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) / norm
        mask = jnp.tril(attn)
        attn = jnp.where(mask[:, :, :l, :l], attn, float("-inf"))
        probs = jax.nn.softmax(attn, axis=-1)
        y = jnp.matmul(probs, v)
        y = jnp.reshape(y, (b, l, d))
        y = nn.Dense(
            self.config.n_embd,
            kernel_init=nn.initializers.normal(
                stddev=0.02 * (2 * self.config.n_layer) ** -0.05
            ),
        )(y)
        return y


class MLP(nn.Module):

    config: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic=True):
        x = nn.Dense(
            self.config.n_embd * 4, kernel_init=nn.initializers.normal(stddev=0.02)
        )(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dense(
            self.config.n_embd,
            kernel_init=nn.initializers.normal(
                stddev=0.02 * (2 * self.config.n_layer) ** -0.05
            ),
        )(x)
        return x


class Block(nn.Module):

    config: ModelConfig

    @nn.compact
    def __call__(self, x):
        x = x + CausalSelfAttention(self.config)(nn.LayerNorm()(x))
        x = x + MLP(self.config)(nn.LayerNorm()(x))
        return x


class GPT(nn.Module):

    config: ModelConfig

    @nn.compact
    def __call__(self, x, deterministic=False):

        B, T = x.shape
        assert T <= self.config.block_size

        pos = jnp.arange(0, T)[None]
        pos_emb = nn.Embed(self.config.block_size, self.config.n_embd)(pos)
        wte = nn.Embed(self.config.vocab_size, self.config.n_embd)
        tok_emb = wte(x)
        x = tok_emb + pos_emb

        for _ in range(self.config.n_layer):
            x = Block(self.config)(x)
        x = nn.LayerNorm()(x)
        # logits = nn.Dense(config.n_embd, config.vocab_size)(x)
        logits = wte.attend(x)  # parameter sharing
        return logits

    def init(self, rng):
        tokens = jnp.zeros((1, self.config.block_size), dtype=jnp.uint16)
        params = jax.jit(super().init, static_argnums=(2,))(rng, tokens, True)
        return params


def count_params(params):
    p = jax.tree_util.tree_map(
        lambda x: x.size if isinstance(x, jnp.ndarray) else 0, params
    )
    return jax.tree_util.tree_reduce(lambda x, y: x + y, p)


config = ModelConfig()
key = jax.random.PRNGKey(0)
model = GPT(config)
params = model.init(key)
print(f"Number of parameters: {count_params(params):,}")


class DataLoader:
    def __init__(self, B, T):
        self.current_position = 0
        self.B = B
        self.T = T

        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        self.tokens = jnp.array(enc.encode(text))
        print(f"loaded {len(self.tokens)} tokens in the datasets")
        print(f" 1 epoch = {len(self.tokens)//(B*T)} batches")

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x, y = jnp.reshape(buf[:-1], (B, T)), jnp.reshape(buf[1:], (B, T))
        self.current_position += B * T
        if self.current_position + B * T + 1 > len(self.tokens):
            self.current_position = 0
        return x, y


def init_train_state(key, config) -> TrainState:
    model = GPT(config)
    params = model.init(key)
    optimizer = optax.adamw(3e-4, b1=0.9, b2=0.98, eps=1e-9, weight_decay=1e-1)
    train_state = TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    return train_state


@jax.jit
def train_step(
    state: TrainState, x: jnp.ndarray, y: jnp.ndarray
) -> Tuple[jnp.ndarray, TrainState]:

    def loss_fn(params: FrozenDict) -> jnp.ndarray:

        logits = state.apply_fn(params, x, False)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return loss

    loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(state.params)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state


# ------------------------------------------------------------
# simple launch:
# python train_gpt2_jax.py

train_steps = 50
data_loader = DataLoader(B=32, T=1024)
train_state = init_train_state(key, config)
x, y = data_loader.next_batch()
for step in range(train_steps):
    t0 = time.time()
    loss, train_state = train_step(train_state, x, y)
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = data_loader.B * data_loader.T
    tokens_per_second = tokens_processed / dt
    print(
        f"Step {step:4d}/{train_steps} | loss: {loss:.6f} | time: {dt*1000:.2f}ms | tok/sec: {tokens_per_second:.2f}"
    )
