import inspect
import math
import os
import time
from dataclasses import dataclass

import tiktoken
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed import destroy_process_group, init_process_group
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, set_seed


@dataclass
class GPTConfig:
    block_size: int = 1024  # Max sequence length
    vocab_size: int = (
        50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    )
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of attention heads
    n_embd: int = 768  # embedding dimension


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) is the embedding dimension
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(
            1, 2
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # att = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02  # roughly equivalent to 1/sqrt(d_model)
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.05
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=std,
            )
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight,
                mean=0.0,
                std=0.02,  # roughly equivalent to 1/sqrt(d_model)
            )

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensor of integers token indices
        B, T = idx.size()
        assert (
            T <= self.config.block_size
        ), f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embed)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embed)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layer norm and the classifier
        x = self.transformer.ln_f(x)  # (B, T, n_embed)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type: str):
        """Loads pretrained GPT-2 model and weights from HuggingFace."""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}

        print(f"Loading weights from pretrained GPT-2 model: {model_type}")

        # n_layer, n_head, n_embd are determined by the model type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args["vocab_size"] = 50257  # always 50,257 for GPT-2 model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT-2 model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")]

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")]

        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a Conv1D module, but we use a Linear module, so we need to transpose
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatch: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert (
                    sd_hf[k].shape[::-1] == sd[k].shape
                ), f"Error: {k} shape mismatch: {sd_hf[k].shape[::-1]} != {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert (
                    sd_hf[k].shape == sd[k].shape
                ), f"Error: {k} shape mismatch: {sd_hf[k].shape} != {sd[k].shape}"
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameter that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for p in param_dict.values() if p.dim() >= 2]
        nodecay_params = [p for p in param_dict.values() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available and on an CUDA device
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, fused=use_fused, betas=(0.9, 0.95), eps=1e-8
        )
        return optimizer


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        # at init load tokens from disk and store them in memory
        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        print(f"loaded {len(tokens)} tokens")
        print(f"1 epoch = {len(tokens) // (B * T)} batches")

        # state
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)  # inputs
        y = buf[1:].view(B, T)  # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, reset the position
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x, y


# ------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=8 train_gpt2.py


NUM_RETURN_SEQUENCES = 5
MAX_LENGTH = 30

# run the training loop

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{ddp_local_rank}")
    torch.cuda.set_device(device)
    master_process = (
        ddp_rank == 0
    )  # only the first process will do logging, checkpointing etc.
else:
    # vanilla non-distributed training
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Using device: {device}")

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
torch.mps.manual_seed(1337)
set_seed(1337)

total_batch_size = 524_288  # 2**19, ~0.5M in number of tokens
B = 64  # micro batch size
T = 1024  # sequence length
assert (
    total_batch_size % (B * T * ddp_world_size) == 0
), "make sure total_batch_size is divisible by (B * T * ddp_world_size)"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size:,}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps:,}")

train_loader = DataLoaderLite(
    B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size
)

torch.set_float32_matmul_precision("high")

# create model
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)
if ddp:
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[ddp_local_rank]
    )
raw_model = model.module if ddp else model  # always contains the "raw" unwrapped model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50


def get_lr(it):
    # 1) linear warmup for warmup_steps
    if it < warmup_steps:
        return max_lr * (it + 1) / warmup_steps
    # 2) if it > lr_decay_steps, return min_lr
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (
        1.0 + math.cos(math.pi * decay_ratio)
    )  # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1, learning_rate=6e-4, device_type=device.type
)
for step in range(50):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right.
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = micro_step == grad_accum_steps - 1
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # detemine the learning rate for this step
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000  # ms
    tokens_processed = (train_loader.B * train_loader.T) * grad_accum_steps
    tokens_per_second = tokens_processed / (t1 - t0)
    if master_process:
        print(
            f"Step {step:4d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | time: {dt:.2f}ms | tok/sec: {tokens_per_second:.2f}"
        )

if ddp:
    destroy_process_group()

# model.eval()
# enc = tiktoken.get_encoding("gpt2")
# token = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(token, dtype=torch.long)  # (8,)
# tokens = tokens.unsqueeze(0).repeat(NUM_RETURN_SEQUENCES, 1)  # (5, 8)
# x = tokens.to(device)

# while x.size(1) < MAX_LENGTH:
#     # forward the model to get the logits for the next token

#     with torch.no_grad():
#         logits = model(x)
#         # take the logits at the last position
#         logits = logits[:, -1, :]
#         # get the probabilities
#         probs = F.softmax(logits, dim=-1)
#         # do top-k sampling of 50 (huggingface pipeline default)
#         # topk_probs here becomes (5, 50), topk_indices becomes (5, 50)
#         topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)

#         # select a token from the top-k probabilities
#         ix = torch.multinomial(topk_probs, 1)

#         # gather the indices of the top 50 tokens
#         xcol = torch.gather(topk_indices, -1, ix)

#         # append the sampled indices to the running sequence
#         x = torch.cat((x, xcol), dim=1)

# for i in range(NUM_RETURN_SEQUENCES):
#     token = x[i, :MAX_LENGTH].tolist()
#     decoded = enc.decode(token)
#     print(">", decoded)
