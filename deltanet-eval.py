import numpy as np
import os, pickle
import math
import tqdm
import jax, jax.numpy as jp
import torch

data_dir = os.path.join('data', 'shakespeare_char')


def get_batch(split):
    fname = 'train.bin' if split == 'train' else 'val.bin'
    data = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode='r')
    ix = np.random.randint(len(data) - block_size, size=(batch_size,))
    x = jp.stack([(data[i:i+block_size]).astype(np.int64) for i in ix])
    y = jp.stack([(data[i+1:i+1+block_size]).astype(np.int64) for i in ix])
    return x, y  # token, next token


meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

key = jax.random.PRNGKey(42)
block_size = 256  # context length
batch_size = 32
n_layer, n_head, n_embd, vocab_size = 8, 12, 384, meta_vocab_size
nh, hs = n_head, n_embd//n_head
beta = 0.01

with open('/tmp/deltanet-params-8100.pkl', 'rb') as fd:
    Wi, Wo, lm_head, wte, wpe, Beta = p0 = pickle.load(fd)

x, y = get_batch('train')
B, T, C = batch_size, block_size, n_embd
pos = jp.r_[:T]
s0 = jp.zeros((B, nh, hs, hs))


def attn_op(s, qli):
    "core op in deltanet style attention"
    q, l, i = qli
    s = jp.einsum('bhij,bhjk->bhik', s, l) + i
    o = jp.einsum('bhj,bhij->bhi', q, s)
    return s, o


def fwd(params, x, phi=jax.nn.sigmoid):
    Wi, Wo, lm_head, wte, wpe, Beta = params
    x = wte[x]  # + wpe  # B,T,C
    def z(x): return (x - x.mean(axis=-1)
                      [..., None])/x.std(ddof=1, axis=-1)[..., None]
    for wi, wo, (b1, b2) in zip(Wi, Wo, Beta):
        q, k, v = z(jp.einsum('ij,bti->btj', wi, x).reshape(B,
                    T, 3, nh, hs)).swapaxes(0, 2)
        q, k = phi(k), phi(v)
        L = b1.reshape(nh, hs, 1) - jp.einsum('tbhi,tbhj->tbhij', k, k*b2)
        I = jp.einsum('tbhi,tbhj->tbhij', k*b2, v)
        _, jvt = jax.lax.scan(attn_op, s0, (q, L, I))
        x = jp.swapaxes(jvt, 0, 1).reshape(B, T, C) @ wo + x
    return z(x) @ lm_head


def loss(W, x, y):
    logits = fwd(W, x)
    yoh = jax.nn.one_hot(y, logits.shape[-1])
    ll = -(jax.nn.log_softmax(logits, axis=-1) * yoh).sum(axis=-1)
    return ll.mean()


jvg = jax.jit(jax.value_and_grad(loss))
jv = jax.jit(loss)

p0 = Wi, Wo, lm_head, wte, wpe, Beta
for i in range(10):
    x, y = get_batch('test')
    initv = jv(p0, x, y)
    print('loss', initv)
