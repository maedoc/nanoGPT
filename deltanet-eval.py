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
block_size = 128 # context length
batch_size = 32
n_layer, n_head, n_embd, vocab_size = 8, 6, 192, meta_vocab_size
nh, hs = n_head, n_embd//n_head
beta = 0.01

with open('/tmp/deltanet-params-5000.pkl', 'rb') as fd:
    Wi, Wo, lm_head, wte, wpe, Wb = p0 = pickle.load(fd)

x, y = get_batch('train')
B, T, C = batch_size, block_size, n_embd
pos = jp.r_[:T]
s0 = jp.zeros((B, nh, hs, hs))


# XXX copy from deltanet.py
def attn_op1h(s, qli):
    "core op in deltanet style attention"
    q, l, i = qli
    s = jp.einsum('bij,bjk->bik', s, l) + i
    o = jp.einsum('bij,bj->bi', s, q)
    return s, o

def attn1h(q,k,v,b1,b2,s0):
    L = b1.reshape(hs,1) - jp.einsum('tbi,tbj->tbij', k, k*b2)
    I = jp.einsum('tbi,tbj->tbij', v, k*b2)
    _, jvt = jax.lax.scan(attn_op1h, s0, (q,L,I))
    return jvt

def Jattn1h(q,k,v,b1,b2,s0):
    return jax.jacobian(
        lambda v_: attn1h(q, k, v_, b1, b2, s0)
    )(v)

def fwd(params, x, phi=jax.nn.sigmoid):
    Wi, Wo, lm_head, wte, wpe, Beta = params
    x = wte[x] #+ wpe  # B,T,C
    z = lambda x: (x - x.mean(axis=-1)[...,None])/x.std(ddof=1, axis=-1)[...,None]
    for wi, wo, (b1,b2) in zip(Wi, Wo, Beta):
        q, k, v = z(jp.einsum('ij,bti->btj', wi, x).reshape(B,T,3,nh,hs)).swapaxes(0,2)
        q, k = phi(q), phi(k)
        ax = -2,-2,-2,-2,-2,-3
        jvt = jax.vmap(attn1h, in_axes=ax, out_axes=-2)(q,k,v,b1,b2,s0)
        # Jv = jax.jacobian(lambda v: attn(q,k,v,b1,b2))(v)
        J1 = Jattn1h(q,k,v,b1,b2,s0)
        import pdb; pdb.set_trace()
        x = jp.swapaxes(jvt,0,1).reshape(B,T,C) @ wo + x
    return z(x) @ lm_head


def loss(W, x, y):
    logits = fwd(W, x)
    yoh = jax.nn.one_hot(y, logits.shape[-1])
    ll = -(jax.nn.log_softmax(logits, axis=-1) * yoh).sum(axis=-1)
    return ll.mean()


jvg = jax.jit(jax.value_and_grad(loss))
jv = jax.jit(loss)

p0 = Wi, Wo, lm_head, wte, wpe, Wb
for i in range(10):
    x, y = get_batch('test')
    initv = jv(p0, x, y)
    print('loss', initv)
