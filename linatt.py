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
n_layer, n_head, n_embd, vocab_size = 8, 12, 384, meta_vocab_size
nh, hs = n_head, n_embd//n_head

wte = jax.random.normal(key, shape=(vocab_size, n_embd)) * 0.0002
# XXX consider something more flexible
wpe = jax.random.normal(key, shape=(block_size, n_embd)) * 0.0002
Wi = jax.random.normal(key, shape=(n_layer, n_embd, 3*n_embd)) * 0.0001
Wo = jax.random.normal(key, shape=(n_layer, n_embd, n_embd)) * 0.0001
lm_head = jax.random.normal(key, shape=(n_embd, vocab_size)) * 0.0001
x, y = get_batch('train')
B, T, C = batch_size, block_size, n_embd
pos = jp.r_[:T]
mask = jp.tril(jp.ones((T,T)))


def fwd(params, x, phi=jax.nn.leaky_relu):
    Wi, Wo, lm_head, wte, wpe = params
    x = wte[x] #+ wpe  # B,T,C
    z = lambda x: (x - x.mean(axis=-1)[...,None])/x.std(ddof=1, axis=-1)[...,None]
    for wi, wo in zip(Wi, Wo):
        q, k, v = z(jp.einsum('ij,bti->btj', wi, x).reshape(B,T,3,nh,hs)).swapaxes(0,2)
        q, k = phi(q), phi(k)  # 2tbhi
        # T is key-value index, t query-output index
        A = jp.einsum('Tbhi,tbhi->tTbh', k, q)
        # A = A / jp.einsum('tTbh->tbh', A).reshape(T, 1, B, nh)
        vt = jp.einsum('tTbh,tT,Tbhi->tbhi', A, mask, v)
        x = jp.swapaxes(vt,0,1).reshape(B,T,C) @ wo + x
    return z(x) @ lm_head


def loss(W, x, y):
    logits = fwd(W, x)
    yoh = jax.nn.one_hot(y, logits.shape[-1])
    ll = -(jax.nn.log_softmax(logits, axis=-1) * yoh).sum(axis=-1)
    return ll.mean()

jit = jax.jit
# jit = lambda f: f
jvg = jit(jax.value_and_grad(loss))
jv = jit(loss)

p0 = Wi, Wo, lm_head, wte, wpe
initv = jv(p0, x, y)
assert jp.isfinite(initv)
print('initial loss', initv)

param_counts = sum(jax.tree_util.tree_map(lambda p: p.size, p0))
print('param count', f'{param_counts/1e6:0.2f} M params')


warmup_iters = 100 # how many steps to warm up for
lr_decay_iters = 15000 # should be ~= max_iters per Chinchilla
learning_rate = 5e-4   # 1e-3 # max learning rate
min_lr = learning_rate/20 # learning_rate / 10 usually

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def weight_decay(o, lr, d=0.1):
    x, m, v = o
    x = jax.tree_util.tree_map(lambda x: x - lr*d*x, x)
    return x, m, v

grad_acc_steps = 1  # 5 * 8

def get_acc_batch():
    return jp.array([get_batch('train') for _ in range(grad_acc_steps)])

def gacc(gw, xy):
    g, w = gw
    g_ = jax.grad(loss)(w, *xy)    
    g = jax.tree_util.tree_map(lambda g1,g2: g1+g2, g, g_)
    return (g,w), 0

from jax.example_libraries.optimizers import adam

oinit, oup, oget = adam(1e-6)
o = oinit(p0)

@jax.jit
def one_step(params, bb):
    g = jax.grad(loss)(params, *bb[0])
    if grad_acc_steps > 1:
        (g,_), _ = jax.lax.scan(gacc, (g,params), bb[1:])
    g = jax.tree_util.tree_map(lambda x: jp.where(jp.isfinite(x), x, 0), g)
    g = jax.tree_util.tree_map(lambda x: jp.clip(x/grad_acc_steps, -1, 1), g)
    return g

bb = get_acc_batch()
g = one_step(oget(o), bb)

jv = jax.jit(loss)

trace = []
bestv = initv, 0
for i in range(lr_decay_iters + 1):
    lr = get_lr(i)
    _, oup, _ = adam(lr)
    g = one_step(oget(o), get_acc_batch())
    o = oup(i, g, o)
    o = weight_decay(o, lr)
    if i % 100 == 0:
        v = jv(oget(o), *get_batch('test'))
        if v < bestv[0]:
            bestv = v, i
        print(f'iter {i}: loss {v:0.4f}')
        with open(f'/tmp/deltanet-params-{i}.pkl', 'wb') as fd:
            pickle.dump(oget(o), fd)

print('best loss', bestv)
