import numpy as np, os, pickle, tqdm, jax, jax.numpy as jp

# setup
key = jax.random.PRNGKey(42)
batch_size = 8
n_layer, n_head, n_embd, vocab_size = 4, 3, 3*32, 65
nh, hs = n_head, n_embd//n_head
B, C = batch_size, n_embd

# params
wte = jax.random.normal(key, shape=(vocab_size, n_embd)) * 0.0002
Wi = jax.random.normal(key, shape=(n_layer, n_embd, 3*n_embd)) * 0.0001
Wo = jax.random.normal(key, shape=(n_layer, n_embd, n_embd)) * 0.0001
lm_head = jax.random.normal(key, shape=(n_embd, vocab_size)) * 0.0001
p0 = Wi, Wo, lm_head, wte
param_counts = sum(jax.tree_util.tree_map(lambda p: p.size, p0))
print(f'{param_counts/1e6:0.2f} M params')

Z = lambda x: (x - x.mean(axis=-1)[...,None])/x.std(ddof=1, axis=-1)[...,None]

def model2_layer(s, wi, wo, xt, phi=jax.nn.gelu):
    q, k, v = Z(jp.einsum('ij,bi->bj', wi, xt).reshape(B,3,nh,hs)).swapaxes(0,1)
    s = s + jp.einsum('bhi,bhj->bhij', phi(k), v)
    vt = jp.einsum('bhi,bhij->bhj', phi(q), s)
    xt = vt.reshape(B,C) @ wo + xt
    return s, xt

def model(params, x, S, outputs):
    Wi, Wo, lm_head, wte = params
    x = wte[x].swapaxes(0, 1)
    assert x.shape == (T, B, C)
    # swap layer & token loops, but otherwise the same for now
    # if 
    S = jp.zeros((nl, B, nh, hs, hs))
    outputs = jp.zeros((nl, B, C)) + 1e-3
    def op(c, x):
        s, o = c
        i = jp.concat([x.reshape(1, B, C), o[:-1]])
        s, o = jax.vmap(model2_layer)(s, Wi, Wo, i)
        return (s, o), o[-1]
    x_ = jp.pad(x, [(0,nl-1),(0,0),(0,0)], constant_values=1e-4)
    (s,o), x_ = jax.lax.scan(op, (S,outputs), x_)
    x = x_[nl-1:]
    return Z(x.swapaxes(0, 1)) @ lm_head

def loss(W, x, y):
    logits = model(W, x)
    yoh = jax.nn.one_hot(y, logits.shape[-1])
    ll = -(jax.nn.log_softmax(logits, axis=-1) * yoh).sum(axis=-1)
    return ll.mean()

from jax.example_libraries.optimizers import adam
oinit, oup, oget = adam(1e-4)
o = oinit(p0)
jvg = jax.value_and_grad(loss)
jvg = jax.jit(jvg)
for lr in [1e-4, 5e-5]:
    _, oup, _ = adam(lr)
    for i in range(2001):
        v, g = jvg(oget(o), *get_batch('train'))
        o = oup(i, g, o)
        if i % 20 == 0:
            print(i, v)
