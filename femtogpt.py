import numpy as np, os, pickle, tqdm, jax, jax.numpy as jp
from sequence_generator import generate_batch

# data
def get_batch(split):
    x = generate_batch(batch_size, block_size + 1)
    y = x[:, 1:]  # Next token targets by slicing
    x = x[:, :-1]  # Remove the extra token from x
    return jp.array(x), jp.array(y)  # token, next token

# setup
key = jax.random.PRNGKey(42)
block_size = 32 # context length
batch_size = 32
n_layer, n_head, n_embd, vocab_size = 4, 4, 4*32, 65
nh, hs = n_head, n_embd//n_head
B, T, C = batch_size, block_size, n_embd
mask = jp.tril(jp.ones((T,T)))

# params
wte = jax.random.normal(key, shape=(vocab_size, n_embd)) * 0.0002
Wi = jax.random.normal(key, shape=(n_layer, n_embd, 3*n_embd)) * 0.0001
Wo = jax.random.normal(key, shape=(n_layer, n_embd, n_embd)) * 0.0001
lm_head = jax.random.normal(key, shape=(n_embd, vocab_size)) * 0.0001
p0 = Wi, Wo, lm_head, wte
param_counts = sum(jax.tree_util.tree_map(lambda p: p.size, p0))
print(f'{param_counts/1e6:0.2f} M params')

Z = lambda x: (x - x.mean(axis=-1)[...,None])/x.std(ddof=1, axis=-1)[...,None]

parallel = True
def model1(params, x, phi=jax.nn.gelu):
    Wi, Wo, lm_head, wte = params
    x = wte[x] # B, T, C
    for wi, wo in zip(Wi, Wo):
        q, k, v = Z(jp.einsum('ij,bti->btj', wi, x).reshape(B,T,3,nh,hs)).swapaxes(0,2)  # (3,T,B,nh,hs)
        if parallel:
            s = jp.einsum('ibhd,jbhd->ijbh', phi(q), phi(k))
            vt = jp.einsum('ijbh,ij,jbhd->ibhd', s, mask, v)
        else:
            state = jp.cumsum(jp.einsum('tbhi,tbhj->tbhij', phi(k), v), axis=0)
            vt = jp.einsum('tbhi,tbhij->tbhj', phi(q), state)
        x = vt.swapaxes(0,1).reshape(B,T,C) @ wo + x
    return Z(x) @ lm_head


def model2_layer(s, wi, wo, xt, phi=jax.nn.gelu):
    q, k, v = Z(jp.einsum('ij,bi->bj', wi, xt).reshape(B,3,nh,hs)).swapaxes(0,1)
    s = s + jp.einsum('bhi,bhj->bhij', phi(k), v)
    vt = jp.einsum('bhi,bhij->bhj', phi(q), s)
    xt = vt.reshape(B,C) @ wo + xt
    return s, xt

def model2(params, x):
    Wi, Wo, lm_head, wte = params
    nl = Wi.shape[0]
    x = wte[x].swapaxes(0, 1)
    assert x.shape == (T, B, C)
    # swap layer & token loops, but otherwise the same for now
    S = jp.zeros((nl, B, nh, hs, hs))
    for t in range(T):
        xt = x[t] # (B,C)
        for i, (wi, wo) in enumerate(zip(Wi, Wo)):
            s, xt = model2_layer(S[i], wi, wo, xt)
            S = S.at[i].set(s)
        x = x.at[t].set(xt)
    return Z(x.swapaxes(0,1)) @ lm_head


def model3_dbg(params, x):
    Wi, Wo, lm_head, wte = params
    nl = Wi.shape[0]
    x = wte[x].swapaxes(0, 1)
    assert x.shape == (T, B, C)
    # swap layer & token loops, but otherwise the same for now
    S = jp.zeros((nl, B, nh, hs, hs))
    outputs = jp.zeros((nl, B, C)) + 1e-3
    if True:
        def op(c, x):
            s, o = c
            i = jp.concat([x.reshape(1, B, C), o[:-1]])
            s, o = jax.vmap(model2_layer)(s, Wi, Wo, i)
            return (s, o), o[-1]
        if False:
            x_ = jp.pad(x, [(0,nl-1),(0,0),(0,0)], constant_values=1e-4)
            for t, x_in in enumerate(x_):
                (S, outputs), x_out = op((S, outputs), x_in)
                x_ = x_.at[t].set(x_out)
        else:
            x_ = jp.pad(x, [(0,nl-1),(0,0),(0,0)], constant_values=1e-4)
            (s,o), x_ = jax.lax.scan(op, (S,outputs), x_)
        x = x_[nl-1:]
    else:
        for t in range(T + nl - 1):
            inputs = jp.concat([x[t].reshape(1, B, C), outputs[:-1]])
            S, outputs = jax.vmap(model2_layer)(S, Wi, Wo, inputs)
            x = x.at[t-(nl-1)].set(outputs[-1])
    return Z(x.swapaxes(0, 1)) @ lm_head


def model3(params, x):
    Wi, Wo, lm_head, wte = params
    nl = Wi.shape[0]
    x = wte[x].swapaxes(0, 1)
    assert x.shape == (T, B, C)
    # swap layer & token loops, but otherwise the same for now
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

model = model1

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
