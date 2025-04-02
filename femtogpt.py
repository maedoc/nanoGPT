import numpy as np, os, pickle, tqdm, jax, jax.numpy as jp

# data
data_dir = os.path.join('data', 'shakespeare_char')
def get_batch(split):
    fname = 'train.bin' if split == 'train' else 'val.bin'
    data = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode='r')
    ix = np.random.randint(len(data) - block_size, size=(batch_size,))
    x = jp.stack([(data[i:i+block_size]).astype(np.int64) for i in ix])
    y = jp.stack([(data[i+1:i+1+block_size]).astype(np.int64) for i in ix])
    return x, y  # token, next token

# setup
key = jax.random.PRNGKey(42)
block_size = 16 # context length
batch_size = 8
n_layer, n_head, n_embd, vocab_size = 4, 3, 24, 65
nh, hs = n_head, n_embd//n_head
B, T, C = batch_size, block_size, n_embd
beta = 0.01

# params
wte = jax.random.normal(key, shape=(vocab_size, n_embd)) * 0.0002
Wi = jax.random.normal(key, shape=(n_layer, n_embd, 3*n_embd)) * 0.0001
Wo = jax.random.normal(key, shape=(n_layer, n_embd, n_embd)) * 0.0001
lm_head = jax.random.normal(key, shape=(n_embd, vocab_size)) * 0.0001
p0 = Wi, Wo, lm_head, wte
param_counts = sum(jax.tree_util.tree_map(lambda p: p.size, p0))
print(f'{param_counts/1e6:0.2f} M params')

Z = lambda x: (x - x.mean(axis=-1)[...,None])/x.std(ddof=1, axis=-1)[...,None]

def model(params, x, phi=jax.nn.gelu):
    Wi, Wo, lm_head, wte = params
    x = wte[x] # B, T, C
    for wi, wo in zip(Wi, Wo):
        q, k, v = Z(jp.einsum('ij,bti->btj', wi, x).reshape(B,T,3,nh,hs)).swapaxes(0,2)
        state = jp.cumsum(jp.einsum('tbhi,tbhj->tbhij', phi(k), v), axis=0)
        vt = jp.einsum('tbhi,tbhij->tbhj', phi(q), state)
        x = jp.swapaxes(vt,0,1).reshape(B,T,C) @ wo + x
    return Z(x) @ lm_head

def loss(W, x, y):
    logits = model(W, x)
    yoh = jax.nn.one_hot(y, logits.shape[-1])
    ll = -(jax.nn.log_softmax(logits, axis=-1) * yoh).sum(axis=-1)
    return ll.mean()

from jax.example_libraries.optimizers import adam
oinit, oup, oget = adam(3e-4)
o = oinit(p0)
jvg = jax.jit(jax.value_and_grad(loss))
for i in range(201):
    v, g = jvg(oget(o), *get_batch('train'))
    o = oup(i, g, o)
    if i % 20 == 0:
        print(i, v)
