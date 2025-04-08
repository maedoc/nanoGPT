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
block_size = 256 # context length
batch_size = 32
n_head = 4
n_layer, n_embd, vocab_size = 2, n_head*16, 65
nh, hs = n_head, n_embd//n_head
B, T, C = batch_size, block_size, n_embd

# params
wte = jax.random.normal(key, shape=(vocab_size, n_embd)) * 0.0002
Wi = jax.random.normal(key, shape=(n_layer, n_embd, 3*n_embd)) * 0.0001
Wo = jax.random.normal(key, shape=(n_layer, n_embd, n_embd)) * 0.0001
lm_head = jax.random.normal(key, shape=(n_embd, vocab_size)) * 0.0001
p0 = Wi, Wo, lm_head, wte, 0.1*jp.ones((3,n_head))
param_counts = sum(jax.tree_util.tree_map(lambda p: p.size, p0))
print(f'{param_counts/1e6:0.2f} M params')

Z = lambda x: (x - x.mean(axis=-1)[...,None])/x.std(ddof=1, axis=-1)[...,None]

def op(Sb, qkv):
    S, b = Sb
    b0, b1, b2 = b
    q, k, v = qkv
    Skk = -jp.einsum('bhij,bhi,bhj,h->bhij', S, k, k, b1)
    vk = jp.einsum('bhi,bhj,h->bhij', v, k, b2)
    S = S*(1-b0.reshape(nh,1,1)) + Skk + vk
    vt = jp.einsum('bhi,bhij->bhj', q, S)
    return (S, b), vt

def model1(params, x, phi=jax.nn.gelu, delta=True):
    Wi, Wo, lm_head, wte, b = params
    x = wte[x] # B, T, C
    for wi, wo in zip(Wi, Wo):
        q, k, v = jp.einsum('ij,bti->btj', wi, x).reshape(B,T,3,nh,hs).swapaxes(0,2)  # (3,T,B,nh,hs)
        q, k, v = phi(Z(q)), phi(Z(k)), Z(v)
        if delta:  # 1.46
            S0 = jp.einsum('bhi,bhj->bhij', v[0], k[0])
            _, vt = jax.lax.scan(op, (S0,b), (q, k, v))
        else:  # 2.33
            S = jp.cumsum(jp.einsum('tbhi,tbhj->tbhij', v, k), axis=0)
            vt = jp.einsum('tbhi,tbhij->tbhj', q, S)
        x = vt.swapaxes(0,1).reshape(B,T,C) @ wo + x
    return Z(x) @ lm_head

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
for lr in [1e-3, 1e-4]:
    _, oup, _ = adam(lr)
    for i in range(1001):
        v, g = jvg(oget(o), *get_batch('train'))
        o = oup(i, g, o)
        if i % 10 == 0:
            print(i, v, 'b', oget(o)[-1].mean(axis=-1))

with open('femto2.pkl', 'wb') as fd:
    pickle.dump(oget(o), fd)
