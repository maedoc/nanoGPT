import numpy as np, os, pickle, tqdm, jax, jax.numpy as jp

data_dir = os.path.join('data', 'shakespeare_char')
def get_batch(split):
    fname = 'train.bin' if split == 'train' else 'val.bin'
    data = np.memmap(os.path.join(data_dir, fname), dtype=np.uint16, mode='r')
    block_size = T * P
    ix = np.random.randint(len(data) - block_size, size=(batch_size,))
    x = jp.stack([(data[i:i+block_size]).astype(np.int64) for i in ix])
    y = jp.stack([(data[i+1:i+1+block_size]).astype(np.int64) for i in ix])
    x, y = [_.reshape(B, T, P) for _ in (x, y)]
    return x, y  # token, next token

# setup
key = jax.random.PRNGKey(42)
block_size = 64 # context length
batch_size = 8
n_layer, n_head, n_embd, vocab_size = 6, 8, 8*32, 65
nh, hs = n_head, n_embd//n_head
B, T, P, C = batch_size, block_size, 8, n_embd
mask = jp.tril(jp.ones((P, P)))

# params
wte = jax.random.normal(key, shape=(vocab_size, n_embd)) * 0.0002
wpe = jax.random.normal(key, shape=(P, n_embd)) * 0.0002
Wi = jax.random.normal(key, shape=(n_layer, n_embd, 4*n_embd)) * 0.0001
Wo = jax.random.normal(key, shape=(n_layer, n_embd, n_embd)) * 0.0001
lm_head = jax.random.normal(key, shape=(n_embd, vocab_size)) * 0.0001
p0 = Wi, Wo, lm_head, wte, wpe
param_counts = sum(jax.tree_util.tree_map(lambda p: p.size, p0))
print(f'{param_counts/1e6:0.2f} M params')

Z = lambda x: (x - x.mean(axis=-1)[...,None])/x.std(ddof=1, axis=-1)[...,None]

def model2d(params, x, phi=jax.nn.gelu, parallel=True):
    Wi, Wo, lm_head, wte, wpe = params
    x = wte[x] + wpe # B, T, P, C
    for wi, wo in zip(Wi, Wo):
        # linear projections
        # XXX p is new index for parallel token size
        q, kp, ks, v = jp.einsum('ij,btpi->btpj', wi, x).reshape(B,T,P,4,nh,hs).swapaxes(0,3)  # (3,T,P,B,nh,hs)
        # only nonlinearities here:
        q, kp, ks, v = phi(Z(q)), phi(Z(kp)), phi(Z(ks)), Z(v)
        # temporal attention
        ss = jp.einsum('tpbhi,tpbhj->tpbhij', ks, v)
        ss = jp.cumsum(ss, axis=0)  # tpbhij
        vs = jp.einsum('tpbhij,tpbhi->tpbhj', ss, q)
        # spatial attention
        sp = jp.einsum('tpbhi,tPbhi->tpPbhi', q, kp)
        vp = jp.einsum('tpPbhi,pP,tPbhi->tpbhi', sp, mask, vs)
        x = vp.reshape(T,P,B,C).transpose(2,0,1,3) @ wo + x
    return Z(x) @ lm_head

def loss(W, x, y):
    logits = model2d(W, x)
    yoh = jax.nn.one_hot(y, logits.shape[-1])
    lls = -(jax.nn.log_softmax(logits, axis=-1) * yoh)
    ll = lls.sum(axis=-1)
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
