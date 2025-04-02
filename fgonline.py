import numpy as np, os, pickle, tqdm, jax, jax.numpy as jp

# setup
key = jax.random.PRNGKey(42)
batch_size = 1
nl, n_head, vocab_size = 8, 8, 65
n_embd = n_head * 32
nh, hs = n_head, n_embd // n_head
B, C = batch_size, n_embd

# params
wte = jax.random.normal(key, shape=(vocab_size, n_embd)) * 0.0002
Wi = jax.random.normal(key, shape=(nl, n_embd, 3*n_embd)) * 0.0001
Wo = jax.random.normal(key, shape=(nl, n_embd, n_embd)) * 0.0001
lm_head = jax.random.normal(key, shape=(n_embd, vocab_size)) * 0.0001
p0 = Wi, Wo, lm_head, wte
param_counts = sum(jax.tree_util.tree_map(lambda p: p.size, p0))
print(f'{param_counts>>10} K params')

Z = lambda x: (x - x.mean(axis=-1)[...,None])/x.std(ddof=1, axis=-1)[...,None]

def model2_layer(s, wi, wo, xt, phi=jax.nn.gelu):
    q, k, v = Z(jp.einsum('ij,bi->bj', wi, xt).reshape(B,3,nh,hs)).swapaxes(0,1)
    s = s + jp.einsum('bhi,bhj->bhij', phi(k), v)
    vt = jp.einsum('bhi,bhij->bhj', phi(q), s)
    xt = vt.reshape(B,C) @ wo + xt
    return s, xt

@jax.jit
def step(pso, ix, iy, lr=1e-4):
    p, s, o = pso
    def f(p, s, o):
        Wi, Wo, lm_head, wte = p
        x = wte[ix]
        i = jp.concat([x.reshape(1, B, C), o[:-1]])
        s, o = jax.vmap(model2_layer)(s, Wi, Wo, i)
        logits = Z(o[-1]) @ lm_head
        yoh = jax.nn.one_hot(iy, logits.shape[-1])
        ll = -(jax.nn.log_softmax(logits, axis=-1) * yoh).sum(axis=-1)
        return ll[0], s, o
    ll0, _, _ = f(p, s, o)  # initial prediction
    # ll is 0 to 5 or so, use it to choose number of gradient steps
    nsteps = jp.array([(ll0+1)**2], jp.int32)[0]
    whval = nsteps, p

    def whcond(c):
        n, p = c
        return n > 0

    def whbody(c):
        n, p = c
        p = jax.tree_util.tree_map(
                lambda p, g: p - lr * g, p,
                jax.grad(lambda p: f(p, s, o)[0])(p))
        return n-1, p

    _, p = jax.lax.while_loop(whcond, whbody, whval)

    if False:
        p, _ = jax.lax.scan(lambda p, i: (
            jax.tree_util.tree_map(
                lambda p,g: p-lr*g, p,
                jax.grad(lambda p: f(p, s, o)[0])(p)), None), p, jp.r_[:nsteps])

    ll1, s, o = f(p, s, o)  # final prediction
    return (p, s, o), ll0, ll1, nsteps


s = jp.zeros((nl, B, nh, hs, hs))
o = jp.zeros((nl, B, C)) + 1e-3
pso = p0, s, o
import tqdm
if False:
    data = np.r_[:vocab_size]
    np.random.shuffle(data)
else:
    import genarith
    data = np.array(genarith.gentokens(50000))
for i in (pbar := tqdm.trange(data.size-1)):
    ix, iy = data[i], data[i+1]
    pso, ll0, ll1, nn = step(pso, ix, iy)
    if i > 10:
        expr = ''.join([chr(_) for _ in data[i-9:i+1]])
        pbar.set_description(f'"{expr}: {ll0:+0.2f}->{ll1:+0.2f}"')
