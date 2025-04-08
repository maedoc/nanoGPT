import numpy as np, os, pickle, tqdm

with open('femto2.pkl', 'rb') as fd:
    p0 = pickle.load(fd)
Wi, Wo, lm_head, wte, b = p0 = [np.array(_) for _ in p0]
n_layer, n_embd, _ = Wi.shape
nh = b.shape[-1]
hs = n_embd // nh
vocab_size = wte.shape[0]

Z = lambda x: (x - x.mean(axis=-1)[...,None])/x.std(ddof=1, axis=-1)[...,None]

# def op(Sb, qkv):
#     S, b = Sb
#     b0, b1, b2 = b
#     q, k, v = qkv
#     Skk = -jp.einsum('hij,hi,hj,h->hij', S, k, k, b1)
#     vk = jp.einsum('hi,hj,h->hij', v, k, b2)
#     S = S*(1-b0.reshape(nh,1,1)) + Skk + vk
#     vt = jp.einsum('hi,hij->hj', q, S)
#     return (S, b), vt
# @jax.jit
# def model1(params, ix, phi=jax.nn.gelu):
#     Wi, Wo, lm_head, wte, b = params
#     x = wte[ix] # T, C
#     T, C = x.shape
#     for wi, wo in zip(Wi, Wo):
#         q, k, v = jp.einsum('ij,ti->tj', wi, x).reshape(T,3,nh,hs).swapaxes(0,1)  # (3,T,nh,hs)
#         q, k, v = phi(Z(q)), phi(Z(k)), Z(v)
#         S0 = jp.einsum('hi,hj->hij', v[0], k[0])
#         _, vt = jax.lax.scan(op, (S0,b), (q, k, v))
#         x = vt.reshape(T, C) @ wo + x
#     return Z(x) @ lm_head

def gelu2(x):
    return 1/(1+np.exp(-(1.702 * x))) * x

def step(params, x, S, phi=gelu2):
    Wi, Wo, lm_head, wte, (b0, b1, b2) = params
    for i, (wi, wo, s) in enumerate(zip(Wi, Wo, S)):
        q, k, v = np.einsum('ij,i->j', wi, x).reshape(3,nh,hs)
        q, k, v = phi(Z(q)), phi(Z(k)), Z(v)

        skk = -np.einsum('hij,hi,hj,h->hij', s, k, k, b1)
        vk = np.einsum('hi,hj,h->hij', v, k, b2)
        s = s*(1-b0.reshape(nh,1,1)) + skk + vk
        vt = np.einsum('hi,hij->hj', q, s)
        S[i] = s

        x = vt.reshape(n_embd) @ wo + x
    return Z(x), S

out_dir = 'out-shakespeare-char'
meta_path = 'data/shakespeare_char/meta.pkl'
print(f"Loading meta from {meta_path}...")
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# ix = jp.array(encode('We good wife'))
# for i in tqdm.trange(32):
#     x = model1(p0, ix)
#     probs = jax.nn.softmax(x[-1])
#     top5 = probs * (probs > probs.sort()[-15])
#     nix = np.random.multinomial(1, np.array(top5)).argmax().item()
#     ix = jp.r_[ix, nix]

ix = encode('We good wife')
S = np.zeros((n_layer, nh, hs, hs), 'f')
x = np.zeros((n_embd,), 'f')

# process prompt
for t in tqdm.trange(len(ix)):
    x, S = step(p0, wte[ix[t]], S)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

for i in tqdm.trange(2048):
    x, S = step(p0, wte[ix[-1]], S)
    probs = softmax(x @ lm_head)
    top5 = probs * (probs > np.sort(probs)[-15])
    nix = np.random.multinomial(1, np.array(top5)).argmax().item()
    ix.append(nix)

print(decode(ix))
