import numpy as np, os, pickle, tqdm

with open('femto2.pkl', 'rb') as fd:
    p0 = pickle.load(fd)
    for _ in p0:
        print(_.shape, _.dtype)
Wi, Wo, lm_head, wte, b = p0 = [np.array(_).astype('f') for _ in p0]
n_layer, n_embd, _ = Wi.shape
nh = b.shape[-1]
hs = n_embd // nh
vocab_size = wte.shape[0]

Z = lambda x: (x - x.mean(axis=-1)[...,None])/x.std(ddof=1, axis=-1)[...,None]

wte_scl = 127/np.abs(wte).max()
wte_q = (wte * wte_scl).astype(np.int8)

# we could simulate quants like so
p0_ = []
qp = {}
for key, p in zip('Wi Wo lm_head wte b'.split(' '), p0):
    p_scl = 128 # 127/np.abs(p).max()
    l2scl = int(np.log2(127/np.abs(p).max()))
    p_scl = 2**l2scl
    p_p_scl = p * p_scl
    ppmax = np.abs(p_p_scl).max()
    print(key, p_scl, ppmax)
    assert ppmax < 127, (key, ppmax)
    p_q = p_p_scl.astype(np.int8)
    p0_.append(p_q.astype('f') / p_scl)
    qp[f'{key}'] = p_q, l2scl
Wi, Wo, lm_head, wte, b = p0 = p0_

def compare(x,xq):
    import pylab as pl
    pl.plot(xq.reshape(-1), x.reshape(-1), 'k.')
    pl.grid(1)
    pl.savefig('xq.png')
    os.system('xdg-open xq.png')
    np.testing.assert_allclose(xq, x)
    1/0

def overhead(*arrays):
    "checks arrays can be multiplied w/o overflowing int32"
    aa = lambda a: np.abs(a).max().astype(np.int64)**2
    imax = np.iinfo(np.int32).max
    for i, a in enumerate(arrays):
        of = 0
        while aa(a>>of) >= imax:
            of += 1
        assert aa(a) < imax, f'overflow, a[{i}]>>={of}'

# quant test
x = wte[0] @ Wi[0]
q, k, v = x.reshape(3, nh, hs)
kk = k[:, None]*k[:, :, None]
vk = v[:, :, None]*k[:, None]

wteq, wtep = qp['wte']
Wiq, Wip = qp['Wi']

# xq = wteq[0].astype(np.int32)[:,None] * Wiq[0].astype(np.int32)

xq = wteq[0].astype(np.int32) @ Wiq[0].astype(np.int32)
xp = wtep + Wip
# compare(x*2**xp, xq)
qq, kq, vq = xq.reshape(3, nh, hs)
overhead(qq, kq, vq)

kkq = kq[:, None]*kq[:, :, None]
vkq = vq[:, :, None]*kq[:, None]
kkp = xp*2
vkp = xp*2
# but this might overflow, so rshift here
rs = 13
kkq >>= rs
vkq >>= rs
kkp -= rs
vkp -= rs
overhead(kkq, vkq)
# compare(kk, kkq.astype('f')/2**kkp)
# compare(vk, vkq.astype('f')/2**kkp)

# s = s*l + b2*vk, l = 1 - b0 - b1*kk
sq, sp = vkq, vkp

# bias values
bq, bp = qp['b']
b0q, b1q, b2q = bq.reshape(3, nh, 1, 1).astype(np.int32)
b0, b1, b2 = b.reshape(3, nh, 1, 1)

# l is tricky since it mixes scales
# - b1*kk scale is bp+kkp
# - b0 is just bp
# - (b0<<kkp) will be bp+kkp
# so lp should be bp+kkp, 1 becomes lp
# 1 is just the scale itself, so just lp
lp = bp + kkp
lq = (1 << lp) - (b0q << kkp) - b1q*kkq
overhead(b1q, kkq)
rs = 15
lq, lp = lq >> rs, lp-rs
overhead(lq)  # -> lq >>= 7
l = 1 - (b[0].reshape(-1, 1, 1) + b[1].reshape(-1, 1, 1)*kk)
# compare(l, lq.astype('f')/2**lp)

# s = s*l
overhead(sq, lq)
sq, sp = sq*lq, sp+lp
# s is recurrent, sp needs to be constant but rs=lp results in prec loss
rs = 15
sq, sp = sq >> rs, sp - rs
overhead(sq)

s = vk
s = s*l
compare(s, sq.astype('f')/2**sp)

1/0

fdc = open('femto2-weights.c', 'w')
fdh = open('femto2-weights.h', 'w')
fdh.write('#include <stdint.h>\n\n')
for key, val in zip('Wi Wo lm_head wte b'.split(' '), p0):
    vals = ','.join([str(_) for _ in val.flat])
    valsh = ','.join([str(_) for _ in val.shape])
    fdh.write(f'const uint32_t fm_{key}_ndim = {{ {len(valsh)} }};\n')
    fdh.write(f'const uint32_t fm_{key}_shape[{len(valsh)}] = {{ {valsh} }};\n')
    fdh.write(f'extern const float *fm_{key};\n')
    fdc.write(f'const float fm_{key}[] = {{ {vals} }};\n')
fdc.close()
fdh.close()

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

def step1(params, x, S, phi=gelu2):
    Wi, Wo, lm_head, wte, (b0, b1, b2) = params
    for i, (wi, wo, s) in enumerate(zip(Wi, Wo, S)):
        q, k, v = np.einsum('ij,i->j', wi, x).reshape(3,nh,hs)
        q, k, v = phi(Z(q)), phi(Z(k)), Z(v)
        # basically S*(1-b0-b1*kk) + b2*vk
        skk = -np.einsum('hij,hi,hj,h->hij', s, k, k, b1)
        vk = np.einsum('hi,hj,h->hij', v, k, b2)
        s = s*(1-b0.reshape(nh,1,1)) + skk + vk
        vt = np.einsum('hi,hij->hj', q, s)
        S[i] = s
        x = vt.reshape(n_embd) @ wo + x
    return Z(x), S


assert os.system('gcc -O3 -c femto2.c') == 0
assert os.system('gcc -shared femto2.o -o femto2.so -lm') == 0
from ctypes import CDLL, c_int
fvec = np.ctypeslib.ndpointer(dtype=np.float32)
ivec = np.ctypeslib.ndpointer(dtype=np.int32)
lib = CDLL('./femto2.so')
lib.layer.restype = None
lib.layer.argtypes = c_int, c_int, fvec, fvec, fvec, fvec, fvec, fvec, fvec
lib.layers.argtypes = c_int, c_int, c_int, fvec, fvec, fvec, fvec, fvec, fvec, fvec


def step2_dbg(params, x, S, phi=gelu2):
    Wi, Wo, lm_head, wte, (b0, b1, b2) = params
    b0, b1, b2 = b.reshape(3, nh, 1, 1)
    qkv = np.zeros((3, nh, hs), 'f')
    vt_ = np.zeros((nh, hs), 'f')
    xo_ = np.zeros((nh * hs, ), 'f')
    for i, (wi, wo, s) in enumerate(zip(Wi, Wo, S)):
        # q, k, v = np.dot(wi.T, x).reshape(3, nh, hs)
        # q, k, v = phi(Z(q)), phi(Z(k)), Z(v)
        # q, k, v = qkv
        s_ = s.copy()
        lib.layer(nh, hs, qkv, x, wi, s_, b, vt_, wo, xo_)
        q, k, v = qkv
        kk = k[:, None]*k[:, :, None]
        vk = v[:, :, None]*k[:, None]
        s = s*(1 - b0 - b1*kk) + b2*vk
        np.testing.assert_allclose(s, s_, 1e-6, 1e-6)
        vt = (q[:, :, None] * s).sum(axis=1)
        np.testing.assert_allclose(vt, vt_, 1e-6, 1e-6)
        S[i] = s
        xo = vt.reshape(n_embd) @ wo + x
        np.testing.assert_allclose(xo, xo_, 1e-6, 1e-6)
        x = xo_.copy()
    return Z(x), S


def step2(params, x, S, phi=gelu2):
    x = x.copy()
    Wi, Wo, lm_head, wte, b = params
    qkv = np.zeros((3, nh, hs), 'f')
    vt_ = np.zeros((nh, hs), 'f')
    # for i, (wi, wo, s) in enumerate(zip(Wi, Wo, S)):
        # lib.layer(nh, hs, qkv, x, wi, s, b, vt_, wo)
    lib.layers(n_layer, nh, hs, qkv, x, Wi, S, b, vt_, Wo)
    return Z(x), S


def step(p,x,S):
    x1, S1 = step1(p, x, S.copy())
    x2, S2 = step2(p, x, S.copy())
    np.testing.assert_allclose(x1, x2, 1e-4, 1e-4)
    np.testing.assert_allclose(S1, S2, 1e-4, 1e-4)
    return x2, S2


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

np.random.seed(42)

for i in tqdm.trange(256):
    x, S = step(p0, wte[ix[-1]], S)
    probs = softmax(x @ lm_head)
    top5 = probs * (probs > np.sort(probs)[-15])
    nix = np.random.multinomial(1, np.array(top5)).argmax().item()
    ix.append(nix)

print(decode(ix))
