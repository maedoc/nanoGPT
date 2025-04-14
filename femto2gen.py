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
quants = {}
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
    quants[f'{key}'] = p_q, l2scl
# Wi, Wo, lm_head, wte, b = p0 = p0_

###### QUANT DEV
def compare(x,xq):
    import pylab as pl
    pl.plot(xq.reshape(-1), x.reshape(-1), 'k.')
    pl.grid(1)
    pl.savefig('xq.png')
    os.system('xdg-open xq.png')
    np.testing.assert_allclose(xq, x)
    1/0

def overhead(a, raise_=True):
    "checks arrays can be multiplied w/o overflowing int32"
    aa = lambda a: np.abs(a).max().astype(np.int64)**2
    imax = np.iinfo(np.int32).max
    of = 0
    while aa(a >> of) >= imax:
        of += 1
    if of > 0:
        if raise_:
            assert aa(a) < imax, f'overflow, a>>={of}'
        else:
            msg = f'a>>={of}'
            print(msg)
            a >>= of


if False:
    # quant test
    x = wte[0]
    q, k, v = (x @ Wi[0]).reshape(3, nh, hs)
    kk = k[:, None]*k[:, :, None]
    vk = v[:, :, None]*k[:, None]

    wteq, wtep = quants['wte']
    Wiq, Wip = quants['Wi']

    # xq = wteq[0].astype(np.int32)[:,None] * Wiq[0].astype(np.int32)

    xq = wteq[0].astype(np.int32)
    xp = wtep
    # compare(x, xq.astype('f')/2**xp)
    qq, kq, vq = (xq @ Wiq[0].astype(np.int32)).reshape(3, nh, hs)
    qp, kp, vp = [xp + Wip for _ in range(3)]
    overhead(qq, kq, vq)

    kkq = kq[:, None]*kq[:, :, None]
    vkq = vq[:, :, None]*kq[:, None]
    kkp = kp+kp
    vkp = vp+kp
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
    bq, bp = quants['b']
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

    # s *= l
    overhead(sq, lq)
    sq, sp = sq*lq, sp+lp
    # s is recurrent, sp needs to be constant but rs=lp results in prec loss
    rs = 15
    sq, sp = sq >> rs, sp - rs
    overhead(sq)

    s = vk
    s = s*l
    # compare(s, sq.astype('f')/2**sp)

    # s += b2*vk
    b2vkq = b2q * vkq
    b2vkp = bp + vkp
    rs = b2vkp - sp  # expected to be positive
    sq = (sq << rs) if rs > 0 else (sq >> (-rs))
    sp = sp + rs
    sq = sq + b2vkq
    rs = 8
    sq >>= rs
    sp -= rs
    overhead(sq)

    s = s + b2*vk
    # compare(s, sq.astype('f')/2**sp)

    # vt = (q[:,:,None] * s).sum(axis=1)
    vt = (q[:, :, None] * s).sum(axis=1)
    vtq = (qq[:, :, None] * sq).sum(axis=1)
    vtp = xp + sp
    rs = 13
    vtq >>= rs
    vtp -= rs
    overhead(vtq)
    # compare(vt, vtq.astype('f')/2**vtp)

    # xo = vt.reshape(n_embd) @ wo + x
    xo = vt.reshape(-1) @ Wo[0] + x
    Woq, Wop = quants['Wo']
    xoq = vtq.reshape(-1) @ Woq[0].astype(np.int32)
    xop = vtp + Wop
    xoq = (xoq >> (xp - xop)) + xq
    xop = xp
    overhead(xoq)
    print(xop, xp)
    compare(xo, xoq.astype('f')/2**xop)

    # [x] so x at end of layer now has same p as start
    # [ ] S? need to try and watch it
    print(sp)
    1/0

if False:
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

# def gelu2(x):
#     return 1/(1 + np.exp(-(1.702 * x))) * x

# def gelu2_(x__):
#     x = 1.702 * x__
#     emx = 1. - x*(1. - x/2.0*(1.0 - (2.5*x)/8))  # div by pow of 2 just shift
#     return 1/(1 + emx) * x__

# def phi_(x):
#     return 1/(1+np.exp(-x))

# def phi(x_):
#     x = -np.abs(x_)
#     emx = 1 - x*(1 - x/2*(1 - x/2))
#     s = 1/(1 + emx)
#     return np.where(x_ >= 0, 1-s, s)


def phi(x, a=0.01):
    return np.where(x >= 0, x, a*x)


def step1(params, x, S):
    Wi, Wo, lm_head, wte, (b0, b1, b2) = params
    for i, (wi, wo, s) in enumerate(zip(Wi, Wo, S)):
        q, k, v = np.einsum('ij,i->j', wi, x).reshape(3,nh,hs)
        q, k, v = phi(Z(q)), phi(Z(k)), Z(v)
        # basically S*(1-b0-b1*kk) + b2*vk
        skk = -np.einsum('hij,hi,hj,h->hij', s, k, k, b1)
        vk = np.einsum('hi,hj,h->hij', v, k, b2)
        s = s*(1-b0.reshape(nh, 1, 1)) + skk + vk
        vt = np.einsum('hi,hij->hj', q, s)
        S[i] = s
        x = vt.reshape(n_embd) @ wo + x
    return Z(x), S


assert os.system('gcc -O3 -ffast-math -mavx2 -march=native -c femto2.c') == 0
assert os.system('gcc -shared femto2.o -o femto2.so -lm') == 0
from ctypes import CDLL, c_int
fvec = np.ctypeslib.ndpointer(dtype=np.float32)
ivec = np.ctypeslib.ndpointer(dtype=np.int32)
lib = CDLL('./femto2.so')
lib.layer.restype = None
lib.layer.argtypes = c_int, c_int, fvec, fvec, fvec, fvec, fvec, fvec, fvec
lib.layers.argtypes = c_int, c_int, c_int, fvec, fvec, fvec, fvec, fvec, fvec, fvec
lib.layers.restype = None
lib.softmax.argtypes = c_int, fvec, fvec
lib.softmax.restype = None
lib.top15.argtypes = c_int, fvec, fvec
lib.top15.restype = c_int


# def step2_dbg(params, x, S):
#     Wi, Wo, lm_head, wte, (b0, b1, b2) = params
#     b0, b1, b2 = b.reshape(3, nh, 1, 1)
#     qkv = np.zeros((3, nh, hs), 'f')
#     vt_ = np.zeros((nh, hs), 'f')
#     xo_ = np.zeros((nh * hs, ), 'f')
#     for i, (wi, wo, s) in enumerate(zip(Wi, Wo, S)):
#         # q, k, v = np.dot(wi.T, x).reshape(3, nh, hs)
#         # q, k, v = phi(Z(q)), phi(Z(k)), Z(v)
#         # q, k, v = qkv
#         s_ = s.copy()
#         lib.layer(nh, hs, qkv, x, wi, s_, b, vt_, wo, xo_)
#         q, k, v = qkv
#         kk = k[:, None]*k[:, :, None]
#         vk = v[:, :, None]*k[:, None]
#         s = s*(1 - b0 - b1*kk) + b2*vk
#         np.testing.assert_allclose(s, s_, 1e-6, 1e-6)
#         vt = (q[:, :, None] * s).sum(axis=1)
#         np.testing.assert_allclose(vt, vt_, 1e-6, 1e-6)
#         S[i] = s
#         xo = vt.reshape(n_embd) @ wo + x
#         np.testing.assert_allclose(xo, xo_, 1e-6, 1e-6)
#         x = xo_.copy()
#     return Z(x), S

c_qkv = np.zeros((3, nh, hs), 'f')
c_vt = np.zeros((nh, hs), 'f')

WiT = Wi.transpose(0, 2, 1).copy()
WoT = Wo.transpose(0, 2, 1).copy()

def step2(params, x, S):
    x = x.copy()
    S = S.transpose(0, 1, 3, 2).copy()
    lib.layers(n_layer, nh, hs, c_qkv, x, WiT, S, b, c_vt, WoT)
    return x, S.transpose(0, 1, 3, 2).copy()

def fq(f, xq, xp):
    z = f(xq.astype('f') / 2**xp)
    zq = (z*2**xp).astype(np.int32)
    return zq

Zq = lambda xq, xp: fq(Z, xq, xp)
phiq = lambda xq, xp: fq(phi, xq, xp)

def step3(params, x, S):
    wteq, wtep = quants['wte']
    Wiq, Wip = quants['Wi']
    Woq, Wop = quants['Wo']
    bq, bp = quants['b']
    b0q, b1q, b2q = bq.reshape(3, nh, 1, 1).astype(np.int32)
    Sp = 8
    Sq = (S*2**Sp).astype(np.int32)
    # xq = wteq[0].astype(np.int32)
    xp = wtep
    xq = (x*2**xp).astype(np.int32)
    for i, (wiq, woq, sq) in enumerate(zip(Wiq.astype(np.int32), Woq.astype(np.int32), Sq)):
        qq, kq, vq = (xq @ wiq.astype(np.int32)).reshape(3, nh, hs)
        qp, kp, vp = [xp + Wip for _ in range(3)]
        qq, kq, vq = phiq(Zq(qq, qp), qp), phiq(Zq(kq, kp), kp), Zq(vq, vp)
        rs = 13  # rshift loses precision but avoids overflow
        kkq = (kq[:, None]*kq[:, :, None]) >> rs
        vkq = (vq[:, :, None]*kq[:, None]) >> rs
        kkp = kp+kp-rs
        vkp = vp+kp-rs
        overhead(kkq, raise_=False)
        overhead(kkq, raise_=False)
        # s *= l, l = 1 - b0 + b1*kk
        lp = bp + kkp
        lq = (1 << lp) - (b0q << kkp) - b1q*kkq
        rs = 15
        lq, lp = lq >> rs, lp-rs
        sq, sp = sq*lq, Sp + lp
        rs = 15
        sq, sp = sq >> rs, sp - rs
        overhead(sq, raise_=False)
        # s += b2*vk
        b2vkq = b2q * vkq
        b2vkp = bp + vkp
        rs = b2vkp - sp  # expected to be positive
        sq = (sq << rs) if rs > 0 else (sq >> (-rs))
        sp = sp + rs
        sq = sq + b2vkq
        rs = 8
        sq >>= rs
        sp -= rs
        overhead(sq, raise_=False)
        vtq = (qq[:, :, None] * sq).sum(axis=1)
        vtp = xp + sp
        rs = 13
        vtq >>= rs
        vtp -= rs
        overhead(vtq, raise_=False)
        # xo = vt.reshape(n_embd) @ wo + x
        xoq = vtq.reshape(-1) @ Woq[0].astype(np.int32)
        xop = vtp + Wop
        xoq = (xoq >> (xp - xop)) + xq
        xop = xp
        overhead(xoq, raise_=False)
        S[i] = (sq.astype('f')/2**sp)
    x = xq.astype('f') / 2**xp
    return Z(x), S


def step(p,x,S):
    x1, S1 = step1(p, x, S.copy())
    x2, S2 = step2(p, x, S.copy())
    # x3, S3 = step3(p, x, S.copy())
    np.testing.assert_allclose(x1, x2, 1e-4, 1e-4)
    np.testing.assert_allclose(S1, S2, 1e-4, 1e-4)
    # np.testing.assert_allclose(x1, x3, 1e-4, 1e-4)
    # np.testing.assert_allclose(S1, S3, 1e-4, 1e-4)
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

ix = encode('We good wife are going for a walk today')
S = np.zeros((n_layer, nh, hs, hs), 'f')
x = np.zeros((n_embd,), 'f')

# process prompt
for t in tqdm.trange(len(ix)):
    x, S = step(p0, wte[ix[t]], S)

# def softmax_(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum()

# def softmax(x):
#     ex = x.copy()
#     lib.softmax(x.size, x, ex)
#     ex2 = softmax_(x)
#     np.testing.assert_allclose(ex, ex2, 1e-6, 1e-6)
#     return ex2

np.random.seed(42)

probs = np.zeros(lm_head.shape[1], 'f')
c_probsort = np.zeros_like((lm_head.shape[1], 2), 'f')
for i in tqdm.trange(256):
    x, S = step(p0, wte[ix[-1]], S)
    lib.softmax(probs.size, x@lm_head, probs)
    top15_ = probs * (probs > np.sort(probs)[-16])

    top15 = probs.copy()
    nix = lib.top15(top15.size, top15, c_probsort)

    # print(top15_[top15_ > 0], top15[top15 > 0])
    np.testing.assert_allclose(top15_[top15_ > 0], top15[top15 > 0])

    # nix = np.random.multinomial(1, top15).argmax().item()
    ix.append(nix)

print(decode(ix))
