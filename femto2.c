#include <stdint.h>
#include <math.h>

void z(int n, float *x)
{
  // compute mean & std
  int count = 0;
  float mean=0., m2=0.;
  for (int j=0; j<n; j++)
  {
    float q = x[j];
    count += 1;
    float delta = q - mean;
    mean += delta / count;
    float delta2 = q - mean;
    m2 += delta * delta2;
  }
  float std = sqrt(m2 / (count - 1));
  // apply for layer norm effect
  for (int j = 0; j < n; j++)
    x[j] = (x[j] - mean) / std;
}

void gelu2(int n, float *x)
{
  for (int i = 0; i < n; i++)
    x[i] = 1.f / (1.f + expf(-(1.702f * x[i]))) * x[i];
}

void layer(int nh, int hs, float *qkv, float *x, float *wi, float *s, float *b, float *vt, float *wo)
{
  int C = nh * hs;
  // q, k, v = np.dot(wi.T, x).reshape(3, nh, hs)
  // wi.shape (128, 3*C)
  for (int j=0; j<(3*C); j++)
  {
    float acc = 0.f;
    for (int i=0; i<C; i++)
      acc += x[i] * wi[(3*C) * i + j];
    qkv[j] = acc;
  }
  // q, k, v = phi(Z(q)), phi(Z(k)), Z(v)
  for (int i = 0; i < (3 * nh); i++)
    z(hs, qkv + i * hs);
  gelu2(2*C, qkv);
  // q, k, v = qkv, each shaped (nh, hs)
  float *q=qkv, *k=qkv+C, *v=qkv+2*C;
  // multi head delta net attention
  for (int h = 0; h < nh; h++) {
    float *sh = s + h * hs * hs; // s.shape (nh, hs, hs)
    float *kh = k + h * hs, *vh = v + h * hs;
    float b0 = b[h], b1 = b[nh + h], b2 = b[2 * nh + h]; // b.shape (3, nh)
    for (int i = 0; i < hs; i++)
      for (int j = 0; j < hs; j++) {
        // kk = k[:, None]*k[:, :, None]
        // vk = v[:, :, None]*k[:, None]
        // s = s*(1 - b0 - b1*kk) + b2*vk
        sh[i * hs + j] *= 1.f - b0 - b1 * kh[i] * kh[j]; // decay state
        sh[i * hs + j] += b2 * vh[i] * kh[j]; // add new key-val
      }
    // retrieve query
    float *qh = q + h * hs, *vth = vt + h * hs;
    for (int i = 0; i < hs; i++) {
      float acc = 0.f; // sum(qh[:]*s[:,i])
      for (int j = 0; j < hs; j++)
        acc += qh[j] * sh[j * hs + i];
      vth[i] = acc;
    }
  }
  // x = vt.reshape(n_embd) @ wo + x
  for (int i=0; i<C; i++) {
    float acc = 0.f;
    for (int j=0; j<C; j++)
      acc += vt[j] * wo[j * C + i];
    x[i] += acc;
  }
}

void layers(int nl, int nh, int hs, float *qkv, float *x, float *wi, float *s, float *b, float *vt, float *wo) {
  int C = nh*hs;
  for (int i=0; i<nl; i++)
    layer(nh, hs, qkv, x, wi+i*C*3*C, s+i*nh*hs*hs, b, vt, wo+i*C*C);
}
