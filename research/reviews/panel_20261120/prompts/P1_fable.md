Your final message must be your complete answer: it is the only text that will be kept, so put your whole response in it. Do not use any tools (no file reads, no searches, no commands). The brief follows verbatim.

---

I'm working on a faster variant of the **SOAP** optimizer (Adam run in the eigenbasis of Shampoo's Kronecker
covariance factors). The variant keeps SOAP's refresh cadence, but it replaces four of every five
decomposition-based basis refreshes with a cheap approximate "warm refresh", and it stores all optimizer state in
BF16. I want your **independent thinking** on two questions:

- **(A)** how to make the refresh events, and the method as a whole, faster;
- **(B)** how to make the warm refresh track the QR refresh more closely, for better loss, at modest cost.

I also want the pitfalls you see in the current design.

**Ground rules.** Answer from your own understanding. Don't search the web, don't read or run anything, and don't
use any tools. Mark every literature claim **[verify]**, and add **[certain]** or **[recalled]** to say how sure you
are. Labeled speculation is welcome. Where you claim an algebraic identity or a cost, show the derivation or the
count, because I will check them numerically.

---

## 1. The method

### 1.1 Notation and state

Take one matrix parameter W ∈ R^{m×n} with clipped gradient G. The optimizer keeps six persistent tensors, all
stored in BF16:
- the first and second moments M, V ∈ R^{m×n}, which live in the rotated coordinates;
- the covariance factors C_L ∈ R^{m×m} and C_R ∈ R^{n×n};
- the square bases Q_L ∈ R^{m×m} and Q_R ∈ R^{n×n}.

The master weights W are FP32.

Two kinds of matrix product appear below. Both take BF16 operands and accumulate in FP32:
- **𝓑(X,Y)** rounds the result to BF16;
- **𝓕(X,Y)** keeps the result in FP32.

R_b(·) means rounding to BF16, and sym(X) = (X + Xᵀ)/2.

**First call (k = 1).** The optimizer:
- initializes M = V = 0;
- accumulates C_L and C_R from G;
- sets each basis by FP32 eigendecomposition of sym(C), with columns in descending eigenvalue order, stored in BF16;
- skips the matrix update.

For later calls the matrix age is t = k − 1.

### 1.2 One call at age t ≥ 1 (schedule "warm10/QR50")

```
1  Ḡ   = 𝓑(𝓑(Q_Lᵀ, G_b), Q_R)                       # G_b = R_b(G)
2  M_f = β1·M + (1−β1)·Ḡ ;  V_f = β2·V + (1−β2)·Ḡ²     # FP32 temporaries
3  U   = 𝓕(𝓑(Q_L, R_b[M_f / (sqrt(V_f) + ε)]), Q_Rᵀ)
   W  ← (W − η_k · c_t · U)·(1 − η_k·λ),   c_t = sqrt(1 − β2^t)/(1 − β1^t)
4  M, V ← R_b(M_f), R_b(V_f)
   C_L ← R_b[β_C·C_L + (1−β_C)·G_b G_bᵀ] ;  C_R ← R_b[β_C·C_R + (1−β_C)·G_bᵀ G_b]
5  if t mod 50 == 0:        # QR repair, an FP32 island
       (Q_L⁺, π_L) = QRRepair(C_L, Q_L) ; (Q_R⁺, π_R) = QRRepair(C_R, Q_R)
       permute the rows of V by π_L and its columns by π_R
   elif t mod 10 == 0:      # warm refresh: matrix products only
       Q_L⁺ = WarmRefresh(C_L, Q_L) ; Q_R⁺ = WarmRefresh(C_R, Q_R)       # V is left unchanged
   else:
       return
6  M ← 𝓑(𝓑((Q_L⁺)ᵀ, 𝓑(𝓑(Q_L, M), Q_Rᵀ)), Q_R⁺)       # transport M through world coordinates
   Q_L, Q_R ← Q_L⁺, Q_R⁺
```

The update direction at a call uses the bases from *before* that call's refresh.

**WarmRefresh(C, Q):**

```
B  = sym(𝓕(𝓑(Qᵀ, C), Q))
d  = diag(B);  s = max(max_i |d_i|, 1e-30);  h_ij = (d_i − d_j)/s
A_ij = −½ · (B_ij / s) · h_ij / (h_ij² + 0.01²)                 # skew-symmetric
a  = min(1, 0.5 / max(‖A‖_F/√2, 1e-12))                        # global Frobenius cap
Y0 = R_b(float(Q) + 𝓕(Q, R_b(a·A)))
for j in 0, 1:                                                 # two Newton–Schulz corrections
    E_j     = 𝓕(Y_jᵀ, Y_j) − I
    Y_{j+1} = R_b(float(Y_j) − ½ · 𝓕(Y_j, R_b(E_j)))
return Y2
```

- **The skew motion** is a damped first-order Jacobi-type motion applied to all coordinate pairs at once. For a
  small skew A, B⁺ ≈ B + BA − AB.
- **The damping** suppresses motion for pairs whose diagonal gap is below about 1% of s. At equal diagonals the
  motion vanishes.
- **Each correction** is the cubic Newton–Schulz (Björck–Bowie) polar iteration Y ↦ (3Y − YYᵀY)/2.
- **Columns are never reordered** by a warm refresh.

**QRRepair(C, Q):** cast C and Q to FP32, then

```
Z  = C·Q
π  = argsort_desc( Σ_i Q_ij Z_ij )        # = diag(QᵀCQ), the Rayleigh quotients
Q⁺ = R_b( qr(Z[:, π]).Q )
```

This is one warm-started orthogonal power (subspace) iteration step with Rayleigh-quotient sorting, not an exact
eigendecomposition. The baseline, **QR10**, applies it every 10 ages, and so does SOAP.

Over 10,000 calls our schedule performs 800 warm refreshes and 199 QR repairs per factor; QR10 performs 999 QR
refreshes.

### 1.3 Precision and execution

- **Matrices.** Equal-shaped matrices are batched; each keeps its own state. The 109M model has four groups of 12
  matrices each, with shapes 2304×768 (fused QKV), 768×768 (attention output), 3072×768 (fused SwiGLU in) and
  768×1536 (SwiGLU out). Both sides are preconditioned, giving 48 matrices and 96 factors.
- **Factor sizes.** Five groups of 12 factors have n = 768, and one group of 12 each has n = 1536, 2304 and 3072.
- **No shortcuts.** There is no blocking, no one-sided preconditioning, and no coverage change relative to SOAP.
- **Compilation.** Ordinary, warm and QR events are separate compiled regions (`torch.compile(fullgraph=True,
  dynamic=False)`, inductor option `emulate_precision_casts=True`), one per event per shape: 12 graphs, with no
  graph breaks. Each region includes packing params and grads, the matrix update, covariance accumulation,
  refresh and transport, and the parameter writes. The initial eigendecompositions are eager.
- **Backend.** PyTorch 2.11.0+cu128 on one RTX 5090 (Blackwell, 32 GB). FP32 matmul precision is "high", so FP32
  products may use TF32. BF16 reduced-precision reductions are disabled. Deterministic algorithms are on, with 4 CPU
  threads.
- **Hyperparameters.** All main runs use β1 = β2 = β_C = 0.95 (half-life ≈ 13.5 calls), ε = 1e-8 and weight decay
  0.05. The matrix peak LR is 1.5e-3, tuned on the FP32 QR10 baseline only. Gradients are clipped to global norm 1.
  The LR schedule is 200 calls of warmup, then cosine decay to 10% at 10,000 calls.
- **Auxiliary parameters.** Embeddings and norms use a separate fused AdamW.
- **Model and data.** 109M parameters: width 768, 12 layers, FF hidden 1536, tied embeddings, RoPE, QK-norm.
  OpenWebText, 32 × 1024-token rows per call, 10,000 calls.

### 1.4 The code

```python
STATE_KEYS = ('m', 'v', 'cl', 'cr', 'ql', 'qr')

def mm32(a, b):
    """BF16 tensor-core operands with FP32 accumulation/output on CUDA."""
    if a.dtype == torch.bfloat16:
        if a.is_cuda:
            return torch.bmm(a, b, out_dtype=torch.float32)
        return torch.bmm(a.double(), b.double()).float()
    return torch.bmm(a, b)


def covariance_ema(old, g, beta):
    if old.dtype == g.dtype:
        # BF16 persistent input/output; cuBLAS accumulates the product in FP32.
        return torch.baddbmm(old, g, g.transpose(-1, -2), beta=beta, alpha=1-beta)
    product = mm32(g, g.transpose(-1, -2))
    return (beta*old.float()+(1-beta)*product).to(old.dtype)


def warm_basis(cov, q, compute_dtype, basis_dtype):
    qb, cb = q.to(compute_dtype), cov.to(compute_dtype)
    b = mm32(torch.bmm(qb.transpose(-1,-2), cb), qb)
    b = .5*(b+b.transpose(-1,-2))
    diagonal = b.diagonal(dim1=-2,dim2=-1)
    scale = diagonal.abs().amax(-1,keepdim=True).clamp_min(1e-30)
    gap = (diagonal.unsqueeze(-1)-diagonal.unsqueeze(-2))/scale.unsqueeze(-1)
    a = -.5*(b/scale.unsqueeze(-1))*gap/(gap.square()+.01**2)
    bound = torch.linalg.vector_norm(a,dim=(-2,-1),keepdim=True)/math.sqrt(2)
    a = a*(.5/bound.clamp_min(1e-12)).clamp(max=1)
    y = (q.float()+mm32(qb,a.to(compute_dtype))).to(basis_dtype)
    for _ in range(2):
        if basis_dtype == torch.bfloat16:
            gram = mm32(y.transpose(-1,-2), y)
        else:
            # Optional precision island: measure the actual FP32 basis Gram.
            gram = torch.bmm(y.transpose(-1,-2), y)
        eye = torch.eye(y.shape[-1],device=y.device,dtype=gram.dtype)
        residual = gram-eye
        y = (y.float()-.5*mm32(y.to(compute_dtype),residual.to(compute_dtype))).to(basis_dtype)
    return y.contiguous()


def qr_basis(cov, q, basis_dtype):
    # FP32 decomposition island, with the nonredundant CQ/Rayleigh contraction.
    c, old = cov.float(), q.float()
    z = torch.bmm(c,old)
    order = (old*z).sum(-2).argsort(dim=-1,descending=True)
    ordered = z.gather(-1,order.unsqueeze(-2).expand_as(z))
    new = torch.linalg.qr(ordered).Q.to(basis_dtype).contiguous()
    return new,order


def transport_moment(m, old_l, old_r, new_l, new_r, compute_dtype):
    world = torch.bmm(old_l.to(compute_dtype),m.to(compute_dtype))
    world = torch.bmm(world,old_r.to(compute_dtype).transpose(-1,-2))
    projected = torch.bmm(new_l.to(compute_dtype).transpose(-1,-2),world)
    if m.dtype == torch.float32:
        return mm32(projected,new_r.to(compute_dtype)).float()
    return torch.bmm(projected,new_r.to(compute_dtype)).to(m.dtype)


def matrix_step(p, g, state, age, lr, *, beta1, beta2, cov_beta, eps,
                weight_decay, event, compute_dtype, basis_dtype):
    m,v,cl,cr,ql,qr = state
    gb = g.to(compute_dtype)
    projected = torch.bmm(ql.to(compute_dtype).transpose(-1,-2),gb)
    projected = torch.bmm(projected,qr.to(compute_dtype))
    mf = beta1*m.float()+(1-beta1)*projected.float()
    vf = beta2*v.float()+(1-beta2)*projected.float().square()
    u = (mf/(vf.sqrt()+eps)).to(compute_dtype)
    u = torch.bmm(ql.to(compute_dtype),u)
    u = mm32(u,qr.to(compute_dtype).transpose(-1,-2))
    correction = torch.sqrt(1-beta2**age)/(1-beta1**age)
    new_p = (p-lr*correction*u.float())*(1-lr*weight_decay)
    new_m,new_v = mf.to(m.dtype),vf.to(v.dtype)
    new_cl = covariance_ema(cl,gb,cov_beta)
    new_cr = covariance_ema(cr,gb.transpose(-1,-2),cov_beta)
    if event == 'warm':
        new_ql = warm_basis(new_cl,ql,compute_dtype,basis_dtype)
        new_qr = warm_basis(new_cr,qr,compute_dtype,basis_dtype)
    elif event == 'qr':
        new_ql,left_order = qr_basis(new_cl,ql,basis_dtype)
        new_qr,right_order = qr_basis(new_cr,qr,basis_dtype)
        new_v = new_v.gather(-2,left_order.unsqueeze(-1).expand_as(new_v))
        new_v = new_v.gather(-1,right_order.unsqueeze(-2).expand_as(new_v))
    else:
        new_ql,new_qr = ql,qr
    if event != 'ordinary':
        new_m = transport_moment(new_m,ql,qr,new_ql,new_qr,compute_dtype)
    return new_p,(new_m,new_v,new_cl,new_cr,new_ql,new_qr)


def group_step(params, grads, state, age, lr, *, event, **config):
    p,g = torch.stack(params),torch.stack(grads)
    new_p,new_state = matrix_step(p,g,state,age,lr,event=event,**config)
    for parameter,value in zip(params,new_p.unbind(0)):
        parameter.copy_(value)
    return new_state

# In the optimizer (production: compute_dtype = basis_dtype = state dtype = bfloat16):
#   event = 'cold' if not age else 'ordinary' if age % 10 else 'qr' if age % 50 == 0 else 'warm'
#   fn = torch.compile(<ordinary|warm|qr>_group, fullgraph=True, dynamic=False,
#                      options={'emulate_precision_casts': True})   # one per event per shape
#   new_state = fn(params, grads, state, age_tensor, lr_tensor, beta1=.95, beta2=.95, cov_beta=.95, ...)
# Cold start: M = V = 0, C from the first gradient, Q = eigh(sym(float(C))).eigenvectors.flip(-1) -> BF16 (eager).
```

---

## 2. Measurements

### 2.1 Whole-optimizer timing in training (109M model, 48 matrices plus auxiliary AdamW)

These are means of one whole optimizer call, by event class. They are synchronized host wall times, from seed 205,
in a shared timing window.

| | ordinary event | warm event | QR event |
|---|---|---|---|
| our variant (BF16 state, compiled) | 10.14 ms | 61.61 ms | 601.18 ms |
| FP32 SOAP reference (QR10, eager, FP32 state) | 34.44 ms | — | 642.37 ms |

- **Event mix.** 90% of calls are ordinary in both arms. Our variant has 8% warm and 2% QR events; the reference
  has 10% QR events.
- **Optimizer mean.** 26.08 ms/call for our variant, against 95.23 ms/call for the reference.
- **Whole step.** 309.5 ms/call for our variant and 380.5 ms/call for the reference (seed 205, calls 101–10000).
  Forward and backward take the rest; the model runs eagerly with BF16 autocast and FlashAttention.

A microbenchmark reproduces these event means to within about 1% (10.13 / 61.09 / 609.24 ms).

Isolated, compiled per-function primitives, summed over the 109M factor set (batch 12 per group):
- **Warm refreshes:** 49.84 ms for all 96 factors, plus 5.06 ms for the 4 transports. Measured warm-event
  increment over ordinary: 51.47 ms.
  - Run eagerly, the warm refreshes take 81.40 ms. So compiled warm is about 1.7× faster than eager.
- **QR refreshes:** 593.74 ms for all 96 factors, plus 5.06 ms of transport and 0.38 ms of V permutation. Measured
  QR-event increment: 591.04 ms.
- **QR primitive per group** (batch 12): 30.8 ms at n = 768, 81.4 at 1536, 143.5 at 2304 and 219.4 at 3072.
- **Transport per group** (batch 12, compiled): 1.47 ms for 2304×768, 0.43 for 768×768, 2.32 for 3072×768 and
  0.84 for 768×1536.

### 2.2 Refresh cost per factor vs n (synthetic inputs, compiled)

Inputs are built as follows:
- **C** is an EMA (β = 0.95, 40 steps) of G Gᵀ, stored in BF16, with G = U·diag(λ^½)·X, power-law λ_i ∝ 1/i and X
  Gaussian of size n × n/4.
- **Q** is the QR of U(I + A), stored in BF16, with A a random skew matrix whose entries are about 0.2/√n.
- The input r_C (defined in §2.4) is about 0.20.

Timings are CUDA-event medians in ms per factor (the call time divided by the batch).

| n | warm, b12 | QR, b12 | warm, b1 | QR, b1 | eigh, b12 | Cholesky-QR, b12 | warm/QR, b12 |
|---|---|---|---|---|---|---|---|
| 256 | 0.067 | 0.62 | 0.79 | 0.89 | 4.5 | 0.068 | 0.108 |
| 512 | 0.067 | 1.52 | 0.80 | 1.82 | 11.8 | 0.104 | 0.044 |
| 768 | 0.069 | 2.56 | 0.84 | 2.91 | 6.5 | 0.171 | 0.027 |
| 1024 | 0.102 | 3.78 | 0.84 | 4.06 | 8.7 | 0.280 | 0.027 |
| 1536 | 0.331 | 6.74 | 0.80 | 6.93 | 14.7 | 0.618 | 0.049 |
| 2048 | 0.731 | 9.78 | 0.92 | 9.91 | 21.4 | 1.16 | 0.075 |
| 2304 | 1.02 | 11.8 | 1.03 | 11.9 | 27.3 | 1.50 | 0.086 |
| 3072 | 2.46 | 18.1 | 2.29 | 18.2 | 44.4 | 3.03 | 0.136 |
| 4096 | 5.54 | 27.7 | 5.77 | 27.9 | 77.0 | 6.23 | 0.200 |
| 6144 | 18.7 | 62.3 | 15.2 | 62.5 | 227 | 19.0 (all NaN) | 0.300 |
| 8192 | 45.5 | 112 | 37.2 | 112 | 575 | 42.2 (all NaN) | 0.406 |

- **eigh** is FP32 `torch.linalg.eigh`, as at initialization.
- **Cholesky-QR** uses the same Z = CQ and sort as QR, then R = chol(ZᵀZ) and Q⁺ = Z R⁻¹, in FP32 with no shift.
  - It fails (NaN) for every factor at n ≥ 6144 on these inputs; κ(ZᵀZ) ≈ κ(C)².
  - Its BF16-stored orthogonality error grows from 0.003 (n = 256) to 0.034 (n = 4096), against 0.0024 for QR.
- **Batching.** The QR per-factor time is the same at batch 1 and batch 12; the batched call loops over the batch.
  Below n ≈ 1024, a batch-12 warm call costs about 0.8 ms regardless of n, so it is latency-bound.
- **Single-refresh quality on these inputs** (input r_C ≈ 0.20):
  - one warm refresh leaves r_C ≈ 0.150 at n = 768 and 0.128 at n = 8192;
  - one QR leaves r_C ≈ 0.025 at n = 768 and 0.010 at n = 8192;
  - normalized Gram error ‖QᵀQ − I‖_F/√n after storage is 0.0018 for warm and 0.0024 for QR.

### 2.3 Loss (109M, validation cross-entropy)

**Full horizon, 10,000 calls.** "FP32 QR10" is the FP32 SOAP reference. "Ours" is warm10/QR50 with BF16 state.

| seed | FP32 QR10 | ours | difference |
|---|---|---|---|
| 204 | 3.411152 | 3.422389 | +0.011237 |
| 205 | 3.409272 | 3.419354 | +0.010082 |

- **Call 500.** The gap is much larger: +0.124 (seed 204) and +0.118 (seed 205).
- **From call 2000 on.** The gap stays at about +0.007 to +0.012 and does not keep shrinking.
- **FP32-state warm arm.** A pilot warm arm with FP32 state shows a similar call-500 gap (+0.118). The early gap is
  therefore not specific to BF16 storage.

**Short controls (seed 204, the full 10k LR schedule, first 2,000 calls; timing over calls 101–2000):**

| policy | CE at 500 | CE at 2000 | step ms | optimizer ms |
|---|---|---|---|---|
| FP32 QR10 (pilot; separate launch, FP32 state, eager) | 4.877601 | 3.985905 | 382.8 | 96.36 |
| BF16 QR10 | pending | pending | | |
| ours, warm10/QR50 (prefix of the main run, earlier launch) | 5.001514 | 3.993939 | 314.1 | 26.42 |
| BF16 QR20 (QR every 20 ages, no warm) | 4.936852 | 3.990383 | 329.6 | 40.46 |
| BF16 QR50 (QR every 50 ages, no warm) | 5.018046 | 3.998679 | 313.6 | 22.86 |

Caveats:
- The FP32 QR10 row is not a matched BF16 control, and the matched BF16 QR10 run is still pending.
- Timing compares across launches.
- These are panels on one seed, not independent replicates.

Observations:
- Warm beats QR50 at all seven non-initial evaluations, and QR20 beats warm at six.
- Measured against FP32 QR10, the warm refreshes close about 12% of the QR50–QR10 gap at call 500 and about 37% at
  call 2000.

### 2.4 Diagonalization quality in training

Define

r_C = ‖offdiag(QᵀCQ)‖_F / ‖QᵀCQ‖_F,

evaluated with FP32 products at "highest" precision.

- **Trained values.** At seed 205, ages 1990 and 9990, four warm refreshes after the previous QR:
  - our mean over the 96 factors is 0.394 and 0.380, with maxima of 0.965 and 0.975;
  - FP32 QR10 has 0.201 and 0.214.
- **Toy drifting covariance.** n = 128, power-law spectrum, a random skew rotation of Frobenius size 0.01·√2 per
  step, 32 samples per step, β_C = 0.95. Mean r_C over steps 100–1000 is 0.075 for QR10, 0.093 for warm10/QR50 and
  0.125 for QR50.

### 2.5 Known constraints and facts

1. **No BF16 decompositions.** CUDA QR (`geqrf`) and the symmetric eigensolver have no BF16 kernels in this PyTorch
   build, so QR and eigh run in FP32.
2. **"Power step plus Newton–Schulz/polar" is a no-op.** For orthogonal Q with B = QᵀCQ symmetric positive
   definite, CQ = Q·B is already a polar decomposition with orthogonal factor Q, so polar(CQ) = Q. Any QR
   substitute has to break this symmetry (QR does it through the column order).
3. **Equal diagonals stall the warm step.** With Q = I and C = [[1, 0.5], [0.5, 1]], all h_ij = 0 and the warm step
   does nothing, despite the nonzero off-diagonal. QR repair is the recovery path.
4. **History is approximate.** V is permuted at QR and left unchanged at warm refreshes; neither rotates a full
   second moment. M transport by transposes is approximate when the stored bases are not orthogonal.
5. **Rounding floor.** BF16 rounding of a freshly computed FP32-QR basis already gives a normalized Gram error of
   about 0.0024.

### 2.6 The test harness available for your candidates

Your "cheap tests" should fit one of these two tools. If a candidate cannot be tested in them, say so and say what
would be needed (for example, a short training run).

**(i) Open-loop covariance replay.** We trained 1-layer versions of the model (widths 256, 512, 1024 and 2048) for
1,000 ages with the FP32 QR10 reference, and saved every factor's C_L and C_R, in BF16, at age 0 and every 10th
age. Refresh schedules are then chained offline over these snapshots: QR10, warm10/QR50, QR20, QR50, eigh10,
Cholesky-QR10, and any new refresh function. Because the covariances do not depend on the replayed bases, this is
open loop.

Metrics before and after every refresh:
- r_C;
- a diagonally normalized r_C (each direction weighted by its inverse variance, above a floor);
- the RMS off-diagonal correlation;
- the top-k subspace overlap with C's true eigenvectors;
- the normalized Gram error.

The harness has **no per-step gradients, no M or V and no loss**. Only the C snapshots every 10 ages exist.

**(ii) Microbenchmark.** It measures ms per factor vs n (256–8192) at batch 1 and 12, compiled and eager, with the
backend settings above. It also times the event regions on the 109M shapes.

---

## 3. What I want from you

### Question A: speed

How can the refresh events, and the method as a whole, be made faster? Cover both:
- **(A-eng)** engineering at fixed mathematics: compile, kernel, fusion, scheduling, batching and dtype choices in
  PyTorch 2.11 on a Blackwell consumer GPU;
- **(A-eq)** equation-level changes that alter the arithmetic but aim to keep quality.

Consider every event type (ordinary step, covariance accumulation, warm refresh, QR repair, transport) and both
regimes: the 109M sizes (n ≤ 3072, where some calls are latency-bound) and larger factors (n = 4096–8192, where the
warm/QR cost ratio rises). Use the measurements in §2 to say where the time actually goes before proposing fixes.

### Question B: tracking quality

How can the warm refresh match the QR refresh more closely, so the loss gets closer to QR10's, at modest cost? The
target is **the QR10 trajectory**: one warm-started power step plus QR every 10 ages, with trained r_C ≈ 0.20. It is
not an exact eigendecomposition. "Modest cost" means at most about 2× the current warm cost at n = 768–3072, and
well below QR at every n.

Address these as well:
- whether r_C is the right proxy for what matters to the loss, and what you would measure instead;
- the role of the call-500 gap;
- the warm step's treatment of V and of column order.

### Pitfalls

What is wrong, fragile or misleading in the current design or its measurements? Consider numerics, BF16, the
damping and cap, the ordering of direction and refresh, transport, the event-class timing method, and anything
else you see.

### Format for every candidate

Give each candidate an ID (A1, A2, … and B1, B2, …) and a name, then:

1. **Exact update:** equations or a short batched PyTorch sketch.
2. **Cost:** the number of n×n (or m×n) products per factor, the other kernels, and which ones are FP32 vs BF16. Show
   the count.
3. **Expected effect:** quantitative where you can (ms saved at which n; r_C or loss change). Say how confident you
   are.
4. **Failure modes:** when it breaks, including numerical breakdown.
5. **Interaction with BF16 storage and β_C = 0.95.**
6. **Cheap test:** in the replay harness or the microbenchmark (§2.6), with the metric and the outcome that would
   reject it. If neither tool can test it, say so and name the smallest test that could.

Then finish with:
- **Your ranking:** your top 5 for A and your top 5 for B, each with one line of reasoning.
- **Your top 3 pitfalls.**
- **What you would want measured** before refining further: up to 5 items.

Write for an expert reader. Be concrete, prefer formulas to adjectives, and keep A and B clearly separated.
