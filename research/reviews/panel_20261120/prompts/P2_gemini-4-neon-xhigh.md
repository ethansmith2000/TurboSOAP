# Stage 2 of a review panel: reveal, cross-reading and ranking

This prompt contains six documents followed by the stage-2 task. Read the documents first; the task is at the end. The ground rules inside Document 1 (no web, no tools) applied to stage 1 only; stage 2 has its own rules, stated in the task.


---

# Document 1: the stage-1 brief (verbatim)

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

---

# Document 2: your stage-1 answer (verbatim)

### Ground-Rules Note on Tags
Throughout this response, **`[certain]`** marks derivations or direct consequences of the numbers/code you gave, **`[recalled]`** marks hardware/library details recalled from memory, **`[verify]`** marks literature or hardware-spec claims to check, and **`[speculation]`** marks untested hypotheses.

---

# Question A: Speed

## 0. Where the Time Actually Goes (Audit of §2)

### 109M model (mean optimizer time = **26.08 ms/call**)
Decomposing $0.90(10.14) + 0.08(61.61) + 0.02(601.18) = 26.08\text{ ms/call}$ into work performed on every step plus refresh increments **`[certain]`**:

| Component | Frequency | Event cost | Mean ms/call | Share of optimizer |
|---|---:|---:|---:|---:|
| Ordinary step (all 48 matrices + aux AdamW) | 100% | 10.14 ms | **10.14 ms** | **38.9%** |
| QR refresh increment (`qr_basis` 593.7 + `transport` 5.1 + `V` perm 0.4) | 2% | 591.04 ms | **11.82 ms** | **45.3%** |
| Warm refresh increment (`warm_basis` 49.8 + `transport` 5.1) | 8% | 51.47 ms | **4.12 ms** | **15.8%** |

Two underappreciated facts emerge immediately from exact FLOP counts **`[certain]`**:

1. **QR repair is still the #1 bottleneck (45.3%) despite running on only 2% of calls**, and its GPU utilization is dismal. Each `qr_basis` call does $2n^3$ ($Z=CQ$) + $\frac{4}{3}n^3$ (`geqrf`) + $\frac{4}{3}n^3$ (`orgqr`) $= \frac{14}{3}n^3 \approx 4.67n^3$ FLOPs — **3× fewer FLOPs than `warm_basis`** ($7 \times 2n^3 = 14n^3$ FLOPs). Yet on the RTX 5090, `qr` achieves only **0.83 TFLOP/s** at $n=768$ ($2.11\text{ GFLOP}/2.56\text{ ms}$), **7.5 TFLOP/s** at $n=3072$, and **22.9 TFLOP/s** at $n=8192$, because `torch.linalg.qr` loops serially over the batch of 12 and is bottlenecked by panel factorizations. As a result, the 60 small $n=768$ factors consume **25.9%** of QR time (154 ms) despite representing only **4.8%** of $\sum n^3$.
2. **The ordinary step is the #2 bottleneck (38.9%) and is ~75% GEMM-bound.** Across the 4 groups ($b=12$, shapes $2304\times 768$, $768\times 768$, $3072\times 768$, $768\times 1536$), $\sum 2bm^2n = 304.41\text{ GFLOP}$ and $\sum 2bmn^2 = 130.46\text{ GFLOP}$, so each projection pair ($\bar{G}$, $U$, or covariance $G G^\top + G^\top G$) is **434.87 GFLOP** — **1,304.6 GFLOP** total per ordinary call. From §2.1, `transport_moment` executes exactly two projection pairs (**869.73 GFLOP**) in **5.06 ms** (**171.9 TFLOP/s**). Scaling by $1.5\times$ shows the 6 ordinary GEMMs take **~7.6 ms** (~5.1 ms for $\bar{G}$ and $U$, ~2.5 ms for $C_L, C_R$), leaving **~2.5 ms** for `torch.stack`/`copy_`, Adam elementwise, $C$ HBM traffic (962 MB read+write), and auxiliary AdamW on the ~38.6M tied embeddings.
3. **Why `warm/QR` rises from 0.027 ($n=768$) to 0.406 ($n=8192$):**
   - At $n \le 1024$, `warm_basis` hits a **~0.80 ms/call launch-latency floor** regardless of batch ($12 \times 0.067 \approx 0.80\text{ ms}$), while `QR` is severely panel-bound (~0.8 TFLOP/s).
   - At $n = 6144\text{–}8192$, `warm_basis` ($14n^3$ FLOPs) is already pinned at the RTX 5090's BF16-with-FP32-accumulation Tensor Core ceiling (~209.5 TFLOP/s peak **`[recalled][verify]`**; `warm, b1` achieves **213.6 TFLOP/s** at $n=6144$ and **206.9 TFLOP/s** at $n=8192$!). Meanwhile `QR`'s blocked trailing updates reach ~23 TFLOP/s while doing **3× fewer FLOPs**, AND `warm, b12` suffers a **22–23% per-factor regression vs `b1`** ($45.5$ vs $37.2\text{ ms}$ at $n=8192$).
   - **Critical crossover **`[certain]`**:** Per 100 steps, `warm10/QR50` costs $8T_{\text{warm}} + 2T_{\text{QR}}$, while `QR20` costs $5T_{\text{QR}}$. Thus `warm10/QR50` becomes **slower than `QR20`** once $T_{\text{warm}}/T_{\text{QR}} \ge 3/8 = 0.375$ — which is **already crossed at $n=8192$** ($588\text{ ms}$ vs $560\text{ ms}$ per 100 steps)!

---

## (A-eng) Engineering Speedups at Fixed Mathematics

### A1. Multi-Stream Parallel `qr_basis` + `mm32(cov, q)` Inside `qr_basis`

1. **Exact update:**
   - Replace `torch.bmm(cov.float(), q.float())` with `mm32(cov, q)`. Since `cov` and `q` are BF16 (7-bit mantissa), their exact products require $\le 16$ mantissa bits, so BF16 Tensor Core MMA with FP32 accumulation (`mm32`) is **mathematically exact** relative to FP32/TF32 matmul of BF16-cast inputs **`[certain]`**.
   - Unbind the `ordered` tensor along the batch dimension and dispatch `torch.linalg.qr` across a fixed pool of $K=4\text{–}8$ CUDA streams (with `torch.cuda.Stream`), synchronizing via events before restacking. (Because `qr_basis` is an isolated FP32 island, either run the multi-stream QR outside the compiled region or wrap it in a custom op.)
2. **Cost:** Identical arithmetic per factor: $1\times$ $n^3$ BF16$\to$FP32 GEMM ($2n^3$, replacing $2n^3$ TF32) + $\frac{8}{3}n^3$ FP32 (`geqrf`+`orgqr`). Cuts `Z = CQ` input bandwidth by $2\times$ and doubles its tensor-core ceiling (~210 vs ~105 TFLOP/s on RTX 5090 **`[recalled][verify]`**).
3. **Expected effect:**
   - `mm32(cov, q)` alone saves ~4 ms on $n=3072$ and ~6 ms across all 96 factors per QR event.
   - Multi-stream QR attacks the <1% SM utilization at $n=768$ (154 ms) and $n=1536$ (81 ms). If cuSOLVER handles run concurrently across streams without internal mutex serialization, expect **2.5×–4×** speedup on $n=768$ and **1.5×–2.5×** on $n=1536$ → **~120–170 ms saved per QR event (~2.4–3.4 ms/call)**. Minimal benefit at $n \ge 4096$ where single QR fills more SMs. Confidence: **medium-high** (depends on whether cuSOLVER + `deterministic_algorithms(True)` serializes stream launches in PyTorch 2.11 / cu128 **`[recalled][verify]`**).
4. **Failure modes:** Extra workspace VRAM ($K\times$ single-QR workspace, tiny at $n \le 3072$); potential cuSOLVER host-lock contention under `use_deterministic_algorithms(True)`; breaks `fullgraph=True` unless wrapped in a `torch.library.custom_op`.
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Exact; bit-identical (up to `mm32` vs TF32 accumulation order).
6. **Cheap test:** Microbenchmark (§2.6(ii)): time `qr_basis` for `b=12` at $n \in \{768, 1536, 2304, 3072\}$ with $K \in \{1, 2, 4, 8\}$ streams. **Reject** if speedup at $n=768$ is $< 1.5\times$ (indicates cuSOLVER serialization).

---

### A2. Contiguous Group Parameter/Grad Buffers + Cross-Group $n=768$ Batching + CUDA Graphs

1. **Exact update:**
   - Allocate each shape group's 12 weight matrices as views into one contiguous `(12, m, n)` FP32 parameter tensor (and accumulate/pack gradients once via `_foreach` or a contiguous grad bucket), eliminating `torch.stack(params)`, `torch.stack(grads)`, and the 48 `parameter.copy_` loops on **100%** of calls.
   - Pack all 60 $n=768$ factors (5 groups of 12) into a single `(60, 768, 768)` call during `warm_basis` (keeping separate `m, v` groups only for $m\times n$ ops).
   - Capture the static ordinary and warm compiled regions into **CUDA Graphs** (`mode="reduce-overhead"` or manual `torch.cuda.CUDAGraph` with static scalar `age`/`lr` device tensors, which you already pass as tensors).
2. **Cost:** Identical GEMM counts. Eliminates ~283 MB param stack + ~283 MB param unpack + ~142–283 MB grad stack (~700–850 MB HBM traffic) on every ordinary step, and collapses 5 separate 0.80 ms kernel-launch latency floors in `warm_basis` into 1.
3. **Expected effect:**
   - Ordinary step: eliminates ~0.6–1.0 ms/call of pack/unpack bandwidth + launch overhead (~6–10% of ordinary step, **~0.8 ms/call** across 100% of calls).
   - Warm event: merging the five $n=768$ groups + CUDA graphs eliminates ~2.5–3.5 ms of latency overhead per warm event (~0.25 ms/call; huge relative win on small $n \le 1024$ where `warm` is currently 100% latency-bound at 0.80 ms/call). Confidence: **high**.
4. **Failure modes:** Contiguous parameter backing requires care with `state_dict` / weight tying (embeddings are separate anyway); CUDA Graphs fail on `qr` if cuSOLVER performs host syncs (keep QR eager/non-graphed) and hold static activation/intermediate pools (~200–500 MB on 109M, fine in 32 GB, problematic at $n=8192$).
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Exact mathematics; no numerical change.
6. **Cheap test:** Microbenchmark (§2.6(ii)): compare `(60, 768, 768)` vs $5 \times (12, 768, 768)$ `warm_basis`, and CUDA-graphed vs ungraphed ordinary/warm event regions on 109M shapes. **Reject** individual sub-parts if ordinary saves $< 0.4\text{ ms}$ or 60-batch warm saves $< 1.5\text{ ms}$.

---

### A3. `warm_basis` Memory-Traffic Reduction (Scalar-Pull + Diagonal Sub) & Large-$n$ Micro-Batching

1. **Exact update:**
   - **Scalar pull:** Use $Q + Q(aA) = Q + a(QA)$. Compute `a_raw` in FP32, fuse writing `a_bf16 = a_raw.to(bf16)` with the Frobenius reduction `bound = ||a_raw||_F / sqrt(2)` (or reduce from `a_bf16`), run `prod = mm32(qb, a_bf16)`, and fold the scalar `a = min(1, 0.5 / bound)` into `y = (q.float() + a * prod).to(bf16)`. Because BF16 has FP32's full 8-bit exponent range ($10^{\pm 38}$), deferring scalar multiplication across BF16 rounding changes `aA` by $\le 0.5$ ULP **`[certain]`**.
   - **No `eye` materialization:** Replace `gram - torch.eye(...)` with index-fused subtraction `gram - (i == j)` or in-place `gram.diagonal(dim1=-2, dim2=-1).sub_(1.0)`.
   - **Micro-batching for $n \ge 3072$:** Loop `warm_basis` over sub-batches of size $b_{\text{chunk}} \in \{1, 2, 4\}$ when $n \ge 3072$ (keeping full $b=12$ for $n \le 2304$).
2. **Cost:** Same 7 BF16 GEMMs ($14n^3$), but eliminates 1 full FP32 write + 1 full FP32 read of $A$ ($2 \cdot 4bn^2$ bytes = **6.44 GB** at $b=12, n=8192$), eliminates `eye` read/writes, and cuts peak transient VRAM by $3\times\text{–}12\times$ (from ~13–16 GB to ~1.3–4 GB at $n=8192$).
3. **Expected effect:**
   - Directly recovers the measured `b12` vs `b1` regression in §2.2: **7%** at $n=3072$ ($2.46 \to 2.29\text{ ms/factor}$, ~2 ms saved on 109M warm), **23%** at $n=6144$ ($18.7 \to 15.2\text{ ms/factor}$), and **22%** at $n=8192$ ($45.5 \to 37.2\text{ ms/factor}$), plus another ~1–2 ms/factor at $n=8192$ from eliminating the 6.4 GB FP32 `a` round-trip. Confidence: **high** (the `b1` vs `b12` gap is already directly measured in §2.2; note that if part of `b12`'s slowdown is sustained 575W thermal/power throttling over a 546 ms burst vs 37 ms burst **`[speculation]`**, micro-batching still recovers the cuBLAS/VRAM/HBM portion).
4. **Failure modes:** If `a_raw` ever overflows $3.4 \times 10^{38}$ (impossible with `clamp_min(1e-30)` and `0.01^2` damping, where $|a_{\text{raw}}| \le 50 \|B\|_\infty / s \sim O(50)$).
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Negligible ($\le 0.5$ ULP on `aA`); identical convergence.
6. **Cheap test:** Microbenchmark (§2.6(ii)) at $n \in \{3072, 4096, 6144, 8192\}$ for $b_{\text{chunk}} \in \{1, 2, 4, 12\}$ with locked vs unlocked GPU clocks (`nvidia-smi -lgc` to isolate thermal throttling from kernel/memory effects). **Reject** chunking if `b12` matches `b1` once clocks are locked and `a` traffic is fused.

---

### A4. Custom Upper-Triangular Batched `SYRK` (Triton/CUTLASS) for $G_b G_b^\top$, $G_b^\top G_b$, and $Y_j^\top Y_j$

1. **Exact update:** Every call executes `G_b @ G_b.T` and `G_b.T @ G_b` (symmetric), and every `warm_basis` executes `Y_j.T @ Y_j` twice (symmetric). A tiled Triton kernel (using Blackwell `tl.dot` BF16$\to$FP32) skips tiles with `pid_m > pid_n`, fuses `beta * C_old + (1 - beta) * acc` (or `acc - (i == j)` for `gram`), and writes both tile `(m, n)` and its transpose `(n, m)` in the epilogue.
2. **Cost:** For an $n\times n$ symmetric output with $T = n/B_M$ tiles per side, computes $\frac{T(T+1)}{2}$ tiles instead of $T^2$ ($\approx 50\%$ FLOPs for $T \ge 12$, e.g. 48% savings at $n=3072, B_M=128$).
   - Ordinary step: cuts covariance GEMMs from **434.9 $\to$ ~225 GFLOP** (saves ~210 GFLOP on 100% of calls!).
   - Warm refresh: cuts the two `Y^T Y` GEMMs from $4n^3 \to \sim 2.1n^3$ (saves **~14% of `warm_basis`**, i.e. 1 of 7 GEMMs, at every compute-bound $n$, AND cuts `Z^T Z` in Cholesky-QR by ~50%).
   - Bonus: guarantees **exact bitwise symmetry** of $C_L, C_R$.
3. **Expected effect:** Saves **~1.1–1.3 ms/call** on ordinary steps (100% of calls) and **~6 ms** per warm event (~0.5 ms/call) → **~1.6–1.8 ms/call** on 109M, and **~14%** on `warm_basis` at $n=4096\text{–}8192$ (assuming the Triton kernel reaches $\ge 85\%$ of cuBLAS BF16 efficiency; if it only reaches 75%, net speedup on those GEMMs is $1.5\times$ instead of $1.9\times$). Confidence: **medium-high** (engineering effort is non-trivial to match Blackwell cuBLASLt MMA schedules).
4. **Failure modes:** Under-tuned Triton kernels on Blackwell (`sm_120`) can lose 20–40% of cuBLASLt peak, eroding half the 2× algorithmic win; tile-transpose epilogue writes to non-contiguous lower-triangle addresses (needs shared-memory transpose before global store).
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Exact same math, strictly cleaner because $C_L, C_R$ are guaranteed exactly symmetric.
6. **Cheap test:** Microbenchmark (§2.6(ii)): standalone batched Triton `SYRK` vs `torch.baddbmm` on `(12, 3072, 768)` and `(12, 8192, 8192)`. **Reject** if wall time is $> 0.70\times$ `torch.baddbmm`.

---

## (A-eq) Equation-Level Speedups

### A5. Column-Equilibrated Shifted Cholesky-QR2 (`sCholQR2`) Replacing `torch.linalg.qr`

1. **Exact update:** Replaces `qr_basis` with a numerically stabilized 2-pass BLAS-3 QR (Fukaya et al. 2020 **`[recalled][verify]`**), using `mm32` for $Z = CQ$, a tiny eigenvector-preserving ridge $\mu I$, column equilibration, and **true FP32** (TF32 disabled!) for the Gram products:
   ```python
   # Z = (C + mu*I) @ Q;  C + mu*I has identical eigenvectors to C, kappa <= 1/tau
   z = mm32(cov, q)                                              # exact BF16->FP32
   d = (q.float() * z).sum(-2)                                   # Rayleigh quotients
   mu = tau * d.abs().amax(-1, keepdim=True).clamp_min(1e-30)    # e.g. tau = 3e-4 or 1e-3
   z = z + mu.unsqueeze(-1) * q.float()
   order = d.argsort(dim=-1, descending=True)
   z = z.gather(-1, order.unsqueeze(-2).expand_as(z))
   # Column equilibration: qr(Z D^{-1}).Q == qr(Z).Q identically, and ||Z_hat||_F^2 == n
   z = z / z.norm(dim=-2, keepdim=True).clamp_min(1e-30)
   # Pass 1 (shifted) + Pass 2 (unshifted) in TRUE FP32 (TF32 OFF!)
   with torch.backends.cuda.sdp_kernel... # disable TF32 via fp32_precision="ieee"
       g0 = torch.bmm(z.transpose(-1, -2), z)
       g0.diagonal(dim1=-2, dim2=-1).add_(s1)                    # s1 = 1e-6 * n
       r0 = torch.linalg.cholesky(g0, upper=True)
       q1 = torch.linalg.solve_triangular(r0, z, upper=True, left=False)
       g1 = torch.bmm(q1.transpose(-1, -2), q1)
       r1 = torch.linalg.cholesky(g1, upper=True)
       q_new = torch.linalg.solve_triangular(r1, q1, upper=True, left=False)
   return q_new.to(basis_dtype).contiguous(), order
   ```
   **Algebraic & stability proof **`[certain]`**:**
   - Since $D = \operatorname{diag}(\|z_j\|_2) > 0$ is positive diagonal, if $\hat{Z} = Z D^{-1} = Q R$, then $Z = Q(R D)$ with $R D$ upper triangular and positive diagonal, so $\operatorname{qr}(\hat{Z}).Q = \operatorname{qr}(Z).Q$ identically.
   - Because $\|\hat{Z}\|_F^2 = n$, $\kappa(G_0) \le \frac{\|\hat{Z}\|_2^2 + s_1}{s_1} \le \frac{n + 10^{-6} n}{10^{-6} n} \approx 10^6 \ll u_{\text{FP32}}^{-1} \approx 1.67 \times 10^7$, so Pass 1 Cholesky cannot fail.
   - After Pass 1, $\sigma_i(Q_1) = \sigma_i(\hat{Z}) / \sqrt{\sigma_i(\hat{Z})^2 + s_1}$, so $\kappa(Q_1) \le \sqrt{1 + s_1/\sigma_{\min}(\hat{Z})^2}$. Since $C + \mu I$ bounds $\kappa(C+\mu I) \le 1/\tau$ and column equilibration further removes the pure-eigenvalue scaling (leaving only basis-misalignment factor $\kappa(\hat{Z}) \sim r_C \kappa(C+\mu I)$), for $\tau = 10^{-3}$ ($\sigma_{\min}(\hat{Z}) \gtrsim 10^{-3}$) and $n \le 4096$ ($s_1 \le 4 \times 10^{-3}$), $\kappa(Q_1) \le 65$, giving Pass 2 orthogonality error $O(u_{\text{FP32}} \kappa(Q_1)^2) \le 3 \times 10^{-4} \ll 2.4 \times 10^{-3}$ (BF16 floor).
   - **Crucial detail:** Your §2.2 naive Cholesky-QR failed because of (i) no column equilibration ($\kappa(Z^\top Z) = \kappa(C)^2$), (ii) no shift / 2nd pass, and (iii) **TF32 truncation on `Z.T @ Z`** (see Pitfall 2). On consumer GeForce Blackwell (RTX 5090), true FP32 CUDA-core peak (~105 TFLOP/s) equals TF32 Tensor Core peak (~105 TFLOP/s) **`[recalled][verify]`**, so disabling TF32 on `g0, g1` costs very little (~15–25% cuBLAS efficiency difference) while dropping $u$ from $5 \times 10^{-4}$ to $6 \times 10^{-8}$.
2. **Cost:** Per factor: $1\times$ $n^3$ BF16$\to$FP32 GEMM ($2n^3$) + $2\times$ [$2n^3$ FP32 GEMM (or $n^3$ SYRK) + $\frac{1}{3}n^3$ FP32 `cholesky` + $n^3$ FP32 `trsm`] $= \mathbf{2n^3\text{ BF16} + 6.67n^3\text{ FP32}}$ (or $4.67n^3$ FP32 with SYRK). All ops are batched BLAS-3 with no serial `geqrf` panel bottlenecks.
3. **Expected effect:** From §2.2, 1-pass CholQR summed over the 96 factors is $60(0.171) + 12(0.618 + 1.50 + 3.03) = \mathbf{72.04\text{ ms}}$ (vs **593.74 ms** for `qr`). Replacing TF32 $Z=CQ$ with BF16 `mm32` and adding Pass 2 yields **~115–140 ms** total across 96 factors (**4.2×–5.2× faster QR event!**), cutting mean optimizer time by **~9.0–9.5 ms/call** (**~35% of the entire optimizer**!). At $n=8192$, expect **~60–72 ms/factor** vs 112 ms for `qr` (**1.6×–1.9× faster**, ~2× with SYRK). Confidence: **high**.
4. **Failure modes:** If BF16 rounding makes $\operatorname{sym}(C)$ have a negative eigenvalue $-\nu$ with $\nu \approx \mu$, $C + \mu I$ can become near-singular. At $n=8192$ with $\sigma_{\min}(\hat{Z}) < 10^{-3}$, Pass 2 may need a tiny safety shift $s_2 = 10^{-8}n$ or a 3rd pass (`sCholQR3`) if $\kappa(Q_1) > 200$. Setting $\tau$ too large (e.g. $10^{-2}$) slows power-iteration contraction on tail eigenvalues $\lambda_i < \tau \lambda_{\max}$.
5. **Interaction with BF16 storage & $\beta_C=0.95$:** BF16-induced negative eigenvalues of $C$ are bounded by $\|\Delta C\|_2 \le 2^{-8}\|C\|_F$, so $\tau \approx 10^{-3}$ safely restores strict positive definiteness and prevents rank-deficiency crashes at early calls (when $C_L$ for $3072\times 768$ has rank $\le 768$).
6. **Cheap test:** Both tools! (i) Microbenchmark at $n = 256\text{–}8192$ ($b=1, 12$) verifying zero NaNs, Gram error $\le 0.0025$ after BF16 cast, and speedup vs `qr`. (ii) Replay harness comparing `sCholQR2` ($\tau \in \{10^{-4}, 3\times 10^{-4}, 10^{-3}\}$) vs `QR10` on $r_C$, RMS correlation, and Gram error. **Reject** if Gram error exceeds $0.003$ at $n=8192$ or replay $r_C$ degrades by $> 0.005$ vs `QR10`.

---

### A6. Degree-4 Matrix Exponential $R_4(\tilde{A})$ + Direct Local Moment Transport $M^+ = R_L^\top M R_R$

1. **Exact update:** Let $\tilde{A} = aA$ ($\tilde{A}^\top = -\tilde{A}$). Replace $Y_0 = Q(I+\tilde{A})$ + two Newton–Schulz iterations (5 GEMMs) with Paterson–Stockmeyer degree-4 Taylor exponential (3 GEMMs including $Q R$), and reuse $R_L, R_R$ for moment transport:
   $$S = \mathcal{F}(\tilde{A}_b, \tilde{A}_b), \quad R = R_b\!\left(I + \tilde{A} + \tfrac{1}{2}S + \mathcal{F}\!\left(R_b\!\left(\tfrac{1}{6}\tilde{A} + \tfrac{1}{24}S\right), R_b(S)\right)\right), \quad Q^+ = \mathcal{B}(Q, R),$$
   $$M^+ = \mathcal{B}(\mathcal{B}(R_L^\top, M), R_R).$$
   **Algebraic derivation of orthogonality error **`[certain]`**:**
   Since $\tilde{A}^\top = -\tilde{A}$, $R_4^\top = I - \tilde{A} + \frac{1}{2}\tilde{A}^2 - \frac{1}{6}\tilde{A}^3 + \frac{1}{24}\tilde{A}^4$, so
   $$R_4^\top R_4 = \left(I + \tfrac{1}{2}\tilde{A}^2 + \tfrac{1}{24}\tilde{A}^4\right)^2 - \left(\tilde{A} + \tfrac{1}{6}\tilde{A}^3\right)^2 = I + \underbrace{(1-1)}_{0}\tilde{A}^2 + \underbrace{\left(\tfrac{1}{4}+\tfrac{1}{12}-\tfrac{1}{3}\right)}_{0}\tilde{A}^4 + \underbrace{\left(\tfrac{1}{24}-\tfrac{1}{36}\right)}_{\frac{1}{72}}\tilde{A}^6 + \tfrac{1}{576}\tilde{A}^8.$$
   Because eigenvalues of $\tilde{A}^2$ are $-\theta_k^2$: at $\|\tilde{A}\|_2 \le 0.5$, $\|R_4^\top R_4 - I\|_2 \le \frac{0.5^6}{72} = \mathbf{2.17 \times 10^{-4}}$ (**11× below** the BF16 floor $0.0024$). Moreover, $R_4$ approximates $\exp(\tilde{A})$ accurately, whereas $\operatorname{polar}(I+\tilde{A}) = \exp(\arctan(\tilde{A}))$ undershoots large angles ($\arctan(0.5)=0.464$).
2. **Cost:**
   - `warm_basis`: $Q^\top C$ (1) + $(Q^\top C)Q$ (2) + $S=\tilde{A}^2$ (3) + $(\frac{1}{6}\tilde{A}+\frac{1}{24}S)S$ (4) + $Q R$ (5) = **5 BF16 GEMMs ($10n^3$)**, down from **7 BF16 GEMMs ($14n^3$)** (**−28.6% FLOPs**).
   - `transport_moment` on warm events: **2 BF16 GEMMs** ($2m^2n + 2mn^2$) down from **4 GEMMs** (**−50% FLOPs**, AND halves BF16 rounding passes on $M$).
3. **Expected effect:** Cuts `warm_basis` by **~26–29%** at compute-bound sizes ($n \ge 1536$: $49.8 \to \sim 37\text{ ms}$ on 109M; $45.5 \to \sim 32.5\text{ ms/factor}$ at $n=8192$) and cuts warm transport from $5.06 \to 2.53\text{ ms}$ → **~15 ms saved per warm event (~1.2 ms/call)** on 109M, and drops the large-$n$ `warm/QR` ratio at $n=8192$ from $0.406 \to \sim 0.29$. Confidence: **high**.
4. **Failure modes:**
   - **No self-healing of pre-existing $Q$ Gram drift:** Unlike Newton–Schulz on $Y$, multiplying $Q \leftarrow R_b(Q R_4)$ does not damp existing $E_Q = Q^\top Q - I$; over 4 warm refreshes between QR50 repairs, BF16 rounding errors add up roughly as $\sim 0.0024\sqrt{4} \approx 0.005$ (worst-case $\le 0.01$). If replay shows drift $> 0.008$, add 1 NS step on the 2nd warm step only (average 5.5 GEMMs) or use $R_4$ on steps 1, 3 and 1 NS polish on steps 2, 4.
   - **Norm limit:** Breaks down if $\|\tilde{A}\|_2 > 0.7$ ($\frac{0.7^6}{72} = 0.0016$, $\frac{1.0^6}{72} = 0.014$). If Question B allows larger steps ($\|\tilde{A}\|_2 \le 1.2$), use 1 scaling-and-squaring step $R = R_4(\tilde{A}/2)^2$ (**6 GEMMs**, still 14% faster than current warm, error $\frac{1}{2304}\|\tilde{A}\|_2^6 = 4.3 \times 10^{-4}$ at $\|\tilde{A}\|_2 = 1.0$).
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Halving transport GEMMs reduces BF16 rounding noise injected into $M$ every 10 steps; monitor 4-step Gram drift in BF16.
6. **Cheap test:** (i) Replay harness: chain 4 consecutive $R_4$ warm steps between QR50s and check max Gram error (reject if $> 0.01$ at $t \bmod 50 = 40$). (ii) Microbenchmark at $n = 768\text{–}8192$ (reject if speedup at $n=3072$ is $< 22\%$).

---

### A7. Strided Covariance EMA Accumulation ($k_C = 2$, $\beta_C' = \beta_C^2 = 0.9025$)

1. **Exact update:** Only execute `covariance_ema` for $C_L, C_R$ on even ages ($t \bmod 2 == 0$, which automatically includes every refresh age $t \bmod 10 == 0$), using $\beta_C' = \beta_C^2 = 0.9025$ to preserve the exact 13.5-call half-life ($\ln 0.5 / \ln 0.9025 = 6.75$ updates $= 13.5$ calls). Compile a second lightweight `ordinary_nocov` graph for odd ages.
2. **Cost:** Eliminates $G_b G_b^\top$ ($2m^2n$), $G_b^\top G_b$ ($2mn^2$) = **434.87 GFLOP** AND **962 MB** of $C_L, C_R$ HBM read+write on **50%** of all calls (amortized saving: **217.4 GFLOP + 481 MB/call**).
3. **Expected effect:** Saves **~1.6–1.8 ms/call** across 100% of calls (~16–18% of ordinary step, **~6.5% of whole optimizer**). Confidence: **high** on speed; **medium-high** on negligible loss impact (effective sample size $N_{\text{eff}} = \frac{1+\beta_C'}{1-\beta_C'}$ drops from $39 \to 19.5$ gradient matrices, which for $3072 \times 768$ is still $19.5 \times 768 \approx 15,000$ rank-1 outer products, $\sim 5\times$ the dimension 3072). Going further to $k_C = 5$ ($\beta_C' = 0.7738$, $N_{\text{eff}} = 7.8$) saves ~2.7 ms/call but drops effective samples for $n=3072$ to ~6,000 ($2\times$ dimension), risking sample-covariance noise.
4. **Failure modes:** Higher variance in $C$ when $m \gg n$ and batch size is small; at $k_C = 2$ on 109M, risk is very low.
5. **Interaction with BF16 storage & $\beta_C=0.95$:** **Positive!** Because $1 - \beta_C'$ nearly doubles ($0.05 \to 0.0975$), new outer-product entries down to $0.04 |C_{ij}|$ (instead of $0.078 |C_{ij}|$) survive BF16 half-ULP rounding when added to $\beta_C' C_{ij}$.
6. **Cheap test:** **Cannot be tested in the open-loop replay harness** (which only saves $C$ every 10 ages, not per-step $G$). Speed: microbenchmark ordinary step with vs without `covariance_ema` (reject if saving $< 1.2\text{ ms/call}$). Quality: 2,000-call training run on seed 204 comparing $k_C=2$ vs $k_C=1$ (reject if CE at 2000 degrades by $> 0.001$).

---

### A8. Size-Aware Refresh Crossover (Drop Warm Refresh Above $n \approx 4096\text{–}6144$ in Favor of `sCholQR2`)

1. **Exact update:** Use warm10/`sCholQR2`-50 only for factor sizes where warm is much cheaper than `sCholQR2`, and switch directly to pure `sCholQR2` at cadence 10 or 20 for large factors ($n \ge 4096$, or even everywhere once A5 is in place).
2. **Cost:** Compare amortized refresh cost per 100 calls at $n = 8192$ (from §2.2 + A5/A6):
   - Current `warm10/QR50`: $8(45.5) + 2(112) = \mathbf{588\text{ ms}}$/100 calls.
   - Current `QR20` (no changes at all!): $5(112) = \mathbf{560\text{ ms}}$/100 calls (already faster AND better loss!).
   - `sCholQR2`-20 (~65 ms/factor): $5(65) = \mathbf{325\text{ ms}}$/100 calls (**1.8× faster** than current `warm10/QR50`, better loss!).
   - `A6-warm10 / sCholQR2-50` (~32.5 ms warm, ~65 ms sCholQR2): $8(32.5) + 2(65) = \mathbf{390\text{ ms}}$/100 calls — **STILL slower than `sCholQR2`-20 (325 ms)**!
   - Even on the 109M model: pure `sCholQR2`-10 (~125 ms + 5 ms transport) costs **13.0 ms/call** refresh overhead, **beating** current `warm10/QR50`'s **15.94 ms/call** with zero loss gap; while `A6-warm10 / sCholQR2-50` costs $0.08(39) + 0.02(130) = \mathbf{5.7\text{ ms/call}}$ (a **2.8×** cut in refresh overhead).
3. **Expected effect:** Eliminates the large-$n$ scaling wall completely (saves **45% of refresh time at $n=8192$** while upgrading from `warm10/QR50` to `QR20` quality). Confidence: **high**.
4. **Failure modes:** Inherits `sCholQR2`'s ridge/conditioning sensitivity (A5).
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Exact same as A5.
6. **Cheap test:** Microbenchmark `A6-warm` vs `sCholQR2` across $n \in \{768..8192\}$ to locate the exact crossover $n^*$ where $8T_{\text{warm}} + 2T_{\text{sCholQR2}} > 5T_{\text{sCholQR2}}$ (i.e. $T_{\text{warm}}/T_{\text{sCholQR2}} > 0.375$). **Reject** crossover only if `A6-warm` beats $0.375\,T_{\text{sCholQR2}}$ at $n=8192$.

---

### A9. Blackwell Consumer Dtype Split: FP16 Basis Storage + FP16-Accumulate Update Back-Projection $U$ `[speculation]`

1. **Exact update:**
   - Store $Q_L, Q_R$ in **FP16** instead of BF16 (same 2 bytes/element; entries of orthonormal $Q$ are $O(1/\sqrt{n}) \in [10^{-3}, 1]$, well inside FP16's normal range $[6.1\times 10^{-5}, 65504]$ for $n \le 10^6$, with **10-bit mantissa** vs BF16's 7-bit).
   - Keep $G, M, V, C_L, C_R$ in BF16 (they require FP32 exponent range).
   - For the back-projection $U = Q_L \hat{U} Q_R^\top$ (where $\hat{U} = M_f / (\sqrt{V_f} + \epsilon)$ has $O(1)$ entries!), cast $\hat{U}$ to FP16 and compute both `bmm`s with **`allow_fp16_reduced_precision_reduction = True`** (FP16 Tensor Core MMA with FP16 accumulation). Do **NOT** use FP16-acc for `Y^T Y` (Newton–Schulz needs $10^{-3}$ Gram precision) or `Q^T C Q`.
   - *(Note on BF16 flag in §1.3: NVIDIA Ampere/Ada/Blackwell BF16 Tensor Core MMA always accumulates in FP32; only **FP16-with-FP16-acc** unlocks the 2× GeForce consumer Tensor Core rate — ~419 TFLOP/s vs ~210 TFLOP/s on RTX 5090 **`[recalled][verify]`**.)*
2. **Cost:** Same 2-byte storage for $Q_L, Q_R$; moves **434.87 GFLOP/call** (33% of ordinary GEMMs, `U = Q_L @ u_hat @ Q_R.T`) from ~210 TFLOP/s peak to ~419 TFLOP/s peak. Casting `Q` FP16$\to$BF16 for `Q_L.T @ G @ Q_R` fuses into register loads or costs 481 MB read/write (~0.35 ms if unfused; check fusion).
3. **Expected effect:** Saves **~0.9–1.3 ms/call** on 100% of calls **and** reduces stored basis Gram error by **8×** ($0.0024 \to \sim 0.0003$!). Confidence: **medium** (FP16-acc speedup on Blackwell `bmm` needs microbenchmark verification; Inductor fusion of FP16$\to$BF16 cast before `Q_L.T @ G` determines net gain).
4. **Failure modes:** FP16 accumulation over $n = 3072$ inner products introduces $\sim \sqrt{n}\cdot 2^{-11} \approx 2.7\%$ relative noise in $U$ (and overflow if an anomalous $\hat{U}$ entry exceeds $\sim 10^3$, though Adam normalized updates are $O(1)$ — clamp $|\hat{U}| \le 10$ before FP16 cast). At $n=8192$, relative noise is $\sim 4.4\%$.
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Strictly improves basis orthogonality (0.0003 vs 0.0024), reducing $M$-transport distortion; noise in $U$ does not feed back into $M, V, C$ (only into weight step $W$).
6. **Cheap test:** Speed: microbenchmark `Q_L @ u_hat @ Q_R.T` in FP16 (FP16-acc) vs BF16 on the 4 109M shapes (reject if saving $< 0.8\text{ ms}$ net of `Q` cast). Quality: **cannot be tested in open-loop replay** (no $U$ or loss); needs a 2,000-call training run (reject if CE at 2000 degrades by $> 0.001$).

---

# Question B: Tracking Quality

## 0. Root-Cause Diagnosis & Direct Answers to Your Three Questions

### Why the current `warm_basis` leaves $r_C = 0.150$ (synthetic), mean $0.38$ / max $0.975$ (trained), and only closes 12–37% of the loss gap

Four distinct mathematical defects compound inside the 5 lines of `warm_basis` **`[certain]`**:

1. **A factor-of-2 under-relaxation bug (`-.5`).**
   For $Q^+ = Q e^A$ ($A^\top = -A$) and $B = D + E$ ($D = \operatorname{diag}(d)$, $E$ off-diagonal):
   $$B^+ = e^{-A}(D + E)e^A = D + E + [D, A] + [E, A] + \tfrac{1}{2}[[D, A], A] + O(E^3).$$
   Since $([D, A])_{ij} = (d_i - d_j) A_{ij}$, exact first-order annihilation ($E + [D, A] = 0$) requires
   $$A^*_{ij} = -\frac{B_{ij}}{d_i - d_j}, \quad \text{yielding } B^+ = D + \tfrac{1}{2}[E, A^*] + O(E^3) = D + O(r_C^2) \quad (\textbf{quadratic convergence!}).$$
   Your code has `-.5`, giving $A_{ij} \approx \frac{1}{2} A^*_{ij}$, which leaves **at least 50%** of every off-diagonal entry intact ($B^+_{ij} \approx +\frac{1}{2} B_{ij}$) even on well-separated pairs with zero clipping.
2. **Global scale $s = \max_i |d_i|$ freezes 90%+ of a power-law spectrum.**
   The damping factor is $\frac{h_{ij}^2}{h_{ij}^2 + 0.01^2}$ with $h_{ij} = (d_i - d_j)/d_{\max}$. For any eigenpair in the bulk or tail with $d_i, d_j \le 0.01 d_{\max}$ — **even a 10:1 well-separated pair** like $d_i = 10^{-3} d_{\max}, d_j = 10^{-4} d_{\max}$ — $|h_{ij}| = 9\times 10^{-4}$, so $h_{ij}^2/(h_{ij}^2 + 0.01^2) = 0.008$: **99.2% of the rotation is killed**.
3. **Global Frobenius cap suffers $O(1/\sqrt{n})$ dimensional collapse.**
   For skew $A$ with eigenvalues $\pm i\theta_k$, $I+A$ has singular values $\sqrt{1+\theta_k^2}$, so Newton–Schulz convergence ($\sigma < \sqrt{3} \iff \|A\|_2 < \sqrt{2}$, 2-step BF16 accuracy at $\|A\|_2 \lesssim 0.55$) depends strictly on the **spectral norm** $\|A\|_2 = \max_k |\theta_k|$, NOT the Frobenius norm $\|A\|_F/\sqrt{2} = \sqrt{\sum_{k=1}^{n/2} \theta_k^2}$. Even on your synthetic benchmark ($A_{\text{true}}$ entries $\sim 0.2/\sqrt{n}$, $\lambda_i \propto 1/i$), a quick integral over the top $\sim 100$ undamped modes gives $\|A_{\text{warm}}\|_F/\sqrt{2} \approx \sqrt{\frac{0.04}{4}\int_0^\infty (1+(0.01x)^2)^{-2}dx} = \sqrt{0.01 \cdot 25\pi} \approx 0.89 > 0.5$, clipping $a \approx 0.56$! Combined with `-.5`, the effective step is $0.5 \times 0.56 = 0.28$, predicting $r_C^+ / r_C \approx 1 - 0.28 = \mathbf{0.72}$ — matching your measured $0.150/0.20 = \mathbf{0.75}$! Worse: if you fix Defect 2 without fixing Defect 3, all $n/2$ planes activate and $\|A\|_F/\sqrt{2}$ explodes by $\sqrt{n}$, throttling the entire step toward zero.
4. **Equal-diagonal paralysis explains your trained max $r_C = 0.975$.**
   If an emerging rank-1 spike $\lambda_1 u u^\top$ projects roughly equally onto $k$ basis columns ($Q^\top u \approx \frac{1}{\sqrt{k}}\mathbf{1}_k$), then the $k\times k$ block of $B$ is $\frac{\lambda_1}{k}\mathbf{1}_k \mathbf{1}_k^\top$, which has **equal diagonals** $d_1 = \dots = d_k = \lambda_1/k$ and $r_C = \sqrt{1 - 1/k}$ (**for $k=20$, $r_C = 0.9747$**!). Because $h_{ij} = 0$, `warm_basis` computes **$A = 0$ identically** and stays frozen at $r_C = 0.975$ for 40 calls until QR50 rescues it. By contrast, `QRRepair` computes $Z = CQ = QB$ whose first column points *directly along $u$*, dropping $r_C$ from $0.975 \to 0$ in one step. The exact same stall occurs at every **eigenvalue level crossing** (where $d_i$ and $d_j$ pass through $d_i = d_j$).

---

### Sub-question 1: Is $r_C$ the right proxy for loss? What to measure instead?

**No — $r_C$ is systematically biased toward the top ~10 eigenvalues and blind to what SOAP's Adam preconditioner actually sees `[certain]`.**

- In a power-law spectrum $\lambda_i \propto 1/i$, the top 10 eigenvalues account for $\sum_{i=1}^{10} i^{-2} / (\pi^2/6) = \mathbf{94.2\%}$ of $\|B\|_F^2$. Thus $r_C$ barely notices even 100% misalignment in modes $11\text{–}3072$.
- Yet SOAP runs **Adam** in the rotated basis, dividing each coordinate by $\sqrt{V_{ij}} + \epsilon \approx \sqrt{d_i^{(L)} d_j^{(R)}}$, which **whitens** the spectrum so bulk and tail coordinates contribute $O(1)$ equally to the parameter update.
- **What actually governs SOAP's diagonal-approximation fidelity:** The Gaussian KL divergence (Hadamard gap) between $C$ and its diagonal approximation $Q D Q^\top$ ($D = \operatorname{diag}(B)$, with Adam floor $\epsilon$):
  $$D_{\text{KL}}\!\left(\mathcal{N}(0, C_\epsilon)\,\|\,\mathcal{N}(0, Q D_\epsilon Q^\top)\right) = \frac{1}{2}\left[\underbrace{\operatorname{tr}(D_\epsilon^{-1} B_\epsilon) - n}_{=\,0} + \log\frac{\det D_\epsilon}{\det C_\epsilon}\right] = -\frac{1}{2}\log\det\!\left(D_\epsilon^{-1/2} B_\epsilon D_\epsilon^{-1/2}\right) \ge 0.$$
  Expanding $R = D_\epsilon^{-1/2} B_\epsilon D_\epsilon^{-1/2} = I + E_{\text{corr}}$ to second order:
  $$-\frac{1}{2}\log\det(I + E_{\text{corr}}) = \frac{1}{4}\|E_{\text{corr}}\|_F^2 + O(\|E_{\text{corr}}\|^3) = \frac{1}{4}\sum_{i \neq j}\frac{B_{ij}^2}{(d_i + \epsilon)(d_j + \epsilon)}.$$
- **What to measure instead (all computable in your existing replay harness!):**
  1. **Whitened correlation error / Hadamard KL gap:** $-\frac{1}{2n}\log\det(D_\epsilon^{-1/2} B_\epsilon D_\epsilon^{-1/2})$ and your existing **RMS off-diagonal correlation** $\sqrt{\frac{1}{n(n-1)}\sum_{i\neq j} B_{ij}^2 / ((d_i+\epsilon)(d_j+\epsilon))}$ — **prioritize these over raw $r_C$!**
  2. **Preconditioned condition number** $\kappa(D_\epsilon^{-1/2} B_\epsilon D_\epsilon^{-1/2})$ (or 99th/1st eigenvalue ratio of $R$): directly controls the step-size/curvature ceiling of SOAP.
  3. **Inter-refresh basis churn** $\chi = 1 - \frac{1}{n}\sum_i |(Q^\top Q^+)_{ii}|$ (after permutation alignment $\pi$), split into top-64 vs tail: **essential guardrail!** Because $V$ is not rotated, an aggressive refresh that slashes tail correlation by whipping degenerate noise eigenvectors around every 10 calls will win on correlation in open-loop replay while **hurting training loss** by invalidating $V$.
  4. **Worst-case (95th/max) $r_C$ across factors**, not just the mean — your mean $0.38$ hides catastrophic factor stalls at $0.975$.

### Sub-question 2: Role of the call-500 gap (+0.124 CE)

1. **Why call 500 is uniquely bad:** At `k=1` (cold start), $C_L$ for SwiGLU-in ($3072\times 768$) comes from a **single** gradient $G_1$, so $\operatorname{rank}(C_L) \le 768$ — **2,304 of 3,072 eigenvalues are identically zero** (random eigenvectors from `eigh`). Over calls 1–500 (warmup + feature emergence), the effective rank explodes from $768 \to 3072$ and dominant spikes rotate violently. Your schedule waits **50 updates** for the first QR repair, and runs only 9 QRs before call 500 (vs 49 for QR10), while `warm_basis` stalls on those near-degenerate emerging modes (closing only **12%** of the QR50–QR10 gap at call 500 vs **37%** at call 2000).
2. **Does call 500 explain the call-10,000 gap (+0.010–0.011)?** Only partially **`[certain]`**. Between call 500 and 2000, the gap shrinks $15\times$ ($+0.124 \to +0.008$) largely because the CE slope flattens. **Crucially**, between call 2000 and 10,000 on seed 204, the gap does *not* keep decaying — it slightly widens ($+0.0080 \to +0.0112$). Thus the final $+0.010\text{–}0.011$ deficit is sustained by **ongoing steady-state basis lag** (mean $r_C = 0.38$, max $0.975$ even at age 9990!) plus whatever portion is caused by **BF16 state** (unmeasured until BF16 QR10 completes!). Fixing *only* early calls (B4) will wipe out the call-500 gap, but you must ALSO fix steady-state tracking (B1–B3/B6) to close the call-10,000 gap.

### Sub-question 3: Treatment of $V$ and column order in the warm step

- **Column order:** `warm_basis` already computes $d = \operatorname{diag}(B)$, the exact Rayleigh quotients used by `qr_basis`, yet never sorts by $\pi = \operatorname{argsort\_desc}(d)$. Leaving columns unsorted across eigenvalue crossings breaks QR10's descending invariant for up to 40 steps, causes sign inversions with tie-breaking, and forces large multi-column shuffles at QR50. Sorting $Q, B, M$ and permuting $V$ by $\pi_L, \pi_R$ costs **only 0.38 ms** across all 48 matrices (§2.1) and should be done on **every** warm refresh.
- **Why naive $V$ rotation $V^+ = (R_L \odot R_L)^\top V (R_R \odot R_R)$ is a trap **`[certain]`**:**
  Although $S = R \odot R$ is doubly stochastic (nonnegative, trace-preserving, exact for permutations), it assumes old coordinates $\bar{G}$ were *uncorrelated*, dropping the cross-term $2 B_{ij}\sin\theta\cos\theta$. When $R$ rotates *into* the true eigenbasis to remove misalignment $\theta$, that cross-term is **negative** (it cancels leaked variance!).
  *Counterexample:* Let $C = U \operatorname{diag}(10^4, 10^{-2}) U^\top$ with old basis misaligned by $\theta = 0.01$. Old diagonals are $d_1 \approx 10^4$ and $d_2 = 10^4 \sin^2(0.01) + 10^{-2}\cos^2(0.01) \approx \mathbf{1.01}$. Rotating back by $-\theta$ restores true variance $\lambda_2 = \mathbf{0.01}$. Leaving $V_2$ unchanged gives $1.01$; applying $(R \odot R)^\top V (R \odot R)$ gives $V_2^+ = \sin^2(0.01)\cdot 10^4 + \cos^2(0.01)\cdot 1.01 = \mathbf{2.01}$ — **2× worse than doing nothing** (it double-counts the leaked variance!). This explains why SOAP itself (Vyas et al. 2024 **`[recalled][verify]`**) only permutes $V$. The right fix (when rotations are large) is **Kronecker-ratio rescaling** using the *actual* post-rotation diagonal $d^+ = \operatorname{diag}(R^\top B R)$, which includes all cross-terms (see **B5**).

---

## Question B Candidates

### B1. Fixed Full-Step Relative-Gap $\arctan$-Jacobi + Rayleigh Sort ($V$-Perm) + Row-$\ell_2$/Spectral Cap + $R_4(\tilde{A}/2)^2$

1. **Exact update:** Fixes all four defects of `warm_basis` in one drop-in replacement:
   ```python
   # 1. Projected covariance + Rayleigh descending sort (also permute V by order!)
   b = mm32(torch.bmm(qb.transpose(-1, -2), cb), qb)
   b = 0.5 * (b + b.transpose(-1, -2))
   d = b.diagonal(dim1=-2, dim2=-1)
   order = d.argsort(dim=-1, descending=True)
   q_ord = q.gather(-1, order.unsqueeze(-2).expand_as(q))
   # permute rows & cols of b by order -> d_ord is monotonically descending!
   b_ord = b.gather(-2, order.unsqueeze(-1).expand_as(b)).gather(-1, order.unsqueeze(-2).expand_as(b))
   d_ord = b_ord.diagonal(dim1=-2, dim2=-1)
   # 2. Pairwise relative-gap exact 2x2 Jacobi angle (NO 0.5 factor; breaks d_i == d_j ties!)
   d_max = d_ord.abs().amax(-1, keepdim=True).clamp_min(1e-30)
   s_ij = d_ord.abs().unsqueeze(-1) + d_ord.abs().unsqueeze(-2) + 1e-5 * d_max.unsqueeze(-1)
   gap_abs = (d_ord.unsqueeze(-1) - d_ord.unsqueeze(-2)).abs()
   idx = torch.arange(b.shape[-1], device=b.device)
   sgn = (idx.unsqueeze(0) - idx.unsqueeze(1)).sign().float()     # +1 for i<j, -1 for i>j, 0 on diag
   a = -0.5 * sgn * torch.atan2(2.0 * b_ord, gap_abs + 0.05 * s_ij)
   # 3. Local row-l2 angle cap (no sqrt(n) collapse; skew-preserving D_w A D_w)
   row_norm = a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
   w = (0.5 / row_norm).clamp(max=1.0)
   a = w * a * w.transpose(-1, -2)
   # 4. 3-step block power iteration (r=4 fixed probes) for spectral norm ||a||_2 <= theta_max = 0.9
   # (omitted for brevity: 3 bmm with (b, n, 4), cost 24 n^2 FLOPs; scale a by min(1, 0.9 / (1.05*sigma)))
   # 5. Degree-4 scaling-and-squaring matrix exponential R = R_4(a/2)^2
   a1 = 0.5 * a
   a1b = a1.to(compute_dtype)
   s = mm32(a1b, a1b)
   p = mm32((a1 / 6.0 + s / 24.0).to(compute_dtype), s.to(compute_dtype))
   r_half = (eye + a1 + 0.5 * s + p).to(compute_dtype)
   r = torch.bmm(r_half, r_half)
   q_new = torch.bmm(q_ord.to(compute_dtype), r).to(basis_dtype).contiguous()
   # Return (q_new, order, r) -> permute V by order, and transport M via r_L^T @ M_perm @ r_R!
   ```
   **Exact algebraic checks **`[certain]`**:**
   - **Exact skew-symmetry:** Swapping $i \leftrightarrow j$ flips `sgn` and leaves `b_ord`, `gap_abs`, `s_ij` invariant, so $A_{ji} = -A_{ij}$ and $A_{ii} = 0$ identically.
   - **Well-separated pairs anywhere in spectrum ($d_i = 10 d_j$, even at $10^{-4} d_{\max}$):** $\operatorname{atan2}(2B_{ij}, 9d_j + 0.55 d_j) \approx \frac{2B_{ij}}{9.55 d_j} \implies A_{ij} \approx -0.94 \frac{B_{ij}}{d_i - d_j}$ (full quadratic step!).
   - **Equal-diagonal stall ($C = \begin{pmatrix}1 & 0.5 \\ 0.5 & 1\end{pmatrix}$):** $A_{12} = -\frac{1}{2}\operatorname{atan2}(1.0, 0.10) = -0.736\text{ rad}$ (94% of exact $-\pi/4 = -0.785$ rad, dropping $B_{12}$ by $10\times$ in one step!), while equal diagonals with tiny noise ($B_{12} = 10^{-4}$) give $A_{12} = -10^{-3}$ (no noise blow-up).
   - **Row-$\ell_2$ cap:** $(D_w A D_w)^\top = -D_w A D_w$; geometrically $\|A_{:, j}\|_2$ is column $j$'s rotation angle, which is $O(1)$ independent of $n$ on dense rotations ($0.2$ on your synthetic benchmark vs $\|A\|_F/\sqrt{2} = 0.2\sqrt{n/2}$!), and prevents one noisy pair from throttling the rest of the matrix.
   - **Orthogonality of $R_4(a/2)^2$:** Since $R_4(a/2)$ is a polynomial in skew $a$, it commutes with its transpose, giving $R^\top R = (I + \frac{1}{72}(a/2)^6 + \dots)^2 = I + \frac{1}{2304} a^6 + O(a^8)$. At $\|a\|_2 = 0.9$, $\|R^\top R - I\|_2 \le 2.3 \times 10^{-4}$ (**10× below** BF16 floor!).
2. **Cost:** $Q^\top C$ (1) + $(Q^\top C)Q$ (2) + $S=a_1^2$ (3) + $P$ (4) + $R = R_{\text{half}}^2$ (5) + $Q_{\text{ord}} R$ (6) = **6 BF16 GEMMs ($12n^3$)** + $24n^2$ block power iter + $O(n^2)$ `atan2`/sort + 0.38 ms $V$-perm, AND **saves 2 GEMMs on transport** ($M^+ = R_L^\top M_{\text{perm}} R_R$). Net warm event is **~15% FASTER than current `warm_basis`**!
3. **Expected effect:** On the synthetic $r_C = 0.20$ input, drops single-refresh $r_C$ from $0.150 \to \mathbf{0.03\text{–}0.05}$ (closing ~80–90% of the gap to QR's $0.025$), dramatically lowers bulk/tail RMS off-diagonal correlation, and removes equal-diagonal freezes (expect trained mean $r_C$ to drop from $0.38 \to \sim 0.22\text{–}0.26$, closing **~60–80%** of the steady-state `warm10/QR50`–`BF16 QR10` loss gap). Confidence: **high**.
4. **Failure modes:**
   - **Over-rotating degenerate tail noise:** If $\delta = 0.05$ is too small on very noisy finite-sample bulk spectra, bulk columns may churn every 10 steps and hurt $V$ (tune $\delta \in \{0.05, 0.10, 0.20\}$ using the churn metric!).
   - **4-step BF16 Gram drift** (same as A6: verify $\le 0.008$ over 4 warm steps; add 1 NS step on warm #2 if needed).
   - **Large multi-column equal-diagonal cluster** ($k = 20$ rank-1 spike): single step with $\|a\|_2 \le 0.9$ breaks the stall and concentrates energy toward column 1, though needs ~2–3 steps (see B2/B3) to fully collapse $r_C = 0.975 \to 0.05$.
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Because $B = \mathcal{F}(\mathcal{B}(Q^\top, C), Q)$ rounds $Q^\top C$ to BF16, tail entries $d_j < 10^{-3} d_{\max}$ can suffer BF16 noise when $Q$ is misaligned (see Pitfall 3); keeping the floor $10^{-5} d_{\max}$ (or $10^{-4} d_{\max}$) prevents dividing by BF16 noise in the deep tail.
6. **Cheap test:** Replay harness (§2.6(i)) on widths 256–2048: measure mean/max $r_C$, RMS off-diagonal correlation, KL gap, tail churn, and Gram error across $\delta \in \{0.05, 0.1, 0.2\}$. **Reject** if single-refresh synthetic $r_C > 0.07$ or replay mean $r_C$ does not beat current warm by $> 40\%$ of the gap to QR10 without doubling tail churn.

---

### B2. Two-Iteration Local-Frame Warm Jacobi (`B1 × 2`)

1. **Exact update:** Run **two** inner iterations of B1 inside a single warm refresh, propagating $B$ in rotated space so $C$ is only touched once:
   - $B_0 = \operatorname{sym}(\mathcal{F}(\mathcal{B}(Q^\top, C), Q))$ (2 GEMMs).
   - Iter 1: sort $B_0$ by $\pi_1$, build $A_1$ via B1, compute $R_1 = R_4(A_1)$ (2 GEMMs, since 2 steps need smaller cap $\|A_1\|_2 \le 0.6$, saving the squaring GEMM!), update $B_1 = \operatorname{sym}(\mathcal{F}(\mathcal{B}(R_1^\top, B_{0,\text{ord}}), R_1))$ (2 GEMMs).
   - Iter 2: sort $B_1$ by $\pi_2$, build $A_2$ via B1, compute $R_2 = R_4(A_2)$ (2 GEMMs), compose $R_{\text{tot}} = \mathcal{B}(R_1[:, \pi_2], R_2)$ (1 GEMM), $Q^+ = \mathcal{B}(Q_{\pi_1 \circ \pi_2}, R_{\text{tot}})$ (1 GEMM). Permute $V$ by $\pi = \pi_1 \circ \pi_2$ and transport $M^+ = R_{\text{tot}, L}^\top M_{\text{perm}} R_{\text{tot}, R}$ (2 GEMMs).
2. **Cost:** **10 BF16 GEMMs ($20n^3$)** per factor ($1.43\times$ current 7-GEMM `warm_basis`), minus 2 transport GEMMs → **~1.3× current warm event cost** at $n = 768\text{–}3072$, well inside your $2\times$ budget and still $3\times\text{–}25\times$ below QR at $n \le 4096$.
3. **Expected effect:** Achieves **$O(r_C^4)$** quartic convergence on separated modes; crucially, when Iter 1 splits an equal-diagonal cluster or hits the rotation cap $\rho_{\max} = 0.5$, Iter 2 immediately diagonalizes the newly separated modes in the same refresh. On synthetic $r_C = 0.20$, expect single-refresh $r_C \approx \mathbf{0.005\text{–}0.02}$ (**beating QR10's $0.025$**!), and trained mean $r_C \approx \mathbf{0.18\text{–}0.22}$ matching QR10. Confidence: **high**.
4. **Failure modes:** Two consecutive BF16 roundings inside $B_1 = R_1^\top B_0 R_1$ can slightly degrade deep-tail precision; at $n=8192$ ($10/7 \times 45.5 \approx 65\text{ ms}$, or ~53 ms with A3 micro-batching), it approaches `sCholQR2`'s cost (so at $n \ge 6144$, A8/B6 is preferable).
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Only ONE BF16 rounding is applied to $Q^+$ ($Q_{\text{perm}} R_{\text{tot}}$), same as B1 — does not double $Q$'s rounding drift.
6. **Cheap test:** Replay harness comparing B2 vs B1 vs QR10 on mean/max $r_C$ and RMS correlation, plus microbenchmark at $n = 768\text{–}3072$. **Reject** if B2 fails to cut B1's remaining gap to QR10 by at least 50% (especially max $r_C$), or if cost exceeds $1.6\times$ current warm at $n = 3072$.

---

### B3. Top-$k$ Exact Power-Subspace Deflation (Procrustes Reflection $H_+$, Zero SVD/QR) + B1 Jacobi

1. **Exact update:** Directly eliminates the $r_C = 0.975$ emerging-spike stall and matches `QR10` **identically** on the top-$k$ subspace ($k = 32$ or $64$) before running B1 on the full matrix, using only tall-skinny GEMMs and a $k\times k$ Cholesky solve:
   - Sort $Q$ by Rayleigh quotient $\pi = \operatorname{argsort\_desc}(d)$ (or reuse previous order; permute $V$ by $\pi$). Split $Q = [Q_1, Q_2]$ with $Q_1 \in \mathbb{R}^{n \times k}, Q_2 \in \mathbb{R}^{n \times (n-k)}$.
   - Exact top-$k$ power step: $Z_1 = \mathcal{F}(C, Q_1) \in \mathbb{R}^{n \times k}$, orthonormalize $U_1 = \operatorname{sCholQR2}(Z_1)$ (only a $k\times k$ Cholesky!).
   - Minimal-rotation orthogonal extension of $Q_1 \to U_1$ to $Q_2$ (exact geodesic on $O(n)$):
     $$M = Q_1^\top U_1 \in \mathbb{R}^{k \times k}, \quad R_p = \operatorname{polar}(M) \;\text{(8 iters of } k\times k\text{ Halley/NS)}, \quad \tilde{U}_1 = U_1 R_p^\top, \quad S = \operatorname{sym}(M R_p^\top) = (MM^\top)^{1/2} \succeq 0,$$
     $$V_+ = Q_1 + \tilde{U}_1, \quad H_+ = I_n - V_+(I_k + S)^{-1} V_+^\top, \quad Q_{\text{mid}} = \big[U_1,\; Q_2 - V_+(I_k + S)^{-1}(\tilde{U}_1^\top Q_2)\big].$$
   - Then run 1 pass of **B1** on $Q_{\text{mid}}$ (or run $R_4$ directly from $B_{\text{mid}} = Q_{\text{mid}}^\top C Q_{\text{mid}}$).
   **Exact algebraic proof **`[certain]`**:**
   - Since $Q_1^\top Q_1 = \tilde{U}_1^\top \tilde{U}_1 = I_k$ and $Q_1^\top \tilde{U}_1 = \tilde{U}_1^\top Q_1 = S \succeq 0$:
     $$V_+^\top V_+ = 2(I_k + S), \qquad H_+^\top H_+ = I_n - 2V_+(I_k+S)^{-1}V_+^\top + V_+(I_k+S)^{-1}\big[2(I_k+S)\big](I_k+S)^{-1}V_+^\top = I_n.$$
   - Because all eigenvalues of $S$ are principal cosines $\cos\theta_j \in [0, 1]$, **$\kappa(I_k + S) \in [1, 2]$ ALWAYS** — completely singularity-free even for an exact $90^\circ$ rotation ($\cos\theta_j = 0$)!
   - Since $V_+^\top Q_1 = I_k + S$, $H_+ Q_1 = Q_1 - V_+ = -\tilde{U}_1$, so $U_1^\top (H_+ Q_2) = -R_p (H_+ Q_1)^\top (H_+ Q_2) = -R_p (Q_1^\top Q_2) = 0$ and $(H_+ Q_2)^\top (H_+ Q_2) = I_{n-k}$.
   - Because $H_+ = I$ on $\operatorname{span}(Q_1, U_1)^\perp$, it perturbs $Q_2$ by the **minimum possible Frobenius distance** (preserving $V$ on the bottom $n-k$ columns!).
2. **Cost:** Top-$k$ deflation: $Z_1 = C Q_1$ ($2n^2k$) + $\tilde{U}_1^\top Q_2$ ($2n(n-k)k$) + $V_+ W$ ($2n(n-k)k$) + $O(nk^2 + k^3)$ = **$6n^2k + O(nk^2)$ FLOPs**.
   As a fraction of ONE $n\times n$ GEMM ($2n^3$): $\frac{3k}{n}$ — for $k = 64$, that is **0.25 GEMM** at $n=768$, **0.0625 GEMM** at $n=3072$, and **0.023 GEMM** at $n=8192$! Combined with B1 (6 GEMMs), total is **~6.1–6.3 GEMMs** (**still faster than current 7-GEMM `warm_basis`!**).
3. **Expected effect:** Completely destroys the max $r_C = 0.975$ failure mode (takes any rank-$\le k$ spike spread across arbitrary equal diagonals straight to $r_C \approx 0$ in 1 step, identical to QR10 on the top $k$ modes!) and dramatically slashes the call-500 gap (where top modes rotate by $>45^\circ$). Confidence: **high**.
4. **Failure modes:**
   - If a mode rotates by **exactly** $90^\circ$ ($\sigma_{\min}(M) = 0$), plain Newton–Schulz for $R_p = \operatorname{polar}(M)$ maps $0 \to 0$ (losing rank in $\tilde{U}_1$); fix by initializing $M \leftarrow M + 10^{-3} I_k$ and using 6–8 steps of cubic Halley iteration $X_{t+1} = X_t(3I + X_t^\top X_t)(I + 3 X_t^\top X_t)^{-1}$ on $k\times k$ (microseconds for $k=64$, converges from $10^{-3}$ to $1 - 10^{-5}$ in 8 steps), or `torch.linalg.svd` on the tiny $(b, k, k)$ matrix.
   - Note: because $Q_{\text{mid}}$ is formed before B1, `transport_moment` needs $R_{\text{full}} = Q^\top Q^+$ or standard 4-GEMM transport (5.06 ms, same as today).
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Keep the tiny $k\times k$ ops and $Z_1$ in FP32; cast $Q_{\text{mid}}$ to BF16 only after B1 (or keep $Q_{\text{mid}}$ in BF16 for B1's $Q^\top C Q$). Because $Z_1 = \mathcal{F}(C, Q_1)$ does not round intermediate $Q^\top C$ to BF16, top-$k$ accuracy is strictly cleaner than `warm_basis`.
6. **Cheap test:** Replay harness on widths 256–2048 + synthetic $r_C = 0.975$ rank-1 equal-diagonal test ($B = \frac{1}{20}\mathbf{1}_{20}\mathbf{1}_{20}^\top$). **Reject** if max $r_C$ across all replay snapshots exceeds $1.25\times$ QR10's max $r_C$, or if $k=64$ deflation adds $> 20\%$ latency at $n=768$ in the microbenchmark.

---

### B4. Front-Loaded Refresh Schedule (QR10 / `sCholQR2`-10 for Calls 1–1000, Warm10/QR100 Thereafter)

1. **Exact update:** Reallocate the **same** (or smaller) total QR budget over the 10,000-call run to match the non-stationarity of the eigenspace:
   - Calls $t \in [1, 1000]$ (plus an extra refresh at $t = 2$ or $t = 5$ to exit the rank-$\le 768$ single-gradient cold start!): run `QR` (or `sCholQR2` from A5) every **10** ages (100 QRs).
   - Calls $t \in [1001, 10000]$: run improved warm (B1/B3) every 10 ages and `QR`/`sCholQR2` every **100** ages (90 QRs).
   - Total QRs = **190** (fewer than your current **199** QRs!).
2. **Cost:** **Negative incremental cost** relative to current `warm10/QR50` (190 vs 199 QRs over 10k calls; 9 fewer QR events = **−0.5 ms/call** amortized, and zero extra cost per warm step!).
3. **Expected effect:** Immediately **eliminates 100% of the +0.124 call-500 gap** (by construction, calls 1–1000 ARE BF16 QR10!), and isolates how much of the call-10,000 gap is an early-trajectory scar vs ongoing steady-state lag. Combined with B1/B3 after call 1000, expect call-10,000 loss to approach BF16 QR10 within **0.001–0.003 CE** **`[speculation]`**. Confidence: **high** for call 500; **medium** for call-10k (depends on B1/B3 holding $r_C \approx 0.20$ over 100-step QR intervals).
4. **Failure modes:** If B1/B3 still drifts over 9 consecutive warm refreshes (age $100k+10 \dots 100k+90$), extending QR cadence to 100 may hurt late-stage loss (test QR50 vs QR100 in replay first!). Also, a 2,000-call short run will look artificially good unless you evaluate the warm phase ($t > 1000$) separately.
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Eliminates the worst cold-start BF16 degeneracy (2,304 zero eigenvalues in $C_L$ during calls 1–50).
6. **Cheap test:** (i) Replay harness (ages 0–1000): verify whether B1/B3 sustains QR10-level $r_C$ and Gram error over a 100-step QR interval (9 consecutive warm refreshes). (ii) 2,000-call training run with crossover at call 500 (`QR10` for 1–500, `B1-warm10/QR50` for 501–2000) to measure both early recovery and post-500 drift. **Reject** QR100 late cadence if replay Gram error $> 0.01$ or $r_C$ climbs $> 0.05$ above QR50 between ages 500 and 1000.

---

### B5. Kronecker-Ratio $V$-Rescaling After Basis Rotations

1. **Exact update:** After refreshing $(Q_L, Q_R) \to (Q_L^+, Q_R^+)$ and permuting $V$ by $(\pi_L, \pi_R)$, rescale $V$ by the **exact** ratio of post- to pre-rotation factor diagonal variances (which incorporates all de-correlating cross-terms, unlike $(R\odot R)^\top V(R\odot R)$!):
   $$d_i^{(L)+} = \big((Q_L^+)^\top C_L Q_L^+\big)_{ii} = \sum_k (B_L R_L)_{ki} (R_L)_{ki}, \qquad \gamma_i^{(L)} = \operatorname{clip}\!\left(\frac{d_i^{(L)+} + \epsilon_d}{d_{\pi_L(i)}^{(L)} + \epsilon_d},\; c_{\min},\; c_{\max}\right),$$
   $$V_{ij}^+ = V_{\pi_L(i), \pi_R(j)} \cdot \big(\gamma_i^{(L)}\big)^\alpha \big(\gamma_j^{(R)}\big)^\alpha,$$
   with conservative bounds $c_{\min} = 0.25, c_{\max} = 4.0$, floor $\epsilon_d = 10^{-4} d_{\max}$, and damping exponent $\alpha \in \{0.5, 1.0\}$. (On the earlier $\lambda_1 = 10^4, \lambda_2 = 10^{-2}, \theta = 0.01$ counterexample, $d_2 = 1.01, d_2^+ = 0.01 \implies$ ratio moves $V_2$ in the **correct downward** direction instead of doubling it!)
2. **Cost:**
   - Inside **B2** (2-iter Jacobi): $B_1 = R_1^\top B_0 R_1$ is already computed, so $d^+$ is **free** (or $1\times n^3$ GEMM $B R$ + $O(n^2)$ dot product if applied after the final rotation in B1).
   - Rescaling $V$: $O(mn)$ fused into the $V$-permutation kernel (**~0.1 ms** across all 48 matrices).
3. **Expected effect:** Eliminates multi-step Adam step-size overshoots/undershoots immediately after large basis rotations (especially during calls 1–1000 and level crossings where $V$ currently lags with a 13.5-call half-life). Likely modest steady-state gain (**~0.001–0.003 CE** **`[speculation]`**), larger early-training stability gain. Confidence: **medium-low** (Kronecker factorization $V_{ij} \propto d_i^{(L)} d_j^{(R)}$ is approximate; when non-Kronecker structure dominates, rescaling all $n$ entries in row $i$ by the row-average variance ratio can inject noise).
4. **Failure modes:** Non-Kronecker gradient structure (a single outlier entry $(i, j)$ drives $d_i^{(L)+}/d_i^{(L)}$ and falsely rescales the other $n-1$ entries in row $i$); deep-tail division noise if $\epsilon_d$ is too small.
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Because $\beta_2 = 0.95$ has a relatively short half-life (13.5 calls), $V$ already adapts within ~10–15 steps after a refresh; this matters most when rotations happen every 10 steps (leaving $V$ perpetually half-adapted!). Clamping $[0.25, 4.0]$ prevents BF16 tail noise from distorting $V$.
6. **Cheap test:** **Cannot be tested in the open-loop replay harness** (no $V$, no gradients, no loss). Smallest test: 2,000-call training run on seed 204 comparing `B1 + V-perm only` vs `B1 + V-perm + Kronecker V-rescale` ($\alpha \in \{0.5, 1.0\}$). **Reject** if CE at 500 and 2000 do not both improve by $\ge 0.001$.

---

### B6. Hybrid `B3` (at $n \le 2304$) + Pure `sCholQR2` Every 10 Steps (at $n \ge 3072$, or Everywhere!)

1. **Exact update:** Notice that once **A5 (`sCholQR2`)** fixes Cholesky-QR's stability, its cost at $n = 3072$ (~4.5 ms/factor) is only **1.8×** current warm (2.46 ms/factor), and at $n = 8192$ (~65 ms/factor) is **1.4×** current warm (45.5 ms/factor) — **strictly inside your Question B budget ($\le 2\times$ warm at $n=3072$, well below QR at every $n$)!**
   - Option B6a (Hybrid by size): Run **B3** (or B1) on warm steps for $n \le 2304$ (where warm is $3\times\text{–}4\times$ faster than `sCholQR2`) and **pure `sCholQR2` every 10 steps** for $n \ge 3072$ (which accounts for 61.5% of FLOPs and the worst rank-deficiency / conditioning!).
   - Option B6b (Everywhere): Run **pure `sCholQR2` every 10 steps** on all 96 factors (dropping warm refresh completely!).
2. **Cost (109M, all 96 factors):**
   - **B6b (pure `sCholQR2`-10):** ~125 ms + 5.1 ms transport = **~130 ms** every 10 calls = **13.0 ms/call** refresh overhead — **2.9 ms/call FASTER than current `warm10/QR50` (15.94 ms/call)**!
   - **B6a (`B1/B3` at $n \le 2304$ + `sCholQR2`-10 at $n = 3072$):**
     - Warm event (8%): $n \le 2304$ B1 (~17 ms) + $n=3072$ `sCholQR2` (~54 ms) + transport (~4 ms) = ~75 ms → $0.08 \times 75 = 6.0\text{ ms/call}$.
     - QR event (2%): all-96 `sCholQR2` (~130 ms) → $0.02 \times 130 = 2.6\text{ ms/call}$.
     - Total refresh overhead = **8.6 ms/call** (**1.85× faster** than current 15.94 ms/call!), while giving exact QR10 tracking on the largest, hardest ($3072\times 768$) factors!
3. **Expected effect:** B6b **replicates the BF16 QR10 trajectory identically** (zero warm-approximation gap, $r_C \approx 0.20$, zero call-500 gap!) while already beating your current `warm10/QR50` wall time! B6a gives ~90%+ of QR10 quality at nearly half the refresh cost. Confidence: **high**.
4. **Failure modes:** Same ridge/conditioning caveats as A5 (`sCholQR2`).
5. **Interaction with BF16 storage & $\beta_C=0.95$:** Exact match to BF16 QR10 (including column ordering and $V$ permutation every 10 calls).
6. **Cheap test:** Already covered by A5's replay + microbenchmark tests; validate B6b against the pending BF16 QR10 training run (reject if CE differs by $> 0.001$ or 109M `sCholQR2` event exceeds 160 ms).

---

# Pitfalls in the Current Design & Measurements

### P1. Triple Compound Defect in `warm_basis` (`-0.5`, global `0.01 s`, $\sqrt{n}$ Frobenius cap, equal-diagonal stall)
Detailed in Question B §0 **`[certain]`**. Critically, the defects **mask each other**: fixing global damping `0.01 s` (Defect 2) without replacing the Frobenius cap (Defect 3) will make $\|A\|_F/\sqrt{2}$ explode by $\sqrt{n}$ and *worsen* tracking! And `A = 0` at $d_i = d_j$ is directly responsible for your trained **max $r_C = 0.975$**.

### P2. Silent TF32 Truncation Inside FP32 Linear-Algebra Islands (`matmul precision = "high"`)
You set FP32 matmul precision to `"high"` (§1.3). Inside `qr_basis`, `torch.bmm(cov.float(), q.float())` is unaffected only because `cov, q` originate in BF16 (7-bit mantissa $\subset$ TF32's 10-bit mantissa). **However**, in your Cholesky-QR benchmark (§2.2), `Z = CQ` is full FP32, so `Z.T @ Z` silently truncates `Z` to TF32 ($u_{\text{TF32}} = 2^{-11} \approx 4.9\times 10^{-4}$) while `trsm` ($Z R^{-1}$) uses untruncated FP32 `Z`! Writing $\tilde{Z} = Z + \Delta Z$ ($\|\Delta Z\| \sim u_{\text{TF32}}\|Z\|$), $R^\top R = \tilde{Z}^\top \tilde{Z}$ leaves orthogonality error $\|(\Delta Z) R^{-1}\| \sim O(u_{\text{TF32}}\,\kappa(Z))$, directly explaining why Cholesky-QR's Gram error degraded to **0.034 at $n=4096$** and crashed at $n \ge 6144$ **`[certain]`**. The same silent TF32 truncation hits the `"Optional precision island"` (`torch.bmm(y.T, y)`) in `warm_basis` when `basis_dtype=fp32`! Every FP32 linear-algebra island must explicitly enforce IEEE FP32 (`fp32_precision="ieee"`).

### P3. BF16 Precision Traps in `Q^T C`, `Q` Storage, Unsymmetrized `C`, and Future $\beta$ Changes
1. **Intermediate BF16 rounding of $T = Q^\top C$ in `warm_basis` (`torch.bmm(qb.T, cb)`):** When $C$ has high $\kappa(C)$ (e.g. $\lambda_1 = 1, \lambda_n = 10^{-4}$) and $Q$ is misaligned by $\theta = 0.01$, row $n$ of $T = Q^\top C$ has norm $\sim \theta \lambda_1 = 10^{-2}$, so rounding $T$ to BF16 injects error $\sim 2^{-8} \cdot 10^{-2} = 4\times 10^{-5}$ — **40% relative error** on $d_n = 10^{-4}$ (and can even flip $d_n$ negative for $\lambda_n < 4\times 10^{-5}$!).
2. **BF16 storage of orthonormal $Q$ wastes 3 mantissa bits on unused exponent range:** Storing $Q_L, Q_R$ in **FP16** (same 2 bytes!) cuts the Gram error floor $8\times$ ($0.0024 \to 0.0003$).
3. **`qr_basis` never symmetrizes `cov`:** BF16 `baddbmm` isn't guaranteed bitwise symmetric; non-symmetry introduces tiny imaginary perturbations.
4. **BF16 EMA stagnation cliff (fragility under hyperparameter changes) **`[certain]`**:** At $\beta = 0.95$, $0.05 x$ is $\ge 6.4$ BF16 ULPs so $M, V, C$ decay cleanly (though new updates $< 0.078 |C_{ij}|$ round to zero). However, because BF16 relative half-ULP at $x \in [1, 2)$ reaches $2^{-9} \approx 0.00195$, **increasing any $\beta$ above $0.998$ (e.g. standard Adam $\beta_2 = 0.999$) completely freezes EMA decay (`R_b(beta * x) == x`)**, and even $\beta = 0.99$ severely distorts effective decay!

### P4. Refresh-After-Update Ordering + 50-Step Cold-Start Nullspace Wait
- At `k = 1`, $C_L$ for SwiGLU-in ($3072\times 768$) has rank $\le 768$: **2,304 eigenvalues are zero**. Your schedule then runs **10 full steps** using that single-minibatch eigenbasis before the first warm refresh, and **50 full updates** before the first QR repair ever affects a weight update!
- Also, because refresh happens *after* step 3 (`U` and `W` update), every refresh step $t \bmod 10 == 0$ pays to compute fresh $Q^+$ using $G_t$, yet applies the **10-step-stale** $Q$ to $G_t$ and double-projects $G_t$'s contribution into $M$ (via old $Q$, then via 4-GEMM transport into $Q^+$). Moving `covariance_ema` + refresh + transport **before** projecting $G_t$ removes 1 step of basis lag and avoids double-projecting $G_t$ into $M$.

### P5. Unmatched Controls, Synchronized Timing Bias, Thermal/Clock Drift, and Open-Loop Replay Blind Spots
1. **Conflated speed and loss baselines (§2.1, §2.3):** Comparing **compiled BF16 `warm10/QR50` (26.08 ms)** against **eager FP32 `QR10` (95.23 ms)** mixes three effects: **24.30 ms/call (35% of the entire 69.15 ms speedup)** comes solely from BF16 + `torch.compile` on the ordinary step, unrelated to warm refresh! Worse, because **BF16 QR10 is still pending**, you do not yet know whether the `+0.010–0.011` final CE gap is mostly BF16 state or mostly `warm10/QR50` (the FP32-warm pilot only checked call 500!). And LR (`1.5e-3`) was tuned solely on FP32 QR10.
2. **Synchronized event-class timing + RTX 5090 boost-clock bias:** Inserting host synchronizations per optimizer event prevents CPU-GPU overlap, while comparing a 37 ms `b1, n=8192` microbenchmark burst against a 546 ms sustained 575W `b12, n=8192` burst (or a low-power serial cuSOLVER `QR` event) on a consumer RTX 5090 can introduce **10–20% boost-clock/thermal bias** **`[speculation]`** unless GPU clocks are locked (`nvidia-smi -lgc`).
3. **Open-loop 1-layer covariance replay blind spots (§2.6(i)):** (a) Has no $V$ or loss, so it rewards aggressive tail rotations that destroy $V$ in closed-loop training; (b) 1-layer models have flatter spectra and milder drift than 12-layer transformers; (c) open-loop $C_t$ from QR10 misses positive feedback where stale bases corrupt $G_{t+1}$ and $C_{t+1}$.
4. **Peak VRAM wall at $n = 4096\text{–}8192$:** Holding multiple `(12, 8192, 8192)` FP32 intermediates (`b`, `a`, `mm32`, `gram`, `residual`, `y.float()`) simultaneously requires **~13–16 GB** transient VRAM on a 32 GB card — infeasible once you scale model size without A3's micro-batching.

---

# Final Synthesis

## Top 5 Ranking for A (Speed)
1. **A5 (`sCholQR2` replacing `torch.linalg.qr`):** Attacks the #1 bottleneck (45.3% of optimizer time), saving **~9 ms/call (~35% of total optimizer)** on 109M and **~1.7×–2×** at $n=8192$ while matching QR10 quality.
2. **A7 (Strided covariance EMA $k_C=2, \beta_C'=0.9025$):** Saves **~1.6–1.8 ms/call** across 100% of calls at trivial code complexity while actually reducing BF16 EMA rounding loss.
3. **A6 (Degree-4 exponential $R_4$ + direct 2-GEMM transport):** Cuts `warm_basis` FLOPs by **28.6%** across all compute-bound $n$ (including $n=4096\text{–}8192$) and halves warm transport (saving **~1.2 ms/call** on 109M).
4. **A3 (`warm_basis` scalar-pull + large-$n$ micro-batching):** Immediately recovers the measured **7–23% `b12` slowdown** at $n = 3072\text{–}8192$ and cuts peak transient VRAM by $3\times\text{–}12\times$ at zero mathematical cost.
5. **A2 (Contiguous group param/grad buffers + 60-batch $n=768$ + CUDA Graphs):** Eliminates ~700–850 MB/call of pack/copy overhead and the 0.80 ms small-$n$ launch floor (**~1.0 ms/call** on 109M, exact math). *(If `sCholQR2` is not adopted, **A1 Multi-Stream QR** jumps to #1 or #2 for QR-bound workloads.)*

## Top 5 Ranking for B (Tracking Quality)
1. **B1 (Full-step relative-gap $\arctan$-Jacobi + Rayleigh sort/$V$-perm + row-$\ell_2$/spectral cap + $R_4(a/2)^2$):** Fixes all four root bugs simultaneously while making `warm_basis` **15% faster**, turning linear 0.5× top-only decay into full-spectrum quadratic convergence.
2. **B3 (Top-$k$ exact Procrustes power-subspace deflation + B1):** Adds only **~0.06–0.25 of one GEMM** ($k=64$) to match QR10 identically on the top-$k$ subspace and permanently kill the max $r_C = 0.975$ rank-$r$ stall.
3. **B6 (`sCholQR2`-10 everywhere or at $n \ge 3072$):** Once A5 stabilizes Cholesky-QR, pure `sCholQR2`-10 beats current `warm10/QR50` wall time on 109M **with zero warm-approximation loss gap**, and strictly beats warm at $n \ge 6144$.
4. **B2 (Two-iteration local-frame Jacobi `B1 × 2`):** Quartic $O(r_C^4)$ convergence at ~1.3× current warm event cost, closing residual gap when B1 hits the angle cap or splits equal diagonals.
5. **B4 (Front-loaded early QR10 + exit cold-start early at $t=2\text{–}5$):** Eliminates the entire +0.124 call-500 gap at **zero extra QR budget** over 10k calls.

## Top 3 Pitfalls
1. **The coupled 4-way defect in `warm_basis` (`-0.5`, global `0.01 s`, $\sqrt{n}$ Frobenius cap, equal-diagonal stall):** Directly explains synthetic $r_C = 0.150$, trained max $r_C = 0.975$, and why fixing global damping without replacing the Frobenius cap breaks the update.
2. **Silent TF32 truncation inside FP32 islands (`matmul precision = "high"`):** Corrupts `Z.T @ Z` in Cholesky-QR (explaining the 0.034 Gram error and NaN failure) and silently degrades FP32 Gram checks unless IEEE FP32 is forced locally.
3. **Unmatched controls (missing BF16 QR10 + eager-FP32 timing baseline) & $r_C$ proxy blind spot:** Without BF16 QR10 you cannot separate BF16 loss from warm-refresh loss, and optimizing raw $r_C$ (94% dominated by the top 10 eigenvalues) ignores the bulk/tail correlations that govern Adam-whitened SOAP updates.

## What I Would Want Measured Before Refining Further (Up to 5 Items)
1. **Complete the matched `BF16 QR10` (compiled) 10,000-call control** (loss at 500, 2000, 10000 + synchronized event timings): establishes the true ceiling for both BF16 speed and BF16 loss, separating BF16-state error from warm-refresh error.
2. **Isolate the `b12` vs `b1` 22% regression at $n = 6144, 8192$ with locked GPU clocks (`nvidia-smi -lgc`) and Nsight Compute/profiler:** tells us whether the gap is thermal throttling, cuBLASLt batched kernel selection under deterministic mode, or FP32 intermediate HBM traffic.
3. **Benchmark `sCholQR2` (A5, TF32 OFF on `Z^T Z`, $\tau \in \{10^{-4}, 3\times 10^{-4}, 10^{-3}\}$) across $n = 256\text{–}8192$ ($b=1, 12$):** checks zero-NaN stability, BF16 Gram error $\le 0.0025$, and exact wall time — if 109M `sCholQR2` lands at $\le 130\text{ ms}$, it reshapes the entire warm-vs-QR trade-off.
4. **Inspect the specific factors hitting $r_C \approx 0.975$ at ages 1990 and 9990** (which layer/shape, eigenvalue spectrum of $C$, diagonal spectrum of $B$, and off-diagonal structure): confirms whether those stalls are equal-diagonal spikes / level crossings and sets the right $k$ for B3.
5. **Replay-harness sweep of B1 ($\delta \in \{0.05, 0.1, 0.2\}$), B2, and B3 ($k \in \{32, 64\}$) reporting BOTH whitened RMS correlation / KL gap AND post-permutation tail churn $\chi$:** ensures we don't pick a warm refresh that wins on $r_C$ by scrambling noisy tail eigenvectors and hurting $V$ in real training.

---

# Document 3: Reviewer X's stage-1 answer (verbatim, anonymized)

## 0. Where the time goes (from §2)

**Per-call optimizer mean, 26.08 ms.**

| component | arithmetic | ms/call | share |
|---|---|---|---|
| ordinary base, every call | 10.14 | 10.14 | 38.9% |
| warm increment | 0.08 × 51.47 | 4.12 | 15.8% |
| QR increment | 0.02 × 591.04 | 11.82 | 45.3% |

- **Amdahl.** The optimizer is 26.08/309.5 = 8.4% of the step, and the refresh increments are 15.94 ms = 5.1%. Forward/backward is about 283 ms, eager. No refresh speedup can buy more than about 5% of step time; the larger prize is the +0.010 loss gap.
- **Inside a warm event (49.84 ms).** 12×2.46 = 29.5 ms (n=3072), 12×1.02 = 12.2 (2304), 12×0.331 = 4.0 (1536), 60×0.069 = 4.1 (five n=768 groups). The two tall left factors are 84%.
- **Inside a QR event (about 598 ms).** 219.4 (3072) + 143.5 (2304) + 81.4 (1536) + 5×30.8 = 154 (768).
- **Flop rates** (one n×n product = 2n³ flops):
  - Warm is 7 products: QᵀC, ·Q, Q·A, and 2×(YᵀY, Y·E). At n=3072 that is 14·2.90e10 = 4.06e11 flops in 2.46 ms, about 165 TFLOP/s. It is compute-bound for n ≥ 2048 (same rate at 8192).
  - Householder QR is (4/3 + 4/3)n³ for geqrf + orgqr, plus 2n³ for Z. At n=3072 that is 1.35e11 flops in 18.1 ms, about 7 TFLOP/s; at n=768 about 0.8 TFLOP/s.
  - So QR does about 3× fewer flops than warm at about 25–200× lower throughput, and its per-factor time does not improve with batching. QR is latency-bound at every size you run. That is the main speed lever.
- **Ordinary step.** Six mixed products per matrix (2 project, 2 back-project, 2 covariance). I count about 1.3e12 flops over the 48 matrices, about 7.9 ms at 165 TFLOP/s against 10.14 measured. It is already about 80% matmul-bound, so there is little engineering slack.
- **Attribution.** A compiled BF16 QR10 arm would cost about 10.14 + 0.1×591 = 69.2 ms/call. The same formula predicts 39.7 for QR20 (measured 40.46) and 22.0 for QR50 (measured 22.86). Of the 95.2 → 26.1 ms headline, about 26 ms is compile+BF16 engineering that the baseline could also have, and about 43 ms is the schedule change.

---

## 1. Four identities the candidates rest on (all derived)

**I1. QR repair is one unshifted QR-algorithm step on B = QᵀCQ.**
If QQᵀ = I then CQ = QB, so qf((CQ)P_π) = Q·qf(B P_π). (Textbook equivalence of subspace iteration and the QR algorithm [verify][certain].)

**I2. First-order form of that step.**
- Write B = D + E, sorted so d_j ≥ d_i for j < i, and qf(B) = I + K with K skew.
- (I+K)(D+R′) = D+E gives, on the strict lower triangle, K_ij d_j = E_ij, so K_ij = B_ij/d_j for i > j.
- The exact first-order diagonalizer in your convention (B⁺ = B + BA − AB) is A*_ij = B_ij/(d_j − d_i).
- So QR moves each pair a fraction 1 − d_min/d_max of the Jacobi angle:

  **A^QR_ij = B_ij · sgn(d_j − d_i) / max(d_i, d_j)**

- Properties: |A^QR_ij| ≤ √(d_min/d_max) ≤ 1 for PSD B; no gap singularity; scale-free; ties are broken by column order, not by a gap.

**I3. The warm generator, relative to QR.**
Your A_ij = A*_ij · ½ · Δ²/(Δ² + (0.01s)²), with Δ = d_i − d_j, times the cap a. The ratio of rotation angles is

  warm/QR = (a/2) · d_max·|Δ| / (Δ² + (0.01·s)²).

- Well-separated pair (Δ ≈ d_max ≫ 0.01s): ratio = a/2 ≤ 0.5.
- Tail pair (d_max = 10⁻³s, Δ ≈ d_max): ratio = (a/2)·0.0099, a 100–200× under-rotation.
- Near-degenerate head pair (d_max = s, Δ = 0.1s): ratio ≈ 5a.
- Consequence: the damping is absolute in units of λ_max. With λ_i ∝ 1/i, every pair with both i, j ≳ 100 is effectively frozen. On that block, warm10/QR50 equals QR50.

**I4. The Frobenius cap is about √n too tight.**
- For skew A, ‖A‖₂ ≤ ‖A‖_F/√2, tight only at rank 2.
- For unstructured A with entry std e: ‖A‖_F/√2 ≈ e·n/√2 and ‖A‖₂ ≈ 2e√n. The slack is √n/(2√2): about 10× at n=768, 20× at 3072, 32× at 8192. Real A is structured, so the true slack needs logging.
- Consistency check on §2.2: for the synthetic inputs I estimate ‖A‖_F ≈ 1.4 (head-versus-all pairs only), so a ≈ 0.5. With the ½ factor that is about ¼ of the Jacobi step, predicting r_C 0.20 → 0.15. You measured 0.150. So the weak single-step quality is explained by ½ × cap, not by first-order error.

**NS and retraction scalars** (σ = singular value of Y, E = σ² − 1):
- One Newton–Schulz step maps E ↦ −¾E² + ¼E³ exactly.
- Y0 = Q(I + aA) has E0 = θ² for a rotation plane with angle θ = a·σ(A). At the cap θ = 0.5: 0.25 → −0.043 → −1.4e-3. That is why two NS steps are needed today.
- Y0 = Q(I + aA + ½a²A²) has E0 = θ⁴/4, since (I − A + A²/2)(I + A + A²/2) = I + A⁴/4. At θ = 0.5: 0.0156 → 1.8e-4 in one step. At θ = 0.7: 0.060 → 2.7e-3 in one step.

**BF16 floor.** If entries carry relative rounding error of RMS σ_u, then ‖QᵀQ − I‖_F/√n ≈ √2·σ_u. Your 0.0024 implies σ_u ≈ 1.7e-3, matching an 8-bit significand (half-ulp between 2⁻⁹ and 2⁻⁸ relative). No amount of NS gets below this.

---

## 2. Question A: speed

### A1. Stabilized Cholesky-QR in place of Householder QR (same as B1)

1. **Update.**
   - Z = 𝓕(C,Q) + τ·Q, with τ = ρ·max_j d_j and d = diag(QᵀCQ) from the same contraction as now.
   - Sort columns by d; scale columns to unit norm (this does not change qf).
   - G = ZᵀZ in true FP32; L = cholesky_ex(G + s·I); Y = Z·L⁻ᵀ by triangular solve.
   - Round to BF16, then 1–2 of your existing NS corrections.
   - Fallback when `info ≠ 0`: `torch.where` to the warm result with the identity order (stays in-graph).
2. **Cost.** Z: 1 product (BF16 operands, FP32 accumulate). Gram: 1 FP32 product. Cholesky: n³/3 flops (1/6 product). Triangular solve: n³ flops (1/2 product). That is about 2.67 product-equivalents, plus 2 BF16 products per NS step. Your table already prices Z + one pass: 0.171 / 0.618 / 1.50 / 3.03 ms per factor, i.e. 60×0.171 + 12×(0.618+1.50+3.03) = 72 ms per event against 598.
3. **Effect.** A QR event drops from about 601 ms to about 95–110 ms with one NS step. At the QR50 cadence that saves about 10 ms/call, 38% of the optimizer mean and 3.2% of the step. Confidence: high on timing (your own table); medium on robustness on trained C.
4. **Failure modes.**
   - Unshifted, it breaks when κ(C)²·u ≳ 1, i.e. κ ≳ 4100 in FP32. With λ ∝ 1/i, κ ≈ n, which predicts failure between 4096 and 6144, exactly where you saw NaN.
   - The ridge τ bounds κ ≤ 1/ρ. ρ = 1e-3 gives κ²u ≈ 0.06, so a single pass plus NS is enough. ρ = 1e-4 needs a second Cholesky pass (roughly +60% cost).
   - The ridge slows rotation among directions with d < τ: the gain becomes 1 − (d_i+τ)/(d_j+τ).
   - If the Gram product runs in TF32 (your "high" setting permits it [verify][recalled]), it will fail far earlier. Force true FP32 for that one product.
5. **BF16 and β_C.** BF16-stored C already has a rounding-noise floor that I estimate at 1e-4·λ₁ or higher, so a ridge near that level costs no fidelity. BF16 inputs are exactly representable in TF32, so Z can use any kernel.
6. **Test.** Replay: Cholesky-QR10 with ρ ∈ {0, 1e-4, 1e-3, 1e-2} against QR10. Reject if top-k overlap or normalized r_C differs from QR10 by more than the QR10-versus-eigh10 difference, or if any factor falls back more than 1% of the time. Microbenchmark at all n, including 6144 and 8192.
   - Literature: CholeskyQR2 (Yamamoto et al. 2015) and shifted CholeskyQR3 (Fukaya et al. 2020) [verify][recalled]; near-optimality of column equilibration (van der Sluis 1969) [verify][recalled].

### A2. R-only Householder plus triangular solve (robust fallback for A1)

1. **Update.** R = `torch.linalg.qr(Z_π, mode='r').R`; Y = Z_π·R⁻¹ by `solve_triangular`; one NS step. Same sign convention as the full QR.
2. **Cost.** geqrf only (4n³/3) plus the solve (n³), instead of geqrf + orgqr (8n³/3). All FP32.
3. **Effect.** If orgqr is about half of the QR time, about −40% per QR event (about −235 ms, −4.7 ms/call). Medium confidence; the geqrf/orgqr split is unmeasured. `mode='r'` skipping Q formation is [verify][recalled].
4. **Failure.** Orthogonality loss scales as κ(Z)·u rather than κ²·u. It is fine to κ ≈ 1e5 with one NS step; use a ridge ρ ≈ 1e-5 beyond that.
5. **BF16.** Same as A1.
6. **Test.** Microbenchmark time. Replay Gram error and overlap against QR10; reject if the Gram error after one NS step exceeds 0.004.

### A3. Second-order retraction with one NS step (6 products instead of 7)

1. **Update.** K = aA + ½a²·𝓕(A,A); Y0 = R_b(Q + 𝓕(Q, R_b(K))); one NS step.
2. **Cost.** 2 (B) + 1 (A²) + 1 (Q·K) + 2 (NS) = 6 BF16-operand products.
3. **Effect.** −14% on compute-bound sizes: about −6.5 ms per warm event, −0.5 ms/call. At n=8192, 45.5 → about 39 ms per factor. Alternatively keep 7 products and raise the cap to θ ≈ 1 (0.25 → 0.043 → 1.4e-3, the same worst case as today). High confidence in the algebra.
4. **Failure.** A² rounded to BF16 is harmless (relative error). If the cap estimate is low, NS still converges for θ⁴/4 < 2.
5. **BF16.** The single NS step also repairs the stored basis's own 0.002 drift.
6. **Test.** Microbenchmark time. Replay Gram error ≤ 0.003, and r_C no worse than current at an equal cap.

### A4. Run the per-factor decompositions concurrently (speculative)

1. **Update.** Unchanged arithmetic. Dispatch the 96 FP32 decompositions over 4 host threads, since the batched call loops and each is latency-bound. Variant: overlap the QR of a C snapshot with the next forward/backward and swap bases 1–2 calls late.
2. **Cost.** Same flops.
3. **Effect.** Guess 1.5–3× on the n ≤ 1536 groups (235 of 598 ms). Low confidence.
4. **Failure.** Kernels serialize on one stream; the decomposition leaves the compiled region; the overlapped variant changes the math by a 1–2 call basis lag.
5. **BF16.** None.
6. **Test.** Microbenchmark only: thread pool over 12 factors at each n. Reject below 1.3×.

### A5. Remove pack/unpack traffic and launch latency in the ordinary step

1. **Update.** Hold each group's 12 weights as views of one stacked leaf, so gradients arrive stacked and `torch.stack` / `copy_` disappear. Batch refreshes by factor size (one batch-60 call for all n=768 factors). Try `mode="reduce-overhead"` for the small-n regions [verify][recalled].
2. **Cost.** Removes about 1.5 GB/call of copies and 4 of 5 latency-bound warm calls.
3. **Effect.** My guess is 1–2 ms on every call, 10–20% of the ordinary base. Low-medium confidence; it rests on the 10.14 − 7.9 ms non-matmul remainder.
4. **Failure.** A model-code change; CUDA graphs may reject in-place parameter writes.
5. **BF16.** None.
6. **Test.** Microbenchmark of the ordinary region with pre-stacked inputs. Reject below 0.5 ms.

### A6. Cheaper covariance accumulation (equation-level, risky)

1. **Update.** Accumulate every second call with C ← β²C + (1−β²)GGᵀ, or sketch with an n×r Gaussian Ω: C ← βC + (1−β)(GΩ)(GΩ)ᵀ/r.
2. **Cost.** Saves about 1/3 of ordinary-step flops on the affected calls. The sketch replaces m²n by mnr + m²r.
3. **Effect.** About 1–1.5 ms/call. Halves the effective sample count in C.
4. **Failure.** Noisier bulk eigenvectors.
5. **BF16.** With β² = 0.9025 the BF16 EMA is safer, not worse.
6. **Test.** Not testable in replay (needs per-step gradients). Smallest test is a 2,000-call run.

**Larger factors (n = 4096–8192).** Warm is flop-bound (14n³ at 165 TFLOP/s) and already costs 0.2–0.4 of a QR. The useful levers are A3 (6 products), A1 (5.3n³ flops; 42 ms measured against 45.5 for warm at 8192, once the NaN is fixed) and B5 (about 4.2 products).

---

## 3. Question B: tracking QR10

### B1. Use the real power step with Cholesky orthogonalization at every refresh

1. **Update.** A1's refresh every 10 ages, with the sort and V permutation exactly as in your QR path. No warm step.
2. **Cost.** As A1: about 86–100 ms per event with 1–2 NS steps, against 49.8 for warm. That is 1.7–2.0× warm for the 109M set and equal to warm for n ≥ 4096. At a 10-age cadence it is about 9–10.5 ms/call including transport, against 15.9 ms/call for today's warm10/QR50 mix.
3. **Effect.** In exact arithmetic this is the QR10 trajectory, since Y = Z·R⁻¹ is the same Q as Householder. It should remove the schedule part of the gap while being cheaper than the current schedule. Confidence: medium-high, conditional on the ridge and shift working on trained C.
4. **Failure.** As A1. A ridge that is too large freezes the tail and reintroduces the gap.
5. **BF16.** After BF16 storage the basis sits at the same 0.0024 Gram floor as QR.
6. **Test.** Replay as in A1, on all metrics. Then one 2,000-call run against BF16 QR10; reject if CE at 2,000 is more than about 0.002 (seed noise) above it.

### B2. Spectral-norm cap instead of the Frobenius cap

1. **Update.** Estimate σ = ‖A‖₂ with 8 power iterations on −A² (FP32 matvecs, fixed start vector). Set a = min(1, θ_cap/(1.15σ)).
2. **Cost.** 16 matvecs = 32n² flops; negligible. No new products.
3. **Effect.** If the cap binds (I4 says a ≈ 0.5 synthetically, smaller in training where r_C ≈ 0.39), the step roughly doubles or more. Predicted single-step r_C 0.20 → about 0.10–0.11. Confidence: medium; logging a settles it.
4. **Failure.** Power iteration underestimates σ. NS still converges for θ < √2, so the cost is a larger Gram error for one refresh.
5. **BF16.** None.
6. **Test.** Replay r_C and normalized r_C after each refresh. Reject if a was already 1 in more than 90% of refreshes.

### B3. QR-matched, scale-free generator

1. **Update.** A_ij = −B_ij·sgn(d_i − d_j)/max(d_i, d_j, ρ·s). This is I2 in your sign convention. At exact ties use sgn(j − i). Combine with B2 and A3.
2. **Cost.** Elementwise only; same 7 (or 6) products.
3. **Effect.** Matches QR's per-pair gain to first order. It removes the factor ½, unfreezes the tail block, and removes the equal-diagonal stall: your 2×2 example gives |A_12| = 0.5 instead of 0. I expect the diagonally normalized r_C to move most of the way from warm to QR10. Confidence: medium.
4. **Failure.** First order only. A dominant direction that no column captures needs a large rotation and gets capped (B5 covers that). The floor ρ must sit at or above C's noise level, or noise pairs rotate.
5. **BF16.** Choose ρ at or above 1e-4 for BF16-stored C.
6. **Test.** Replay distance of the chained basis to the QR10 chain, ‖Q_QR10ᵀ Q_test‖ in bands of eigen-index, plus normalized r_C. Reject if the tail bands are no closer to QR10 than the current warm.

### B4. Head-block power step with deflation

1. **Update.** Take the k = 64–128 columns with the largest d. Z_k = (C + τI)·Q[:, idx]; H = thin CholeskyQR2(Z_k); T = Q_rest − H(HᵀQ_rest); write H and T back to their slots; two NS steps on the whole Y.
2. **Cost.** About 4 + 3k/n full products alone, or about 7.2 when appended to the generator step.
3. **Effect.** Captures a dominant direction in one shot, as the power step does. It targets the factors at r_C ≈ 0.97. Confidence: medium on the mechanism; unknown size.
4. **Failure.** T's singular values are the cosines of the head-subspace rotation. A near-90° rotation leaves a small singular value that NS grows only 1.5× per step.
5. **BF16.** The thin step runs in FP32 on n×k, which is cheap.
6. **Test.** Replay top-k subspace overlap after the refresh. Reject if it is not within 0.02 of QR10 on the worst decile of factors.

### B5. Front-loaded schedule, optionally with a trigger

1. **Update.** QR, or B1, every 10 ages for ages ≤ 200–500, then the normal schedule. Optionally promote a factor to QR when a < 0.5.
2. **Cost.** 16 extra QR events for ages ≤ 200 is about 16×540 ms = 8.6 s once, about 0.3% of a 10k run.
3. **Effect.** It tests whether the call-500 gap is a cold-start effect. The basis starts from a single-gradient eigendecomposition and your first QR is at age 50, while the reference has had four. Confidence: medium that it removes most of the +0.12; low that it changes the +0.010 plateau.
4. **Failure.** None numerically. A data-dependent trigger breaks the static graphs.
5. **BF16.** None.
6. **Test.** Replay metrics over ages 0–200. The loss effect needs a 500–2,000-call run.

### B6. Deadband-free basis storage (minor)

1. **Update.** Stochastic rounding for Y, or an FP32 master basis with a BF16 shadow for the hot path.
2. **Cost.** O(n²) elementwise, or +0.96 GB.
3. **Effect.** With round-to-nearest, an entry moves only if its increment exceeds half an ulp. Column rotations below about 2⁻⁹–2⁻⁸ rad are dropped and never accumulate. I estimate only a few percent of rotation energy is lost at your caps, so this is small.
4. **Failure.** Stochastic rounding raises the Gram floor slightly.
5. **BF16.** This is the BF16 issue itself.
6. **Test.** Replay with your `basis_dtype=float32` switch. Reject if the metrics are indistinguishable.

### B7. Refit V's marginals after a refresh (speculative, goes beyond QR10)

1. **Update.** With β₂ = β_C and orthogonal bases, Σ_j V_ij = diag(Q_LᵀC_LQ_L)_i and Σ_i V_ij = diag(Q_RᵀC_RQ_R)_j, because Σ_j Ḡ_ij² = (Q_LᵀGGᵀQ_L)_ii. After a refresh, rescale V's rows and columns to the new diagonals, using d⁺_i ≈ d_i + 2a·Σ_k B_ik·A_ki.
2. **Cost.** O(mn + n²) elementwise.
3. **Effect.** Unknown.
4. **Failure.** Multiplicative rescaling amplifies noise in small rows.
5. **BF16.** Fine at β = 0.95.
6. **Test.** Not testable in replay; needs a training run. As a free diagnostic, log the row sums of V against d as a V-staleness measure.

### The three questions you asked me to address

- **Is r_C the right proxy?** No, by I3. r_C weights by λ², so it sees only the head. The update is Adam-normalized, so every rotated coordinate counts about equally.
  - Evidence: on the toy, warm closes (0.125 − 0.093)/(0.125 − 0.075) = 64% of the r_C gap. On loss it closes 12–37%.
  - Use instead: (a) the correlation form R = D^(−½)·B·D^(−½), reported as ‖R − I‖_F/√n and −log det R / n; (b) distance to the QR10 chain by eigen-index band; (c) in replay, the cosine between idealized updates U(Q) = Q_L[(Q_LᵀGQ_R) ⊘ √(d_L d_Rᵀ)]Q_Rᵀ under the test and QR10 bases, with synthetic G = C_L^(½)·X·C_R^(½).
  - Validate by checking that the metric orders QR10 < QR20 < warm < QR50 as the loss does.
- **The call-500 gap.** It appears with FP32 state too, so it is schedule-driven. At call 500 the loss is still falling steeply, so +0.12 is probably a delay of tens of calls. Report "calls behind" rather than ΔCE. The persistent +0.007 to +0.012 from call 2,000 is the steady-state tracking deficit and is the number to optimize. B5 separates the two.
- **V and column order.** Leaving V unchanged at a warm step is correct to first order: the diagonal second moment changes by Σ_k A_ki²(V_k − V_i), which is second order, while M changes at first order and is transported. Not reordering is also correct for a continuous rotation, since each slot keeps its V. Order only matters for triangular steps (QR, B1, B3's tie-break). The second-order V error is largest for head-to-tail rotations, and is inherited from SOAP, which only permutes V [verify][recalled].

---

## 4. Pitfalls

1. **The warm step is head-only and quarter-gain** (I3 and I4): a global, √n-loose cap times ½, and damping tied to λ_max. One badly placed pair also throttles the whole factor.
2. **Attribution is confounded.** The reference is eager FP32. LR 1.5e-3 was tuned on it only, so the gap is biased against the variant. The matched BF16 QR10 run is pending, so the BF16/schedule split of +0.010 is unknown. The short controls are one seed with differences (0.005) near seed spread (about 0.002–0.003). CE at call 2,000 of a 10k cosine schedule is not the target horizon.
3. **BF16 state has deadbands that the replay harness cannot see.**
   - The EMA update is rounded away when (1−β)|X − C| is under half an ulp, i.e. |X − C|/|C| below 3.9–7.8% at β = 0.95.
   - Pure decay stops entirely for β ≳ 0.996, so β₂ or β_C cannot be raised under BF16 without stochastic rounding.
   - Replay C comes from an FP32 run with one rounding. It contains no accumulated-BF16 effects, no closed loop, and no M or V.
4. **The small-rotation premise is weak at β_C = 0.95.** 1 − 0.95¹⁰ = 40% of C's mass is new at each refresh and 87% is new between QRs. B⁺ ≈ B + BA − AB is not a safe assumption in early training.
5. **Stall points are wider than the 2×2 example.** A rank-1-dominated C spread evenly over columns is also a stall or slow point. The 0.965–0.975 maxima suggest this happens in training.
6. **"FP32 island" may mean TF32.** Harmless for C·Q on BF16-exact inputs; not harmless for any Gram, r_C or Cholesky computation on true FP32 data.
7. **Timing method.** Per-call synchronization changes overlap. The QR time is host-latency-sensitive and you have 4 CPU threads. Timing compares across launches. Microbenchmark medians hide tails.
8. **Minor.** QR uses an unsymmetrized C. ε = 1e-8 should be checked against √V in tail coordinates late in training. M transport error from the 0.2% non-orthogonality is negligible.

---

## 5. Rankings

**Top 5 for A (speed)**
1. A1, stabilized Cholesky-QR: removes about 85% of the 45% QR share, using timings you already measured.
2. A2, R-only Householder plus solve: robust, about −40% of QR time, with almost no numerical risk.
3. A3, 6-product retraction: a free 14% on the compute-bound warm sizes and the main lever at n ≥ 4096.
4. A5, pre-stacked storage and size-batched refreshes: the only item that touches the 39% ordinary base.
5. A4, concurrent decompositions: possibly large for small n, but speculative.

**Top 5 for B (tracking)**
1. B1, Cholesky-QR at every refresh: it is the target arithmetic, inside the cost budget and cheaper than the current mix.
2. B3, QR-matched generator: fixes the gain profile and the stall with no extra products.
3. B2, spectral cap: a one-line change that likely doubles the step.
4. B5, front-loaded schedule: about 0.3% cost, and it isolates the call-500 gap.
5. B4, head-block power step: aimed at the r_C ≈ 0.97 factors.

**Top 3 pitfalls:** 1 (head-only, quarter-gain warm step, invisible to r_C), 2 (confounded attribution and tuning), 3 (BF16 deadbands outside the harness's view).

---

## 6. What I would want measured first

1. The matched BF16 QR10 run, with per-factor r_C distributions (not only means) for both arms, and which factors sit near 0.97.
2. Per refresh and per factor: the cap scalar a, ‖A‖₂ against ‖A‖_F/√2, and the realized rotation by eigen-index band for warm and for QR.
3. The scale-free metrics above in replay, and whether they rank the four schedules as the loss does.
4. The geqrf/orgqr time split; Cholesky failure rate and κ(C) on trained snapshots against the ridge; TF32 on and off.
5. A front-loaded run to 2,000 calls, and a small LR sweep for the variant at the full horizon.

---

# Document 4: Reviewer Y's stage-1 answer (verbatim, anonymized)

I would attack the **QR implementation first for speed**, and the **choice of rotation first for loss**. A QR repair costs an extra 591 ms when it fires. The current warm rotation, meanwhile, approximates a step toward an eigenbasis with a coefficient proportional to an inverse *gap*. The first-order rotation made by ordered QR has a different coefficient, proportional to the inverse *leading column’s Rayleigh quotient*. That difference gives a direct replacement to test on your snapshots.

The derivations below are my own. Timings and empirical observations come from your prompt. Speed projections are estimates, and none of the proposed loss improvements are established.

## Where the time goes

Using the shared-window event means, the BF16 optimizer costs

\[
10.14 + 0.08(61.61-10.14)+0.02(601.18-10.14)
=10.14+4.12+11.82=26.08\ \mathrm{ms/call}.
\]

Thus the **QR increment accounts for 45%** of optimizer time, the work common to all calls for 39%, and the warm increment for 16%. Halving each component separately would save 5.91, 5.07, and 2.06 ms/call, respectively. Even eliminating the entire optimizer would improve measured whole-step speed by only \(309.5/(309.5-26.08)=1.092\times\), assuming the rest of the step stays fixed. The *achieved* whole-step speedup over the eager reference is \(380.5/309.5=1.229\times\).

For a matrix \(m\times n\), write \(P=m^2n+mn^2\), and count a multiply-add as two FLOPs. The ordinary projection and unprojection each cost \(2P\). The two full covariance products together cost another \(2P\). The ordinary step therefore has **six GEMMs and \(6P\) leading FLOPs**. Moment transport adds four rectangular GEMMs and \(4P\). For a square factor of order \(d\), warm uses **seven** dense products, two for \(B\), one for \(QA\), and two per polar correction. That's \(14d^3\) leading FLOPs. Its speed advantage over QR comes from the kernels and their parallelism, despite this substantial arithmetic bill.

Two details change the priorities by size. At batch 12, five 768 groups alone consume about \(5(30.8)=154\) ms per QR event. They are worth optimizing alongside the 219 ms 3072 group. For warm, a whole small batch takes about 0.8 ms, whereas at 8192 a *single factor* takes 45.5 ms amortized at batch 12. Eliminate launches at the small end and dense products at the large end. At 6144 and 8192, batch 1 even has better warm time per factor than batch 12 in your table.

## A. Speed

### A1. Remove ordinary-step packing and fuse the GEMM boundaries

1. **Update.** Keep the ordinary equations. Give same-shape parameters and gradients stable views into contiguous group storage where the model permits it. Read those views directly and write the FP32 master weights in place. Fuse the output of the second projection with the FP32 moment updates and formation of the BF16 normalized direction. Fuse the final FP32-output GEMM with the weight update where the backend supports the epilogue. Preserve every specified BF16 rounding point, including the rounded projected gradient used for its square.
2. **Cost.** Still six products and \(6P\) FLOPs per matrix. The two covariance products and the first three projection/unprojection products have BF16 outputs. The final unprojection has FP32 output or its equivalent inside the epilogue. Savings come from eliminated copies, launches, and large intermediate reads and writes. For example, physically stacking and then copying back the FP32 parameters alone entails two copies, each reading and writing \(4mn\) bytes, or \(16mn\) bytes per matrix. Credit that saving only if the compiled region actually materializes both copies.
3. **Expected effect.** High confidence in the priority of measuring this, low confidence in a specific speedup. A 10% reduction of the common 10.14 ms is about 1 ms on *every* call. Twelve full graphs establish capture coverage. They do not establish which buffers or library calls Inductor eliminated. I would compare the existing graphs, stable-address CUDA graph replay where applicable, and a custom epilogue only for the boundaries that still materialize.
4. **Failure modes.** Gradient hooks, clipping, accumulation, or model kernels can force a copy elsewhere. A slow custom GEMM can cost more than its epilogue saves. In-place writes must respect all old-state reads. Changing when an intermediate rounds changes the experiment.
5. **BF16 and \(\beta_C\).** Keep BF16 operands, FP32 accumulation, and the existing covariance EMA. There is no intended interaction with its 0.95 decay. Keep `emulate_precision_casts=True` for the control.
6. **Cheap test.** Use the event-region microbenchmark with production strides and aliasing. Compare stored states and FP32 parameter deltas from identical inputs. Reject if the common event does not improve beyond timing variation, or if savings move into gradient preparation outside the measured region. The latter requires a short in-training timing window.

### A2. Use symmetric products for covariance and polar Grams

1. **Update.** Compute one triangle of each \(GG^\top\) and \(G^\top G\), apply the EMA there in FP32, round once, and mirror into the full BF16 tensor. In warm, compute one triangle of each \(Y^\top Y\), form the residual, and mirror for the correction GEMM. Use a tiled symmetric kernel so diagonal tiles and the final mirror do not dominate. The two products forming \(Q^\top C Q\) are general GEMMs in the supplied evaluation order.
2. **Cost.** For the two covariance factors the number of computed dot-product terms falls from \(P\) to
   \[
   \tfrac12m(m+1)n+\tfrac12n(n+1)m=P/2+mn.
   \]
   This saves \(P-2mn\) FLOPs, about one sixth of the ordinary step's dense arithmetic. For each warm Gram, \(d^3\) dot-product terms fall to \(d^2(d+1)/2\). Across two Grams this saves about one full GEMM equivalent out of seven. Inputs are BF16, Gram outputs FP32, covariance storage BF16. Mirroring is quadratic traffic.
3. **Expected effect.** Most promising for large warm factors and large ordinary covariance products. The arithmetic ceilings are about 17% of ordinary and 14% of warm FLOPs. Wall-time savings can be much smaller. At small \(d\), I would favor fusing residual formation and mirroring over dispatching an additional library operation.
4. **Failure modes.** A highly tuned full GEMM can beat a triangular kernel. Changing the reduction tree can change rounding. Mirroring the EMA presumes the old state is symmetric, which should be verified once at entry.
5. **BF16 and \(\beta_C\).** Apply \(\beta_C C+(1-\beta_C)GG^\top\) before the storage cast. Rounding the product first would introduce an extra error every call. Symmetric storage does not guarantee positive semidefiniteness.
6. **Cheap test.** Time both covariance products and both Gram corrections at actual shapes inside their event regions. Reject size by size if event time does not fall. Compare EMA results and Gram residuals with the original implementation.

### A3. Implement ordered QR as a genuinely batched, blocked Householder operation

1. **Update.** Retain the existing \(Z=CQ\), score contraction, sort, and gather. Factor each ordered \(Z\) in panels, accumulating a panel's reflectors as \(H=I-YTY^\top\). Apply \(H\) or \(H^\top\) to the trailing matrix by two rectangular GEMMs around a small triangular multiply. Form the explicit basis by applying the panels in the appropriate reverse order to the identity. Batch independent factors at each panel index. Compare with a small number of concurrent library QR calls if a custom implementation is expensive.
2. **Cost.** One FP32 \(d\times d\) product for \(CQ\), \(O(d^2)\) contraction and gather, sort, QR factorization, explicit Q, and one BF16 cast. For the leading count, a reflector on a trailing \(j\times j\) square costs about \(4j^2\) FLOPs for a dot and a rank-one update. Summing gives \(4d^3/3\) for factorization and approximately the same for forming Q using its trailing structure. Including \(CQ\) gives roughly \((2+4/3+4/3)d^3\). Blocking reorganizes the work. Keep the panel, norms and reflector arithmetic FP32. Check the actual matmul precision of the trailing updates.
3. **Expected effect.** Highest speed priority. Halving the measured QR increment would save 5.91 ms/call. That is a target, not a prediction. The near absence of a batch-12 gain for QR gives a specific opportunity at small sizes. At 3072–8192, panel dependency and single-factor utilization matter more.
4. **Failure modes.** Launching 12 streams need not help if one QR already occupies the GPU. Poor panel kernels, workspace pressure, or an expensive explicit-Q phase can erase the gain. Handle rank deficiency and agree on sign and tie conventions when validating. Sign differences alone need not affect the optimizer if coordinates are handled consistently.
5. **BF16 and \(\beta_C\).** Factor the same FP32-expanded state and round only the final Q. The BF16 orthogonality floor gives little reason to demand bitwise agreement between two stable QR implementations. Ill-conditioned input still makes *trailing directions* sensitive. The short EMA makes those directions change between repairs.
6. **Cheap test.** Split existing QR time into \(CQ\), scores/gather, factorization, explicit Q, and cast. Sweep panel width and concurrency in the microbenchmark. Replay with sign-aligned column and ordered-prefix comparisons. Reject if end-to-end QR time fails to improve or prefix agreement degrades beyond ordinary FP32 implementation differences.

### A4. Drop the second polar correction where one suffices

1. **Update.** Return \(Y_1\) on warm events for the factor groups that pass replay. Keep the two-correction arm as the control. A static choice by size/group avoids a device-to-host decision on every refresh.
2. **Cost.** Five dense products instead of seven. Remove one BF16-input/FP32-output Gram, one BF16-input/FP32-output correction product, its residual arithmetic, and a basis cast. To see what is lost, write \(E=Y^\top Y-I\). In exact arithmetic the correction is \(Y'=Y(I-E/2)\), hence
   \[
   E'=(I-E/2)^2(I+E)-I=-3E^2/4+E^3/4.
   \]
   For orthogonal input \(Q\) and applied skew \(K\), the initial \(E=K^\top K\). A real skew matrix has paired singular values, so the present cap gives \(\|K\|_2\le\|K\|_F/\sqrt2\le0.5\). Thus \(0\preceq E\preceq0.25 I\). At the extreme eigenvalue the first correction leaves \(-0.04297\), appreciably above the storage floor. Many distributed rotations can have a much smaller spectral norm at the same cap.
3. **Expected effect.** The compute-dominated estimate is \(2/7\) of warm arithmetic, about 0.70 ms/factor at 3072 or 13 ms at 8192 if products cost equally. Lower confidence at 768. Removing launches can still help there.
4. **Failure modes.** BF16 input Q is already nonorthogonal. A small normalized Frobenius Gram error can conceal a bad singular direction. One correction is least attractive when motion concentrates in a single plane.
5. **BF16 and \(\beta_C\).** The extra cast can itself perturb Q, while an extra correction can reduce prior storage error. Replay *chains* with the actual ten-age spacing; a single synthetic refresh cannot select the policy.
6. **Cheap test.** Record spectral Gram error in addition to the normalized Gram error, including the fourth consecutive warm event. Reject the one-correction policy for groups with accumulating spectral outliers, poorer ordered-prefix agreement, or no timing gain.

### A5. Reassociate moment transport on the short side

1. **Update.** For \(m>n\), form \(S_R=Q_R^\top Q_R^+\), then compute \((Q_L^+)^\top[Q_L M]S_R\). For \(n>m\), form \(S_L=(Q_L^+)^\top Q_L\) and apply the two right-side basis multiplications separately. Keep the existing association for square matrices unless benchmarking favors another. This uses associativity even when the bases are imperfectly orthogonal.
2. **Cost.** Four products in both versions. For \(m>n\), the original costs \(2(2m^2n+2mn^2)\) FLOPs, and the new order costs \(2(2m^2n+mn^2+n^3)\). The saving is \(2n^2(m-n)\). Forming *both* overlaps blindly can increase the cost because \(m^3+n^3-(m^2n+mn^2)=(m+n)(m-n)^2\). Try BF16 output for the overlap first, then FP32 output with a single cast. Both change the rounding path.
3. **Expected effect.** Modest. Even eliminating the measured 5.06 ms of transport altogether would save only about \(0.1(5.06)=0.51\) ms/call. The stronger opportunity occurs if the refresh supplies its own small coordinate rotation, as in B3 below.
4. **Failure modes.** The short-side GEMM can be more efficient than its replacement despite doing more FLOPs. Forming a nearly identity overlap in BF16 changes the moment error.
5. **BF16 and \(\beta_C\).** This preserves the real-arithmetic transpose transport, including its sensitivity to a nonorthogonal basis. With \(\beta_1=0.95\), a transport error survives ten calls with multiplier \(0.95^{10}=0.599\).
6. **Cheap test.** Microbenchmark the four production shapes and compare with FP32 transport of the *stored* bases on artificial moments. Reject by shape if slower or if the relative error grows materially. Replay lacks moments; loss needs a short run.

### A6. Reallocate a fixed QR budget toward the early trajectory

1. **Update.** As a concrete budget-neutral test, replace the four warm events at ages 10, 20, 30, 40 of every 50-age interval through age 500 by QR. This buys 40 extra early QRs. Convert 40 later scheduled QRs to warm, for example ages \(600,800,\ldots,8400\). Every ten-age event still refreshes. A speed-oriented second point uses the same later substitutions with only the first 250 ages on QR10.
2. **Cost.** The first policy exchanges 40 events each way and has the original total arithmetic by event class. The second buys 20 early QRs and removes 40 late ones. Using shared-window means, its estimated saving is \(20(601.18-61.61)/10000=1.08\) ms/call. Early and late timings need not be identical.
3. **Expected effect.** High confidence this tests a consequential timing question. Low confidence in the late loss outcome. It spends accurate refreshes where the observed loss gap opens fastest. This is a schedule alternative, not a cheap per-event QR replacement.
4. **Failure modes.** Removing a later QR creates a 100-age interval between repairs. Hard late factors may deteriorate, and the persistent late gap may require continued accuracy. A schedule tuned to a single seed may overfit.
5. **BF16 and \(\beta_C\).** After 50 ages only \(0.95^{50}=0.077\) of an old covariance contribution survives. A late basis cannot count on the covariance that originally determined it. Compare against matched BF16 QR10.
6. **Cheap test.** Replay the first 1,000 ages, including the first enlarged repair intervals. Reject the later substitutions if tracking degrades sharply in those intervals. The early-loss hypothesis needs a paired 500-call training run; the budget trade needs the full horizon.

## B. Tracking the QR10 trajectory

For a clean local target, take the same rounded \(C,Q\) at a snapshot, sort by the incumbent QR scores, and call the reordered orthogonal basis \(Q_p\). Write \(B=Q_p^\top C Q_p=D+E\), with \(E\) off-diagonal. Choose signs for positive \(R\) diagonal. To first order,
\[
D+E=(I+K)(D+T)+O(E^2)=D+KD+T+O(E^2),
\]
where \(K\) is skew and \(T\) upper triangular. Consequently
\[
\boxed{K_{ij}=E_{ij}/d_j\quad(i>j),\qquad K_{ji}=-K_{ij}.}
\]
This expansion needs positive, sufficiently resolved pivots and small relative perturbations. It follows because QR of \(CQ_p=Q_p B\) has orthogonal factor \(Q_p\,\mathrm{qr}(B).Q\), up to signs. The BF16 Q makes this a local model.

For two ordered directions with \(d_1>d_2\), the exact QR first-column angle is \(\arctan(B_{21}/d_1)\). The present warm angle is approximately \(+\tfrac12 B_{21}/(d_1-d_2)\) when damping and the cap are inactive. Depending on the spectrum it can **undershoot or overshoot** QR. At equal positive diagonals it returns zero, while the QR angle is still \(\arctan(B_{21}/d_1)\). Increasing the present step size alone cannot fix this discrepancy.

For all B candidates I would report a **cost gate** beside quality. At batch 12 the 2× warm ceilings are 0.138, 0.662, 2.04, and 4.92 ms/factor at 768, 1536, 2304, and 3072. I would also use \(0.5\times\) measured QR as a concrete interpretation of “well below QR” and report batch 1 separately. At batch 1 and 256, **the existing warm implementation itself misses that second gate** (0.79 versus 0.89 ms). FLOP counts cannot certify it there.

### B1. Use the ordered QR tangent instead of the inverse-gap tangent

1. **Update.** Compute \(B\) as now, sort its diagonal descending with the QR tie policy, and permute both axes and Q consistently. For the first controlled experiment use
   \[
   L_{ij}=\mathbf1_{i>j}\,B_{ij}/\max(d_j,\delta),\quad K=L-L^\top,\qquad
   \delta=0.01\max_j|d_j|
   \]
   with an absolute positive floor. Use the existing cap and retraction on \(Q_p(I+K)\). Permute \(V\) on both sides of the matrix. For an explicit cap ablation, estimate \(\|K\|_2\) with four power iterations on \(K^\top K\) from a fixed non-axis-aligned vector, and scale by \(\min(1,0.4/\max(\widehat{\|K\|_2},10^{-12}))\). Measure spectral Gram outliers for this ablation.
2. **Cost.** Same seven square products, BF16 inputs and FP32 accumulation as current warm. Add sort and quadratic gathers. The cap ablation adds eight matrix-vector products per factor, not eight GEMMs of cubic cost. All denominator, skew and reduction arithmetic should be FP32. The cheap diagonal score differs slightly from production QR's \(Q\odot(CQ)\) contraction owing to intermediate rounding; record permutation agreement.
3. **Expected effect.** My first choice for a full-basis approximation. It matches the local ordered QR generator before limiting and repairs the positive equal-diagonal stall. I expect better *same-input QR agreement* for resolved directions. The existing global cap may mask nearly all the gain, hence the explicit ablation. No defensible CE prediction.
4. **Failure modes.** Off-diagonal mass may be too large for first order, especially in trailing columns. The floor changes weak-direction dynamics. A power estimate can underestimate the norm. With tied scores, small BF16 perturbations can select a different QR ordering.
5. **BF16 and \(\beta_C\).** The pivot floor limits amplification of BF16 noise in a weak column. Ten calls replace about 40% of the covariance history; expanding around an almost diagonal basis may be a poor approximation in precisely the factors that need help. Stratify by pre-refresh error.
6. **Cheap test.** Branch both methods from identical replay states and compare signed-aligned column overlaps, prefix overlaps, permutation agreement, and Gram tails; then chain both. Record the cap multiplier and estimated spectral norm before and after clipping. Reject if it only improves \(r_C\), or if the new cap yields poor Gram tails or misses the timing gates.

### B2. Match the leading QR panel and give the tail a smooth orthogonal completion

1. **Update.** Compute \(Z=CQ\) and order as in QR. In the ordered basis form only \(X=Q_p^\top Z_{p,1:k}\), and take its tall Householder QR. Let \(H\) be the implicit full orthogonal product of its \(k\) reflectors. Return \(Q_pH\), with the first \(k\) signs aligned to the positive-diagonal convention. Use the stable reflector sign and apply its column-sign correction separately. For a diagonal positive input this gives the old basis up to signs, including its untouched tail. Start with \(k=d/8\). Permute \(V\).
2. **Cost.** One full \(CQ\) (BF16 inputs, FP32 output if desired), one \(d\times d\) by \(d\times k\) product, FP32 tall QR, and implicit right application of \(k\) reflectors to Q. The latter costs at most \(4d^2k\) FLOPs by a dot and update per row per reflector; the tall QR costs about \(2dk^2-2k^3/3\) by summing its shrinking trailing updates. No full square QR or explicit \(H\). For the \(Q_p^\top Z\) product choose FP32 or deliberately round \(Z\) for BF16 tensor cores and measure both. Batch the panel application.
3. **Expected effect.** With orthogonal Q and exact arithmetic the first \(k\) QR columns agree up to sign: their QR depends only on the first \(k\) input columns. This is the most interpretable quality/speed knob. It can also reduce large-\(d\) warm arithmetic substantially. Its narrow factorization can still be latency-bound at 768.
4. **Failure modes.** Leading variance directions need not dominate the Adam update. The trailing completion does not perform their ordered power iteration. Explicitly forming a full QR completion defeats the cost argument. If the stored Q is nonorthogonal then \(Q_p Q_p^\top Z\ne Z\); a right orthogonal update also preserves existing singular-value defects before storage rounding.
5. **BF16 and \(\beta_C\).** Stored-basis drift is a real risk across the four warm events. The leading set can turn over as the short EMA changes, so sort before selecting it. Track turnover near the \(k\)-boundary.
6. **Cheap test.** Sweep \(k/d=1/16,1/8,1/4\). Compare to the QR10 *prefix* at several ranks inside and outside the panel, and inspect normalized off-diagonal correlations in the tail. Reject panel sizes that miss either speed gate or improve the leading subspace while materially degrading the bulk.

### B3. Perform exact ordered QR rotations within small coordinate groups

1. **Update.** Compute \(Z=CQ\) and sort as for QR. Partition ordered coordinates into groups \(J\) of size \(b\). Compute \(B_J=Q_J^\top Z_J\), \(H_J=\mathrm{qr}(B_J).Q\) with consistent signs, and return \(Q_J H_J\). Start with \(b=16,32\). Compare contiguous groups with a fixed interleaving that places separated ranks in the same group. For \(b=2\), this is just a Givens rotation with cosine and sine proportional to \((B_{11},B_{21})\). It moves in your equal-diagonal example.
2. **Cost.** One full \(CQ\), plus \((d/b)\) products of shape \(b\times d\) by \(d\times b\) and \((d/b)\) products of shape \(d\times b\) by \(b\times b\). Together the small products cost \(4d^2b\) FLOPs. Small FP32 QR with explicit Q costs about \((8/3)db^2\) in leading arithmetic. Use BF16-input GEMMs with FP32 local \(B_J\), FP32 QR, and BF16-stored Q. With two-element groups, direct FP32 `hypot` and column rotations avoid the solver dispatch altogether. The known \(H_J\) also permit moment transport in coordinate blocks without forming world coordinates, exact for ideal orthogonal bases before rounding.
3. **Expected effect.** Most attractive at 4096–8192: only one cubic GEMM and no full retraction. It reproduces QR exactly for a block-diagonal ordered \(B\) with those groups and consistent signs. On dense factors, the fraction of omitted coupling determines quality.
4. **Failure modes.** Contiguous grouping can miss a strong leading-to-tail coupling. Interleaving can miss neighboring directions. Many tiny QR dispatches can erase all gains; a single batched small kernel is important. Nonorthogonality can accumulate as in B2.
5. **BF16 and \(\beta_C\).** FP32 local rotations avoid dividing by noisy eigenvalue gaps. They still respond to a BF16-rounded \(C\). Keep the full repair on its original cadence during the comparison, especially if group membership changes.
6. **Cheap test.** Measure \(\sum_J\|\mathrm{offdiag}(B_J)\|_F^2/\|\mathrm{offdiag}(B)\|_F^2\) offline to explain success or failure. Compare chained QR-prefix agreement, normalized correlations, spectral Gram error, and grouped-kernel time. Reject if captured coupling is small and QR agreement fails to improve even with interleaving.

### B4. Stabilize Cholesky-QR by shifting the power operator

1. **Update.** Form \(Z_\mu=(C+\mu I)Q_p=Z_p+\mu Q_p\), using the *original* sort scores. Compute \(R^\top R=Z_\mu^\top Z_\mu\), then solve \(Q^+R=Z_\mu\) by a right triangular solve. Sweep \(\mu/\|C\|_F\) over \(0,10^{-4},10^{-3},10^{-2}\); avoid an explicit inverse. Form the Gram with verified full FP32 multiplicand precision. Shifting \(C\) matters here. Adding a shift only to the Gram yields \(Q^{+\top}Q^+=I-\nu R^{-\top}R^{-1}\) in exact arithmetic and can leave visibly short columns.
2. **Cost.** One square FP32 \(CQ\), one symmetric FP32 Gram (about \(d^3\) FLOPs with triangular computation), one Cholesky (about \(d^3/3\), from summing trailing triangular updates), and one triangular solve (about \(d^3\), from \(d\) right-hand sides each costing about \(d^2\)). Add \(O(d^2)\) shift, norm, scores, gather, and BF16 cast. A full GEMM Gram costs \(2d^3\). Full-precision Gram time must be measured separately from the supplied “high” setting.
3. **Expected effect.** At zero shift, successful accurate Cholesky-QR gives the ordered QR target up to signs, since \(Z=Q^+R\) and \(Q^{+\top}Q^+=I\). For orthogonal Q and PSD \(C\), the shifted Gram condition number is \(((\lambda_{\max}+\mu)/(\lambda_{\min}+\mu))^2\). Shifting damps the local QR angle by changing its denominator from \(d_j\) to \(d_j+\mu\). Your existing timing is already outside the 2× gate at 768 (0.171 versus 0.138 ms), inside it at 1536–3072, and numerically disqualified at 6144–8192. **This is a conditional size-specific candidate.**
4. **Failure modes.** A small shift may leave an unusable Gram; a large one suppresses the desired rotation. The rounded \(C\) can be indefinite. A successful Cholesky status alone gives no orthogonality guarantee. Do not count a NaN run as a fast refresh.
5. **BF16 and \(\beta_C\).** Shifting cannot restore tail information lost in the BF16 covariance. A fixed absolute shift would change strength substantially over training; tie it to the current factor scale. Check Gram error both before and after basis storage.
6. **Cheap test.** Replay each scale choice with factorization status, worst singular-value defect, and same-input QR agreement. Reject a size if no shift passes all three of numerical reliability, trajectory quality, and timing. I would not put this on the training critical path before that sweep.

### B5. Transport diagonal second-moment history with the known rotations

1. **Update.** Always apply the same permutations to both coordinate axes of \(V\) that the refresh uses for the bases. For an actual rotation, let \(T_L=Q_L^{+\top}Q_L,\ T_R=Q_R^\top Q_R^+\). Then a gradient becomes \(g'=T_L g T_R\) for ideal orthogonal bases. Under a diagonal **raw second-moment** approximation in the old coordinates,
   \[
   V'=(T_L\odot T_L)V(T_R\odot T_R).
   \]
   This follows by expanding \(g_{ij}'{}^2\) and dropping cross terms between distinct old coordinates. Test first with B3's known block rotations after permutation. For a pair rotated by \(\theta\), each updated diagonal variance is a mixture with weights \(\cos^2\theta,\sin^2\theta\).
2. **Cost.** With known rotations of block size \(b\) on both sides, two collections of block applications cost \(2mnb+2mnb=4mnb\) FLOPs, plus squaring the coefficients, permutation, and one BF16 storage cast. Use FP32 coefficients and weighted sums for small blocks. For dense *known* rotations, the two products cost \(2P\); inferring both dense overlaps from the world bases costs another \(2(m^3+n^3)\). I would avoid that latter version on the larger factors.
3. **Expected effect.** Zero effect on open-loop \(r_C\). Potential loss improvement if large rotations leave diagonal variances attached to the wrong directions. Pure permutations are exact for diagonal history. Keep a permutation-only arm: QR10 itself uses that approximation, so extra mixing could move the optimizer away from the desired trajectory.
4. **Failure modes.** Correlated coordinates have missing cross terms, and repeated mixing can flatten genuinely anisotropic \(V\). The product of squared orthogonal entries conserves total variance but discards correlations, so conservation alone is a weak test. Explicit coordinate rotations differ slightly from overlaps of stored Q's.
5. **BF16 and \(\beta_C\).** Round the mixed moment once. Here \(\beta_2=\beta_C=0.95\); about 60% of the previous refresh's history remains ten ages later. Consequently stale denominators and over-mixed denominators can both matter through successive refreshes.
6. **Cheap test.** The microbenchmark can price the block kernels; neither supplied harness can evaluate moment fidelity. The smallest informative extension is a short saved per-step gradient stream for several actual matrices, replaying \(M,V\) and comparing the world update to BF16 QR10. A paired 500-call training run is needed for CE. Reject if mixing increases direction error or gives no reproducible loss gain over permutation alone.

## What the quality measurements establish

**Use two QR comparisons.** First, branch a candidate and production QR from the *same* \(C,Q\) at every saved snapshot. That measures whether the proposed refresh implements the target map. Second, chain both schedules from age zero. That measures accumulated trajectory error. For each, compare sorted leading-prefix projectors and sign-aligned ordered columns. For orthonormal \(U,V\) with \(k\) columns,
\[
\|UU^\top-VV^\top\|_F^2=2k-2\|U^\top V\|_F^2
\]
by expanding the square and cycling the trace. Orthonormalize diagnostic prefixes in FP32 or account for their Gram defects. The full-square subspace overlap is vacuous. Compare several proper prefixes, and distinguish a near-tied eigenvalue cluster's subspace from the labeling of columns within it.

**Keep \(r_C\) as one diagnostic.** It is invariant to column permutations and sign flips, strongly weights high-energy directions, and cannot see stale \(V\). An exact eigenbasis can beat QR10 on \(r_C\) while giving a different next Adam update. The normalized correlation and floored inverse-variance metrics already in your harness help with tail directions. Report distributions by shape, side, age, and time since QR. A mean of 0.38 and a maximum of 0.975 warrant finding out *which* factors produce the tail. In a gradient replay I would prioritize the cosine **and norm ratio** of the world update relative to BF16 QR10, immediately after a refresh and over the next ten calls. Cosine alone misses a denominator-induced step-size error.

**Treat call 500 as a discriminator.** The large early gap survives in the FP32-state warm pilot. That makes BF16 storage an insufficient explanation for the early loss difference. The short-control fractions are \((5.018046-5.001514)/(5.018046-4.877601)=11.8\%\) at 500 and \((3.998679-3.993939)/(3.998679-3.985905)=37.1\%\) at 2000. Both denominators use the unmatched FP32 reference. BF16 QR20 also has better CE than warm at those two evaluations, so the evidence favors testing what a higher-quality refresh buys, as well as its frequency. I would evaluate CE at equal calls **and** equal training wall time; the two comparisons answer different questions. The persistent late gap makes a call-500 success insufficient for selecting a final method.

## Other pitfalls that affect interpretation

- **The Frobenius cap is a dimension-dependent motion budget.** If there are \(p\) independent planes with equal skew coefficient \(x\), then \(\|K\|_F/\sqrt2=\sqrt p\,|x|\), so after clipping each plane gets at most \(0.5/\sqrt p\). The spectral norm in this example is only \(|x|\). A large factor can therefore be severely under-rotated even when every individual rotation would be safe. The shared damping scale further suppresses gaps in a long weak tail. Log the *applied* motion, not just the proposed skew.
- **Refresh age and update age differ.** At age 10 the parameter uses the *old* basis; the first parameter update using the new one is at age 11. A replay's post-refresh number at age 10 describes the basis available for age 11. Moving refresh ahead of the update changes how the current gradient influences its own basis and requires transporting history before the EMA. It is a different experiment.
- **BF16 casts alter more than orthogonality.** Near a well-scaled entry, BF16 unit roundoff is about \(2^{-8}\). The covariance increment is \(0.05(P_{ij}-C_{ij})\), so an entry can stick when this is below roughly half a local storage ulp, with the exact threshold depending on the mantissa. The same issue affects moments. Symmetry survives a symmetric rounding path; PSD need not. Also, the QR ordering score \(q^\top Cq\) is a literal Rayleigh quotient only for unit-norm \(q\); BF16 rounding perturbs norms and can break close ties. QR's “FP32 island” also includes a matmul that may use TF32 under your settings.
- **The optimizer equations should match the reference literally.** Your decay multiplies the adaptive update by \(1-\eta\lambda\) as well as shrinking the old weight. Compared with \(W(1-\eta\lambda)-\eta c_tU\), the difference is \(+\eta^2\lambda c_tU\). Likewise, applying \(c_t\) outside a denominator containing \(\epsilon\) corresponds to a bias-corrected denominator with \(\epsilon/\sqrt{1-\beta_2^t}\). These are small but checkable differences. Check the skipped first update, age used for correction, and clipping against the reference before attributing a residual CE gap.
- **Timing attribution needs matched arms.** Isolated warm plus transport is 54.90 ms versus a 51.47 ms event increment; isolated QR plus transport and permutation is 599.18 versus 591.04 ms. Attribute gains using whole compiled regions and report primitive times separately. A rough BF16 QR10 counterfactual is \(10.14+0.1(591.04)=69.24\) ms/call, but it is not a measured matched control. Synchronizing every host event, periodic evaluation or logging at multiples of 10 or 50, clock changes, and cross-launch differences can bias event classes. Include an uninterrupted whole-window timing with synchronization only at its boundaries, and report tails plus successful-factor counts.

## Ranking

| Rank | A — speed | B — QR tracking and loss |
|---|---|---|
| 1 | **A3, batched blocked QR.** The measured repair increment costs 11.82 ms per average call. | **B1, ordered QR tangent.** Tests the actual QR generator with nearly the existing warm machinery. |
| 2 | **A1, ordinary data movement.** Savings recur on every call; inspect materialized copies first. | **B2, leading QR panel.** Gives an exact-arithmetic guarantee on a tunable part of the target. |
| 3 | **A4, one polar correction.** Easy to falsify in replay and removes two cubic products. | **B3, grouped QR.** One cubic product and a direct way to test how much coupling must be retained. |
| 4 | **A2, symmetric products.** Applies to both ordinary and warm work; most promising toward large \(d\). | **B5, second-moment transport.** Directly tests a loss mechanism invisible to every covariance metric. |
| 5 | **A6, time-dependent QR budget.** Can trade later repairs for early accuracy with a specified total cost. | **B4, shifted Cholesky-QR.** Close to the target when stable, but timing and conditioning restrict its usable range. |

**Top three pitfalls**

1. Optimizing a gap-based diagonalization step under a global motion cap when the desired QR map uses ordered pivots and can move strongly at zero gap.
2. Selecting by average \(r_C\) while missing column identity, tail directions, and the denominator history that actually scales the update.
3. Attributing the loss or speed difference to warm refreshes before obtaining the matched compiled BF16 QR10 arm.

**Five measurements I would want next**

1. The matched BF16 QR10 CE and timing, paired through call 500 and 2000, followed through 10k for the selected comparison.
2. Same-input QR agreement and chained QR10 agreement by factor and age, including ordered prefixes, column turnover, and tail correlations.
3. Warm cap multiplier, spectral norm of applied skew, pivot/gap distribution, and spectral Gram error after zero, one, and two corrections.
4. A short per-step gradient replay with \(M,V\), reporting world-update cosine and norm ratio through the ten calls after refresh.
5. A whole-event device/host timing breakdown covering packing, covariance, \(CQ\), QR factorization, explicit Q, warm GEMMs, and moment transport, reconciled against uninterrupted whole-step timing.

---

# Document 5: Reviewer Z's stage-1 answer (verbatim, anonymized)

## Where the Time Actually Goes (§2 Diagnosis)

Before proposing changes, we decompose the 109M timings and synthetic scaling curves against the hardware ceilings of an RTX 5090 (Blackwell GB202: ~209.5 TFLOPS dense BF16 with FP32 accumulate due to the 2× GeForce FP32-accumulator throttle, ~104.8 TFLOPS FP32/TF32, 96 MB L2, ~1.79 TB/s GDDR7 **[verify][recalled]**).

1. **Factor cubic skew ($\sum n^3$).** Across the 96 factors ($60\times 768,\; 12\times 1536,\; 12\times 2304,\; 12\times 3072$):
   $$\sum_{f=1}^{96} n_f^3 = 60(768^3) + 12(1536^3) + 12(2304^3) + 12(3072^3) = (27.18 + 43.49 + 146.77 + 347.89)\times 10^9 = 565.33\times 10^9.$$
   The 24 left factors of fused QKV ($2304$) and fused SwiGLU-in ($3072$) carry **87.5% of all $O(n^3)$ FLOPs**, while the 60 factors at $n=768$ carry **4.8% of FLOPs** across 5 separate batch-12 calls.

2. **Amortized mean optimizer budget ($26.08\text{ ms/call}$):**
   $$26.08\text{ ms} = \underbrace{10.14\text{ ms}}_{\text{ordinary body (100\%): } 38.9\%} + \underbrace{0.08 \times 51.47\text{ ms}}_{\text{warm increment ($4.12\text{ ms}$): } 15.8\%} + \underbrace{0.02 \times 591.04\text{ ms}}_{\text{QR increment ($11.82\text{ ms}$): } 45.3\%}.$$

   - **QR repair ($593.74\text{ ms}$ for 96 factors) runs at 4.44 TFLOPS (~2% of GPU peak).** `qr_basis` performs $Z = CQ$ ($2n^3$) plus Householder `geqrf` + `orgqr` ($\frac{4}{3}n^3 + \frac{4}{3}n^3 = \frac{8}{3}n^3$ **[verify][certain]**), totaling $\frac{14}{3}\sum n^3 = 2.638\text{ TFLOPs}$. Dividing $2.638\text{ TFLOPs} / 0.59374\text{ s} = 4.44\text{ TFLOPS}$ ($0.83\text{ TFLOPS}$ at $n=768$, $7.47\text{ TFLOPS}$ at $n=3072$). Because PyTorch's batched `torch.linalg.qr` loops sequentially over the batch on one CUDA stream **[verify][certain]**, 96 low-occupancy panel factorizations serialize back-to-back. Notice that the five $n=768$ groups alone take $5 \times 30.8 = 154.0\text{ ms}$ (26% of QR time for 4.8% of FLOPs).
   - **Ordinary step ($10.14\text{ ms}$) is ~70% Tensor Core GEMMs and ~30% pack/state bandwidth.** Each matrix group ($b=12$) executes 6 GEMMs ($\bar{G}$: $2m^2n + 2mn^2$; $U$: $2m^2n + 2mn^2$; $C_L, C_R$: $2m^2n + 2mn^2$), totaling $72mn(m+n)$ per group:
     $$(0.391_{2304\times 768} + 0.065_{768\times 768} + 0.652_{3072\times 768} + 0.196_{768\times 1536}) = 1.305\text{ TFLOPs}.$$
     At 209.5 TFLOPS peak, ideal GEMM time is $6.23\text{ ms}$ (~$7.2\text{ ms}$ achieved at $k=768$). One-third of this ($0.435\text{ TFLOPs}$ + $962\text{ MB}$ R/W) is `covariance_ema`, computing both triangles of symmetric matrices on every step. Another ~$1.5\text{–}2.2\text{ ms}$ goes to `torch.stack(params)`, `torch.stack(grads)`, and 48 `copy_` unpacks (~$1.4\text{ GB}$ redundant traffic) plus moment updates, and ~$0.7\text{ ms}$ to auxiliary AdamW.
   - **Warm refresh ($49.84\text{ ms}$) has two distinct regimes.** Summing §2.2's `warm, b12` across the 109M counts reproduces §2.1 identically: $60(0.069) + 12(0.331) + 12(1.02) + 12(2.46) = 4.14 + 3.97 + 12.24 + 29.52 = 49.87\text{ ms}$. `warm_basis` runs 7 GEMMs ($14n^3$ FLOPs):
     - *Latency-bound ($n \le 1024$):* Both `b1` and `b12` take a flat **$0.80\text{–}0.84\text{ ms/call}$** (~15 sequential cuBLAS + Triton launches at ~$53\text{ }\mu\text{s}$ each, amplified by deterministic workspace resets).
     - *Compute + FP32-buffer bandwidth-bound ($n \ge 2304$):* At $n=6144, 8192$, `b1` achieves $3247\text{ GFLOPs}/15.2\text{ ms} = 213.6\text{ TFLOPS}$ and $7697\text{ GFLOPs}/37.2\text{ ms} = 206.9\text{ TFLOPS}$—100% of the RTX 5090's BF16-FP32acc ceiling. Yet `b12` regresses by **22–23%** ($37.2 \to 45.5\text{ ms}$ at $n=8192$) because 5 unfused `mm32` calls materialize $12\times 8192\times 8192$ FP32 buffers ($3.22\text{ GB}$ each) to DRAM instead of fusing BF16 epilogues, blowing out the 96 MB L2 cache.

---

## Question A: Making Refreshes and the Method Faster

### Part (A-eng): Engineering at Strictly Fixed Mathematics

#### Candidate A1 — Fused `baddbmm` Epilogues, Cross-Group Factor Coalescing, and Contiguous Param/Grad Banks

1. **Exact update:**
   - **In `warm_basis` (BF16 production path):** Pre-cache `neg_eye = -torch.eye(n, dtype=torch.bfloat16, device=dev).expand(b, n, n)`. Because `0.0` and `-1.0` are exact in BF16 and cuBLAS `baddbmm` on BF16 uses `CUBLAS_COMPUTE_32F` internally before rounding once to BF16 **[verify][certain]**, replace 4 of the 5 `mm32` + Triton-cast chains with single fused `baddbmm` calls:
     ```python
     # Exact replacement for y0 and the 2 Newton-Schulz iterations (basis_dtype == compute_dtype == bf16):
     y = torch.baddbmm(q, q, a.to(torch.bfloat16), beta=1.0, alpha=1.0)
     for _ in range(2):
         res = torch.baddbmm(neg_eye, y.transpose(-1, -2), y, beta=1.0, alpha=1.0)
         y = torch.baddbmm(y, y, res, beta=1.0, alpha=-0.5)
     ```
   - **In `qr_basis`:** Replace `c, old = cov.float(), q.float(); z = torch.bmm(c, old)` (TF32 GEMM) with `z = mm32(cov, q)` (BF16 Tensor Core GEMM with FP32 output). Since BF16's 7 mantissa bits embed losslessly into TF32's 10 mantissa bits **[verify][certain]**, the multiplicands are bit-identical while running at 2× Tensor Core rate.
   - **Cross-group factor coalescing:** Compile one optimizer-wide step across all 4 groups and stack the five $n=768$ factor batches ($5\times 12$) into one $b=60$ tensor for `warm_basis`.
   - **Contiguous parameter/grad banks:** Allocate one `(12, m, n)` FP32 master bank `P_bank` per shape group and bind each module parameter as `nn.Parameter(P_bank[i])` (with a `post_accumulate_grad_hook` writing into a contiguous `(12, m, n)` BF16 `G_bank`), eliminating `torch.stack(params)`, `torch.stack(grads)`, and `unbind`/`copy_`.

2. **Cost:**
   - Same GEMM count ($7$ BF16 $n\times n$ in `warm_basis`, $6$ BF16 $m\times n$ in `matrix_step`), except `qr_basis`'s $Z=CQ$ shifts from 1 FP32/TF32 GEMM to 1 BF16-in/FP32-out GEMM.
   - Kernel count in `warm_basis` drops from ~15 to **9** (7 cuBLAS GEMMs + 2 Triton kernels for `sym(b)`/skew `a`/norm).
   - DRAM savings in `warm_basis`: each of the 4 replaced `mm32` chains previously wrote $4n^2\text{ B}$ (FP32) and read $4n^2\text{ B} + 2n^2\text{ B}$ to write $2n^2\text{ B}$ ($12n^2\text{ B}$), vs $4n^2\text{ B}$ for `baddbmm`—saving **$32n^2\text{ bytes/factor}$** ($2.15\text{ GB/factor}$ or $25.8\text{ GB}$ at $b=12, n=8192$).
   - Contiguous banks save $2\times 283\text{ MB}$ (`stack`/`copy_` of `P`) + $425\text{ MB}$ (`stack`+cast of `G`) $\approx \mathbf{0.99\text{ GB}}$ every call.

3. **Expected effect:** *High confidence.*
   - Ordinary step: saves **$0.8\text{–}1.1\text{ ms/call}$** ($10.14 \to \sim 9.1\text{ ms}$).
   - Warm refresh at 109M: coalescing $5\times(b=12, n=768) \to (b=60, n=768)$ cuts $4.14\text{ ms} \to \sim 1.7\text{ ms}$, and fused `baddbmm` saves ~$2.5\text{ ms}$ across $n=1536\text{–}3072$ (total warm event: **$-5\text{ ms}$**).
   - Warm refresh at $n=6144\text{–}8192$ (`b12`): recovers most of the L2/DRAM regression vs `b1`, saving **$2.5\text{–}7.0\text{ ms/factor}$** at $n=6144\text{–}8192$.
   - `mm32` in `qr_basis`: saves ~$4\text{ ms}$ per 109M QR event and ~$3.5\text{ ms/factor}$ at $n=8192$.

4. **Failure modes:** `neg_eye.expand(b, n, n)` has batch stride `0`; in some cuBLAS versions `cublasGemmStridedBatchedEx` requires distinct output and `C` buffers when `strideC == 0`—allocate a contiguous `(b, n, n)` `neg_eye` if `out` aliases `C`, or let PyTorch allocate a fresh output tensor (default in `torch.baddbmm`). Weight-tying on embeddings is unaffected since embeddings are in auxiliary AdamW.

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** None—arithmetic is mathematically identical (`CUBLAS_COMPUTE_32F` accumulation rounded once to BF16).

6. **Cheap test:** §2.6 (ii) Microbenchmark: verify bitwise (or $\le 1\text{ ULP}$) equivalence of `warm_basis` outputs and check that `warm, b12` at $n=8192$ drops below $40\text{ ms/factor}$ and $b=60, n=768$ takes $<2.0\text{ ms}$ total ($<0.033\text{ ms/factor}$). Reject if `b12` at $n=8192$ saves $<5\%$.

---

#### Candidate A2 — Batched Triton Symmetric-Tile (`syrk`) Kernels for `covariance_ema` and Newton–Schulz Grams

1. **Exact update:** Both $G_b G_b^T$ / $G_b^T G_b$ (every call) and $Y_j^T Y_j - I$ (twice per warm refresh) are symmetric Gram matrices, yet `bmm`/`baddbmm` compute all $n^2$ tiles. Replace both with a batched Triton kernel grid `(cdiv(n, B) * (cdiv(n, B) + 1) // 2, batch)` that computes only upper-triangular blocks `pid_m <= pid_n` ($B=128$) in FP32 Tensor Core `tl.dot`, applies `beta * old + (1 - beta) * acc` (or `acc - (row == col)`), casts to BF16, and writes both `(pid_m, pid_n)` and transposed `(pid_n, pid_m)` (similar to symmetric Gram kernels used in Muon **[verify][recalled]**).

2. **Cost:**
   - Tile fraction computed is $\frac{N_t(N_t + 1)}{2 N_t^2} = \frac{1}{2}\left(1 + \frac{B}{n}\right)$ where $N_t = \lceil n / 128 \rceil$:
     - $n=768$ ($N_t=6$): $21/36 = 58.3\%$ of GEMM FLOPs;
     - $n=2304$ ($N_t=18$): $171/324 = 52.8\%$;
     - $n=3072$ ($N_t=24$): $300/576 = 52.1\%$;
     - $n=8192$ ($N_t=64$): $2080/4096 = 50.8\%$.
   - `covariance_ema` FLOPs across the 109M model drop from $0.435\text{ TFLOPs}$ to **$0.233\text{ TFLOPs}$** ($-46.4\%$), and `old` BF16 reads drop by $46.4\%$ ($241\text{ MB}$ saved/call).
   - `warm_basis` effective GEMM equivalents drop from $7$ to **$6.04\text{–}6.17$** ($-13.7\%$ FLOPs at $n=3072$, $-14.1\%$ at $n=8192$).

3. **Expected effect:** *Medium-high confidence* (assuming well-tuned Triton warp-specialized persistent tiles on Blackwell):
   - Ordinary step: saves **$0.9\text{–}1.2\text{ ms/call}$** ($10\text{–}12\%$ of every ordinary call).
   - Warm refresh: saves **$3.5\text{–}5.5\text{ ms}$** at 109M and **$5\text{–}6\text{ ms/factor}$** at $n=8192$.

4. **Failure modes:** Naive Triton GEMM configs on consumer Blackwell (`sm_120`) without tuned `num_stages` / TMA descriptors can achieve 75–85% of cuBLAS TFLOPS, eroding half of the 2× tile advantage; autotune over `BLOCK_M/N in {64, 128}`, `BLOCK_K in {64, 128}`.

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** Strictly *improves* invariants—guarantees exact bitwise symmetry $C = C^T$ in BF16 (whereas `baddbmm` can produce 1-ULP asymmetry between $(i,j)$ and $(j,i)$ when split-K reduction orders differ).

6. **Cheap test:** §2.6 (ii) Microbenchmark on `covariance_ema` and `warm_basis` across $n \in [768, 8192]$. Reject if the Triton `syrk` kernel is $<1.35\times$ faster than `torch.baddbmm` at $n \ge 1536$.

---

#### Candidate A3 — Multi-Stream Factor-Parallel `QRRepair` + Asynchronous Post-`W`-Write Overlap

1. **Exact update:**
   - **Multi-stream QR (within `QRRepair`):** Instead of calling `torch.linalg.qr` on $(12, n, n)$ on one stream, pre-create $S = 8$ CUDA streams (each with pre-allocated static workspace so the caching allocator never crosses streams) and assign the 96 individual $(n, n)$ `qr` calls by Longest-Processing-Time-First (LPT): each stream gets 1–2 factors of $n=3072$, 1–2 of $n=2304$, 1–2 of $n=1536$, and 7–8 of $n=768$, bracketed by `stream.wait_event(start_ev)` and `end_ev[s].record(stream)`.
   - **Post-`W`-write pipeline overlap (across the training step):** In §1.2, `W` is completely updated at Step 3 using pre-refresh $Q_L, Q_R$, and Steps 5–6 (`WarmRefresh` or `QRRepair`, `V` permute, `transport_moment`) touch **only** optimizer state (`Q, V, M`), which is untouched during Call $k+1$'s $283.4\text{ ms}$ forward+backward pass. Split the compiled step into `step_and_cov_graph` (Steps 1–4, returning immediately to start Call $k+1$'s forward pass) and `refresh_and_transport_graph` (Steps 5–6, launched on low-priority background stream(s) and synchronized via CUDA event right before Call $k+1$'s optimizer call).

2. **Cost:** Identical arithmetic and total FLOPs ($2.638\text{ TFLOPs}$ for 96 QRs; $7.915\text{ TFLOPs}$ for 96 warms), zero extra memory beyond static workspaces (~$250\text{ MB}$ across 8 streams at 109M).

3. **Expected effect:** *High confidence.*
   - **Multi-stream QR alone:** Single-factor QR at $n=3072$ uses $7.5\text{ TFLOPS}$ (~7% of FP32 peak); running 8 streams raises aggregate GPU utilization from $4.4\text{ TFLOPS}$ to ~$15\text{–}22\text{ TFLOPS}$, shrinking 109M QR wall time from **$593.7\text{ ms} \to \sim 130\text{–}190\text{ ms}$** (critical path bounded by $2\times 18.1 + 2\times 11.8 \approx 60\text{ ms}$ single-factor latency plus SM contention).
   - **With Fwd+Bwd overlap:** Hides ~$40\text{ ms}$ of the $51.5\text{ ms}$ warm event and ~$150\text{–}220\text{ ms}$ of QR behind Call $k+1$'s $283.4\text{ ms}$ Fwd+Bwd, dropping **mean exposed optimizer wall time from $26.08\text{ ms/call} \to \sim 11.5\text{–}13.5\text{ ms/call}$**.

4. **Failure modes:** In PyTorch's ATen cuSOLVER wrapper, `DeviceThreadHandlePool` shares one `cusolverDnHandle_t` per CPU thread per GPU, calling `cusolverDnSetStream` before each op **[verify][recalled]**; if `geqrf` allocates temporary workspace via the default allocator across streams without static workspace pools, allocator syncs can serialize streams. At $n=8192$, 8 concurrent FP32 QRs need $8 \times 268\text{ MB} \approx 2.15\text{ GB}$ scratch. Also, overlapping Tensor-Core-heavy `WarmRefresh` with Fwd+Bwd stretches Fwd+Bwd by its Tensor Core time (~$35\text{–}40\text{ ms}$), saving mainly launch/memory gaps (~$10\text{–}15\text{ ms}$), whereas low-occupancy `QRRepair` overlaps heavily with Fwd+Bwd GEMMs.

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** None—bitwise identical even under `torch.use_deterministic_algorithms(True)`, because Fwd+Bwd and background refresh operate on strictly disjoint tensors with deterministic per-stream kernels.

6. **Cheap test:**
   - Multi-stream QR: directly testable in §2.6 (ii) Microbenchmark over the 96 factors of 109M. Reject if 8-stream wall time exceeds $280\text{ ms}$ (<2× speedup).
   - Post-`W`-write overlap: cannot be tested in §2.6 (ii) alone (no Fwd+Bwd); test with a 200-call training timing run verifying bitwise-identical loss to the synchronous baseline and whole-step mean $<297\text{ ms}$.

---

### Part (A-eq): Equation-Level Speedups Preserving Quality

#### Candidate A4 — Regularized Column-Normalized Shifted Cholesky-QR2/3 (`sCholeskyQR`) Replacing Householder QR

1. **Exact update:** Why did §2.2's naive Cholesky-QR fail at $n=6144$? Because $\lambda_i \propto 1/i \implies \kappa(Z^T Z) \approx n^2$; right at $n = 4096 = 2^{12}$, $u_{\text{FP32}} \kappa(Z^T Z) = 2^{-24} \cdot (2^{12})^2 = 1.0$, and at $n = 6144$, $2^{-24} \cdot 6144^2 = 2.25 > 1$, making $Z^T Z$ numerically indefinite.
   We cure both rank-deficiency and conditioning using three exact-eigenvector-preserving steps based on CholeskyQR2 / Shifted CholeskyQR3 **[verify][recalled]**:
   - Compute $Z = \mathcal{F}(C, Q)$ via `mm32(cov, q)`; Rayleigh quotients $r_j = \sum_i \text{float}(Q_{ij}) Z_{ij}$; sort $\pi = \operatorname{argsort}_{\text{desc}}(r)$; scale $s = \max(\max_j |r_j|, 10^{-30})$.
   - **Isotropic shift before QR:** $Z_\delta = Z + (\delta s)\,\text{float}(Q)$ with $\delta = 2\times 10^{-4}$ (for 2-pass) or $\delta = 10^{-5}$ (for 3-pass). Note $Z_\delta = (C + \delta s I)Q$: since $C$ and $C + \delta s I$ have **identical eigenvectors in identical order**, the target eigenbasis is unchanged, while $\kappa(Z_\delta) \le \kappa(Q)(1 + \delta^{-1})$ is strictly bounded even if $\operatorname{rank}(C) = 1$.
   - **Column normalization:** $Z_0 = Z_{\delta,\pi} D_0^{-1}$ where $(D_0)_{jj} = \|z_{\delta,\pi(j)}\|_2$. Since $D_0 \succ 0$ is diagonal (hence upper-triangular with positive diagonal), by QR uniqueness $\operatorname{qr}(Z_0).Q \equiv \operatorname{qr}(Z_{\delta,\pi}).Q$ **[verify][certain]**, and now $\operatorname{diag}(Z_0^T Z_0) = I$, $\|Z_0\|_F^2 = n$.
   - **2-pass (or 3-pass) Cholesky-QR in IEEE FP32** (disable TF32 on $Z_k^T Z_k$ so precision is $2^{-24}$, not $2^{-10}$):
     ```python
     # Pass 1 (shifted so chol never fails):
     g1 = ieee_fp32_bmm(z0.transpose(-1, -2), z0) + (11 * n * 6e-8) * eye
     r1, _ = torch.linalg.cholesky_ex(g1, check_errors=False)
     z1 = torch.linalg.solve_triangular(r1, z0, upper=False, left=False).transpose(...) # Z0 @ R1^{-T} if lower L1: Z1 = Z0 @ L1^{-T}
     # More cleanly with upper=True:
     u1, _ = torch.linalg.cholesky_ex(g1, upper=True, check_errors=False)
     z1 = torch.linalg.solve_triangular(u1, z0, upper=True, left=False)
     # Pass 2 (unshifted polish to O(u_FP32) << 0.0024 BF16 floor):
     g2 = ieee_fp32_bmm(z1.transpose(-1, -2), z1) + 1e-7 * eye
     u2, _ = torch.linalg.cholesky_ex(g2, upper=True, check_errors=False)
     q_new = torch.linalg.solve_triangular(u2, z1, upper=True, left=False).to(basis_dtype)
     ```
     *(Note: `solve_triangular(u, z, upper=True, left=False)` solves $X U = Z \implies X = Z U^{-1}$ with zero host sync via `cholesky_ex` **[verify][certain]**).*

2. **Cost:** Per factor (2-pass):
   - 1 BF16-in/FP32-out $n\times n$ GEMM ($Z = CQ$: $2n^3$ FLOPs);
   - 2 IEEE-FP32 $n\times n$ Gram GEMMs ($4n^3$ FLOPs, or $2n^3$ with `syrk`);
   - 2 FP32 batched Cholesky `potrf` ($2 \times \frac{1}{3}n^3 = \frac{2}{3}n^3$ FLOPs **[verify][certain]**);
   - 2 FP32 `trsm` solves ($2 \times n^3 = 2n^3$ FLOPs **[verify][certain]**).
   Total: **$8.67n^3$ FLOPs** (6.67$n^3$ in fast IEEE FP32 BLAS-3/Cholesky + 2$n^3$ BF16 TC), zero Householder panel loops.

3. **Expected effect:** *High confidence.* Scaling from §2.2's 1-pass `Cholesky-QR` ($72.0\text{ ms}$ across 96 factors):
   - 109M QR event drops from **$593.7\text{ ms} \to \sim 110\text{–}130\text{ ms}$** (**4.6–5.4× faster**, cutting whole-optimizer mean by **~$9.4\text{ ms/call}$ (36%)**!).
   - Per factor (`b12`): $n=768$: $2.56 \to \sim 0.28\text{ ms}$ (9×); $n=3072$: $18.1 \to \sim 4.8\text{ ms}$ (3.8×); $n=8192$: $112 \to \sim 68\text{ ms}$ (1.65×, **0 NaNs**).
   - Orthogonality after BF16 storage matches Householder QR ($0.0024$), and $r_C$ matches within $<0.002$.

4. **Failure modes:**
   - If TF32 is accidentally left enabled on `g1`/`g2`, input truncation at $2^{-10}$ limits conditioning to $\kappa(Z_0) \lesssim 30$; must enforce IEEE FP32 on those two GEMMs (e.g., a Triton IEEE-FP32 `tl.dot(..., input_precision="ieee")` `syrk` kernel inside `torch.compile`).
   - Within tail eigenspaces where $\lambda_i, \lambda_j \ll \delta s = 2\times 10^{-4} s$, power-step separation slows to $\frac{\lambda_i + \delta s}{\lambda_j + \delta s} \approx 1$; adding Pass 3 (`sCholeskyQR3`) lets $\delta$ drop to $10^{-6}$ at +45% pass cost (~$165\text{ ms}$ total, still 3.6× faster than QR).

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** Pass 2 reaches FP32 orthogonality ~$10^{-6}$, well below the $0.0024$ BF16 storage floor. With $\beta_C = 0.95$, $C$ can still have decayed tail modes ($0.95^{50} \approx 0.077$ per 50 idle steps on inactive features), which $\delta s$ stabilizes cleanly.

6. **Cheap test:** Test in **both** §2.6 (i) Open-loop replay (all widths 256–2048) and §2.6 (ii) Microbenchmark ($n = 256\text{–}8192$, checking $n=6144, 8192$ specifically). Reject if any NaN occurs, if normalized Gram error exceeds $0.0026$, or if replay $\rho_{\text{rms}}$ / top-$k$ overlap deviates from `QR10` by $>1\%$.

---

#### Candidate A5 — Single Newton–Schulz Correction in `WarmRefresh` (5 GEMMs vs 7)

1. **Exact update:** Change `for _ in range(2):` in `warm_basis` to `for _ in range(1):`.

2. **Cost & algebraic error bound:**
   - Drops `warm_basis` from **7 to 5** $n\times n$ BF16 GEMMs ($14n^3 \to 10n^3$, **$-28.6\%$ FLOPs**; 1 BF16-out + 4 FP32-out or `baddbmm`).
   - **Exact derivation:** For $Q^T Q = I + E_{\text{in}}$ ($\|E_{\text{in}}\|_F/\sqrt{n} \approx 0.0024$) and skew $\tilde{A} = aA$ ($\tilde{A}^T = -\tilde{A}$), first-order skew terms cancel in $Y_0 = Q(I + \tilde{A})$:
     $$E_0 = Y_0^T Y_0 - I = (I - \tilde{A})(I + E_{\text{in}})(I + \tilde{A}) - I = -\tilde{A}^2 + (I - \tilde{A})E_{\text{in}}(I + \tilde{A}).$$
     One cubic Björck–Bowie step $Y_1 = Y_0(I - \frac{1}{2}E_0)$ maps each eigenvalue $e_0$ of $E_0$ to **[verify][certain]**:
     $$e_1 = (1 + e_0)\left(1 - \frac{1}{2}e_0\right)^2 - 1 = -\frac{3}{4}e_0^2 + \frac{1}{4}e_0^3.$$
     Under the cap $\|\tilde{A}\|_F/\sqrt{2} \le 0.5$, the **worst-case** concentration puts all norm in a single 2D plane ($\sigma_1 = \sigma_2 = 0.5 \implies e_0 = 0.25$), giving $|e_1| = \frac{3}{4}(0.25)^2 - \frac{1}{4}(0.25)^3 = 0.04297$ on 2 modes, contributing $\frac{\sqrt{2}\times 0.04297}{\sqrt{n}} = \mathbf{0.00219}$ at $n=768$ (already below the $0.0024$ BF16 floor!). When norm spreads across $\ge 10$ pairs ($\sigma \le 0.158$), $|e_1| \le 0.00047$. Furthermore, the single NS step at the *next* warm refresh contracts any prior $e_{\text{in}} = 0.043$ quadratically to $0.0014$.

3. **Expected effect:** *High confidence.*
   - 109M warm refresh drops from **$49.84\text{ ms} \to \sim 35.8\text{ ms}$** ($-28\%$, saving ~$1.1\text{ ms/call}$ mean).
   - Large-$n$ regime (where warm/QR reaches $0.406$): saves **$1.6\text{ ms/factor}$ at $n=4096$**, **$5.3\text{ ms/factor}$ at $n=6144$**, and **$13.0\text{ ms/factor}$ at $n=8192$ (`b12`)**.

4. **Failure modes:** If paired with a Question B change that raises the spectral cap above $\|\tilde{A}\|_2 = 0.55$ ($e_0 > 0.30 \implies |e_1| > 0.06$), a coherent multi-mode rotation can leave noticeable non-orthogonality affecting transpose `transport_moment`; keep $\|\tilde{A}\|_2 \le 0.45$ when using 1 NS step (or use Candidate B2's order-2 seed).

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** Second NS iteration in the current code mostly polishes values already drowned by the final `R_b(Y_1)` BF16 rounding ($0.0018$); independent of $\beta_C$.

6. **Cheap test:** §2.6 (i) Replay harness + §2.6 (ii) Microbenchmark. Reject if normalized Gram error over 4 chained warm refreshes ever exceeds $0.0040$ or if replay $r_C$ / $\rho_{\text{rms}}$ degrades by $>0.002$.

---

#### Candidate A6 — Subsampled Covariance EMA (Every $s=2$ Calls with $\beta_{C,2} = \beta_C^2$) + Dimension-Aware `transport_moment`

1. **Exact update:**
   - **Subsampled `covariance_ema`:** Since $C_L, C_R$ are never read on the 9 calls between refreshes, update $C_L, C_R$ only on even ages ($t \bmod 2 == 0$) with matched-half-life decay $\beta_{C,2} = \beta_C^2 = 0.9025$ and $1 - \beta_{C,2} = 0.0975$ (compiling an `ordinary_nocov` graph for odd ages).
   - **Dimension-aware `transport_moment`:** For $M \in \mathbb{R}^{m\times n}$, contract the *smaller* side's relative rotation $T$ first instead of passing both sides through world coordinates:
     - If $m > n$ ($2304\times 768$, $3072\times 768$): $T_R = \mathcal{B}(Q_R^T, Q_R^+)$, then $M^+ = \mathcal{B}(\mathcal{B}((Q_L^+)^T, \mathcal{B}(Q_L, M)), T_R)$. Cost: $4m^2n + 2mn^2 + 2n^3$ vs $4m^2n + 4mn^2$, saving $2n^2(m - n) > 0$ FLOPs ($6n^3$ saved at $m=4n$).
     - If $n > m$ ($768\times 1536$): $T_L = \mathcal{B}((Q_L^+)^T, Q_L)$, then $M^+ = \mathcal{B}(\mathcal{B}(T_L, M), Q_R^T, Q_R^+)$, saving $2m^2(n - m) > 0$ FLOPs.

2. **Cost:**
   - On odd calls (50% of all calls): ordinary-step GEMMs drop from **6 to 4** per matrix ($1.305 \to 0.870\text{ TFLOPs}$, **$-33.3\%$**), and $962\text{ MB}$ of $C_L, C_R$ BF16 R/W is eliminated completely.
   - `transport_moment`: saves $7.5\%\text{–}8.3\%$ FLOPs on rectangular shapes and reduces BF16 rounding depth on the smaller axis from 2 matmuls to 1.

3. **Expected effect:** *High confidence on speed, medium-high on loss.* Saves ~$2.5\text{ ms}$ on every odd step (**~$1.25\text{ ms/call}$** off whole-optimizer mean, or ~12% of ordinary cost) plus ~$0.4\text{ ms}$ on transport.

4. **Failure modes:** Effective sample count of the EMA drops from $\frac{1+\beta_C}{1-\beta_C} = 39$ to $\frac{1+\beta_C^2}{1-\beta_C^2} = 19.5$ steps **[verify][certain]**; however, each gradient already aggregates $32\times 1024 = 32,768$ tokens and rank $\ge 768$ inner contractions, so subsampling $s=2$ (or $s=5$ in Distributed Shampoo/SOAP variants **[verify][recalled]**) rarely hurts unless batch size is tiny.

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** *Beneficial!* In BF16 (half-ULP $u_{\text{BF16}} = 2^{-8} = 0.00391$ **[verify][certain]**), `0.95 * C + 0.05 * G G^T` stalls completely whenever $|(GG^T)_{ij} - C_{ij}| < \frac{2^{-8}}{0.05}|C_{ij}| = 7.81\%|C_{ij}|$ (for mantissas near $1.0^+$). Doubling $1 - \beta_C$ to $0.0975$ **shrinks the BF16 EMA stagnation deadzone by $1.95\times$** (down to $4.0\%$).

6. **Cheap test:** Speed in §2.6 (ii) Microbenchmark (`ordinary_nocov` vs `ordinary`). Quality **cannot** be tested in §2.6 (i) (snapshots are fixed every 10 ages); test with a 2,000-call training control (seed 204), rejecting if CE at call 500 or 2000 degrades by $>0.002$.

---

## Question B: Tracking the QR10 Trajectory for Better Loss

### Foundational Analysis of the Three Core Questions

#### 1. Why `WarmRefresh` currently misses QR10 (and hits $r_C = 0.975$ in training)
Let $B = Q^T C Q = D + E$ with descending Rayleigh quotients $d_1 \ge \dots \ge d_n > 0$ and $E = \operatorname{offdiag}(B)$.
- **Exact first-order trajectory of `QRRepair`:** Since $CQ = Q B = Q(I + E D^{-1})D$ and $D \succ 0$ is diagonal, $\operatorname{qr}(CQ).Q = Q \cdot \operatorname{qr}(I + E D^{-1}).Q$. Writing $I + E D^{-1} = (I + K_{\text{QR}})(I + R_{\text{upper}})$ with skew $K_{\text{QR}}^T = -K_{\text{QR}}$ and upper-triangular $R_{\text{upper}}$ (zero for $i > j$), matching strictly lower entries ($i > j$) gives:
  $$(K_{\text{QR}})_{ij} = +\frac{B_{ij}}{d_j} \quad (i > j), \qquad (K_{\text{QR}})_{ij} = -\frac{B_{ij}}{d_i} \quad (i < j).$$
  Thus for $i > j$ ($d_j \ge d_i$):
  $$(B_{\text{QR}}^+)_{ij} \approx B_{ij} + (d_i - d_j)(K_{\text{QR}})_{ij} = B_{ij} + (d_i - d_j)\frac{B_{ij}}{d_j} = \mathbf{\frac{d_i}{d_j}\, B_{ij}} + O(\|E\|^2).$$
  Furthermore, on **any** $2\times 2$ SPD block $B = \begin{pmatrix} d_1 & b \\ b & d_2 \end{pmatrix}$—**including equal diagonals $d_1 = d_2$**—$I + K_{\text{QR}} = \begin{pmatrix} 1 & -b/d_1 \\ b/d_1 & 1 \end{pmatrix}$ has exact polar factor $\frac{1}{\sqrt{d_1^2 + b^2}}\begin{pmatrix} d_1 & -b \\ b & d_1 \end{pmatrix} \equiv \operatorname{qr}(B).Q$ to **all orders**.

- **Three compounding throttles in current `WarmRefresh`:** Substituting $A_{ij} = -\frac{1}{2}\frac{B_{ij}}{s}\frac{h_{ij}}{h_{ij}^2 + 0.01^2}$ ($s = d_1,\; h_{ij} = \frac{d_i - d_j}{d_1}$) into $(DA - AD)_{ij} = (d_i - d_j)A_{ij}$ gives (with cap $a$):
  $$(B_{\text{warm}}^+)_{ij} \approx \left(1 - \frac{a}{2}\frac{h_{ij}^2}{h_{ij}^2 + 0.01^2}\right) B_{ij}.$$
  1. **Factor-of-$\frac{1}{2}$ under-relaxation:** Exact $2\times 2$ Jacobi satisfies $\tan(2\theta) = \frac{2B_{ij}}{d_i - d_j} \implies \theta \approx \frac{B_{ij}}{d_i - d_j}$ **[verify][certain]** (the $2$'s cancel!). The $-\frac{1}{2}$ halves the step even at $a = 1, h_{ij} = 1$.
  2. **Global normalization by $s = d_1$ freezes power-law tails:** Under §2.2's $\lambda_i \propto 1/i$, every pair with $i, j > 100$ ($87\%$ of coordinates at $n=768$; $98.8\%$ at $n=8192$) has $d_i, d_j < 0.01 d_1 \implies |h_{ij}| < 0.01$, suppressing motion by $\frac{1}{2}\frac{h_{ij}^2}{0.01^2} < 3\%$ even when $d_i / d_j = 2$, while Adam's $M/(\sqrt{V}+\epsilon)$ gives those tail coordinates equal step weight!
  3. **$\sqrt{n}$-dependent Frobenius cap:** In §2.2, $A_0$ has entries $\sim 0.2/\sqrt{n}$ ($r_C \approx 0.20$). Over the ~$100n$ pairs with $h_{ij} \gg 0.01$, $\|A\|_F/\sqrt{2} \approx \sqrt{100n \cdot (0.1/\sqrt{n})^2} \approx 1.0$, triggering $a \approx 0.5$. Combined with $-\frac{1}{2}$, max contraction is $1 - 0.25 = 0.75$, predicting $r_C^+ \approx 0.20 \times 0.75 = \mathbf{0.150}$—matching §2.2's measured $0.150$! Worse, for a full-rank rotation of angle $\theta$ across all $n$ coordinates, $\|A\|_2 \sim \theta$ (governing Taylor/NS validity) while $\|A\|_F/\sqrt{2} \sim \theta\sqrt{n/2}$, throttling per-coordinate rotation to $0.5/\sqrt{1536} = \mathbf{0.73^\circ}$ per refresh at $n=3072$.
  4. **Why trained max $r_C$ hits $0.965\text{–}0.975$:** When a sharp gradient spike aligns across $k \approx 20$ old basis vectors ($B \approx \frac{\lambda_1}{k}\mathbf{1}_k\mathbf{1}_k^T$), all $k$ coordinates get **nearly equal diagonals** $d_1 \approx \dots \approx d_k \approx \lambda_1/k$, giving $r_C = \sqrt{1 - 1/k} = \sqrt{19/20} = \mathbf{0.9747}$ and $h_{ij} \approx 0$—completely stalling `WarmRefresh` until QR repair.

#### 2. Is $r_C$ the right proxy for loss? What to measure instead
**No.** Because $\|Q^T C Q\|_F^2 = \sum_i \lambda_i^2$ is dominated by the top few eigenvalues in a power-law spectrum, $r_C$ ignores misalignment across $>95\%$ of coordinates that Adam scales up to unit step size.
To derive the *exact* metric governing the SOAP step, let $g = C^{1/2}z$ ($z \sim \mathcal{N}(0, I_n)$) and compare the approximate whitened step $Q D_Q^{-1/2} Q^T g$ ($D_Q = \operatorname{diag}(Q^T C Q)$) against ideal $C^{-1/2}g = U z$. Expanding $Q = U(I + A)$ ($A^T = -A$ small, $D_Q = \Lambda + O(A^2)$, $B_{ij} = (\lambda_j - \lambda_i)A_{ij} + O(A^2)$):
$$\left(U^T(Q D_Q^{-1/2} Q^T C^{1/2} - I)U\right)_{ij} = A_{ij}\left(1 - \sqrt{\frac{\lambda_j}{\lambda_i}}\right) + O(A^2).$$
Summing squares over $\{i, j\}$ ($A_{ji} = -A_{ij}$):
$$\Delta_{\text{pre}}^2(Q; C) := \frac{1}{n}\mathbb{E}_z\|Q D_Q^{-1/2} Q^T g - C^{-1/2}g\|_2^2 = \frac{1}{n}\sum_{i < j} |A_{ij}|^2 (\sqrt{\lambda_i} - \sqrt{\lambda_j})^2\left(\frac{1}{\lambda_i} + \frac{1}{\lambda_j}\right) + O(A^3)$$
$$= \frac{1}{n}\sum_{i < j} \frac{B_{ij}^2}{\lambda_i \lambda_j}\cdot \underbrace{\frac{\lambda_i + \lambda_j}{(\sqrt{\lambda_i} + \sqrt{\lambda_j})^2}}_{\in [1/2,\, 1]\;\forall \lambda_i,\lambda_j > 0} + O(A^3).$$
Up to a universal factor in $[1/\sqrt{2}, 1]$, **the exact whitened update error is the regularized RMS off-diagonal correlation $\rho_{\text{rms}}$** (already in your §2.6 i harness!):
$$\rho_{\text{rms}}(Q; C) = \sqrt{\frac{1}{n(n-1)}\sum_{i \neq j} \frac{B_{ij}^2}{(d_i + \epsilon_C)(d_j + \epsilon_C)}}, \qquad \epsilon_C \approx 10^{-12}\text{–}10^{-10}.$$
Because your target is specifically **QR10** (not `eigh`), supplement $\rho_{\text{rms}}$ in §2.6 (i) with **relative preconditioner distance to the replayed QR10 trajectory**: $\|Q(D_Q+\epsilon_C I)^{-1/2}Q^T(C+\epsilon_C I)^{1/2} - Q_{\text{QR10}}(D_{\text{QR10}}+\epsilon_C I)^{-1/2}Q_{\text{QR10}}^T(C+\epsilon_C I)^{1/2}\|_F / \sqrt{n}$.

#### 3. Role of the Call-500 Gap, $V$, and Column Order
- **Call-500 gap (+0.124 CE):** At $k=1$, $Q$ comes from `eigh` of a *single* random-init batch gradient $G_1$ (rank-limited and noisy). Over calls 1–500 (spanning the 200-call warmup right across peak LR $1.5\times 10^{-3}$), the true eigenbases undergo $O(1)$-radian rotations across all $n$ dimensions. Throttled to $0.73^\circ$ RMS by the $\sqrt{n}$ Frobenius cap and blind below $0.01 d_1$, `warm10/QR50` tracks like `QR50` ($5.0015$ vs $5.0180$), wasting peak-LR steps; under a fixed 10k cosine decay, early lost progress becomes a persistent effective-step lag (~$0.005\text{–}0.007\text{ nats}$) stacked atop ongoing tracking/transport loss.
- **$V$ and column order:** Leaving $V$ unpermuted during *infinitesimal* non-crossing rotations is benign, **except**: (i) over 40 steps ($\approx 3$ EMA half-lives) eigenvalues cross, and B1's triangular QR generator *requires* descending $d$ so $d_i/d_j \le 1$ for $i > j$; (ii) much worse, **rotating $M$ linearly while only permuting (or freezing) $V$ breaks Adam's moment bound!**
  *2×2 proof:* Suppose Mode 1 has $M_1 = 1, V_1 = 1$ and Mode 2 has $M_2 = 10^{-3}, V_2 = 10^{-6}$ ($|M_i/\sqrt{V_i}| = 1$). A rotation by $\theta = 0.2\text{ rad}$ ($\pi = \text{id}$) transports $M_2^+ = -\sin\theta\, M_1 + \cos\theta\, M_2 \approx -0.199$, while leaving $V_2^+ = 10^{-6}$, producing a **$199\times$ step spike** $M_2^+/\sqrt{V_2^+} \approx -199$ on the next step (especially severe right after 50-step `QRRepair` corrects a $r_C = 0.975$ spike!). Candidate B4 fixes this exactly.

---

### Question B Candidates ($\le 2\times$ Warm Cost at $n = 768\text{–}3072$)

#### Candidate B1 — Sorted Triangular Power-QR Skew Generator (`K_QR`) + Block-Power Spectral Cap

1. **Exact update:** Replace Jacobi's symmetric gap generator and Frobenius cap in `warm_basis` (and permute $V$'s corresponding axis by `order`, just as in `qr`):
   ```python
   def warm_qr_skew_basis(cov, q, v_0_fixed, compute_dtype, basis_dtype):
       qb, cb = q.to(compute_dtype), cov.to(compute_dtype)
       b = mm32(torch.bmm(qb.transpose(-1, -2), cb), qb)
       b = 0.5 * (b + b.transpose(-1, -2))
       d = b.diagonal(dim1=-2, dim2=-1)
       order = d.argsort(dim=-1, descending=True)
       # Permute Q, d, and rows & cols of B into descending Rayleigh order:
       qb = qb.gather(-1, order.unsqueeze(-2).expand_as(qb))
       d = d.gather(-1, order)
       b = b.gather(-2, order.unsqueeze(-1).expand_as(b)).gather(-1, order.unsqueeze(-2).expand_as(b))
       s = d.abs().amax(-1, keepdim=True).clamp_min(1e-30)
       d_safe = torch.hypot(d.clamp_min(0.0), 1e-4 * s)          # local Rayleigh denominator, soft floor 1e-4*s
       lower = (b / d_safe.unsqueeze(-2)).tril(-1).clamp(-1.0, 1.0)
       k_qr = lower - lower.transpose(-1, -2)                    # exact 1st-order QR skew generator
       # 2-step block power iteration on deterministic fixed (n, 8) probe for spectral norm ||k_qr||_2:
       w1 = torch.bmm(k_qr, torch.bmm(k_qr, v_0_fixed.expand(b.shape[0], -1, -1)))
       w1_n = w1 / torch.linalg.vector_norm(w1, dim=-2, keepdim=True).clamp_min(1e-20)
       w2 = torch.bmm(k_qr, torch.bmm(k_qr, w1_n))
       spec_est = torch.linalg.vector_norm(w2, dim=-2).amax(-1, keepdim=True).sqrt().unsqueeze(-1)
       a = k_qr * (0.45 / spec_est.clamp_min(1e-12)).clamp(max=1.0)   # keeps ||a||_2 <= ~0.50 < sqrt(2)
       y = torch.baddbmm(qb, qb, a.to(compute_dtype), beta=1.0, alpha=1.0)
       for _ in range(2):   # (or 1 step with A5)
           gram = mm32(y.transpose(-1, -2), y)
           res = (gram - torch.eye(y.shape[-1], device=y.device, dtype=gram.dtype)).to(compute_dtype)
           y = torch.baddbmm(y, y, res, beta=1.0, alpha=-0.5)
       return y.to(basis_dtype).contiguous(), order
   ```
   *(Optional Jacobi-boost multiplier on `lower` for $i > j$: setting $r_{ij} = \frac{(d_j - d_i)_+}{d_{\text{safe},j}}$ and multiplying `lower` by $\frac{r_{ij} + 0.25}{r_{ij}^2 + 0.25} \in [1, 1.62]$ gives exact QR gain $1$ at both $d_i = d_j$ and $d_i \ll d_j$, while accelerating $d_i/d_j \in [0.2, 0.8]$ toward exact Jacobi.)*

2. **Cost:** **7 BF16 $n\times n$ GEMMs** (identical to current `warm_basis`; **5 GEMMs** with A5) + 4 FP32 $n\times 8$ block matvecs ($64n^2$ FLOPs = **0.15% of 1 GEMM** at $n=3072$) + `argsort`/gathers on $Q, B, V$ (~$1.2\text{ ms}$ across all 96 factors). Total 109M warm cost: **~$1.03\times$ current warm** (~$51\text{ ms}$, or ~$37\text{ ms}$ with A1+A5).

3. **Expected effect:** *High confidence.*
   - On §2.2 synthetic inputs ($r_C \approx 0.20$), single-step $r_C$ drops from $0.150 \to \mathbf{\sim 0.035\text{–}0.050}$ (approaching QR's $0.025$), and $\rho_{\text{rms}}$ drops by $>3\times$ across the power-law tail.
   - Eliminates equal-diagonal stalling ($r_C = 0.975$ spikes break immediately) and removes the $O(1/\sqrt{n})$ early-training rotation bottleneck, closing an estimated **60–80% of the loss gap** to QR10.

4. **Failure modes:**
   - If the fixed $(n, 8)$ probe $V_0$ were orthogonal to a rank-2 coherent skew spike (probability negligible if $V_0$ uses 8 columns of a normalized Walsh–Hadamard / DCT matrix), `spec_est` could underestimate $\|K_{\text{QR}}\|_2$, causing Newton–Schulz overshoot if $\sigma_0(Y_0) = \sqrt{1 + \alpha^2} > \sqrt{3}$ ($\alpha > \sqrt{2}$). Safeguard: also clamp `spec_est` below by $\|K_{\text{QR}}\|_F / \sqrt{2k_{\text{max}}}$ or verify $\max_j (Y_0^T Y_0 - I)_{jj} = \max_j \|a_{:,j}\|_2^2$.
   - Gathering both rows and columns of FP32 `b` adds $8n^2\text{ B}$ memory traffic ($0.85\text{ ms}$ across 96 factors at 109M)—can be fused into the Triton kernel reading `b[order[i], order[j]]` directly without materializing permuted `b`.

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** Because $B$ uses an intermediate BF16 `bmm(qb.T, cb)`, tail blocks below $10^{-4} d_1$ contain BF16 rounding noise from dominant modes; `hypot(d, 1e-4 * s)` and `clamp(-1, 1)` prevent noise amplification while reaching **100× deeper** into the spectrum than $0.01 s$.

6. **Cheap test:** §2.6 (i) Open-loop replay (all widths 256–2048) + §2.6 (ii) Microbenchmark, plus §2.5 Fact 3's $2\times 2$ equal-diagonal test $C = \begin{pmatrix}1 & 0.5\\0.5 & 1\end{pmatrix}$. Reject if replay $\rho_{\text{rms}}$ is not at least halfway from current `warm10/QR50` to `QR10`, or if latency exceeds $1.15\times$ current warm.

---

#### Candidate B2 — Local-Scale Full-Gain Jacobi + Order-2 Exponential Seed (`I + A + ½A²`) with 1 NS Step

1. **Exact update:** Keeps symmetric unsorted `WarmRefresh` ($V$ untouched), fixing all three scaling bugs and replacing Euler + 2 NS with an order-2 Taylor/Padé exponential seed + 1 NS step:
   $$s_{ij} = \max(|d_i|, |d_j|, 10^{-4} s), \qquad g_{ij} = \frac{d_i - d_j}{s_{ij}}, \qquad A_{ij} = -\frac{B_{ij}}{s_{ij}}\frac{g_{ij}}{g_{ij}^2 + 0.02^2},$$
   scale $\tilde{A} = R_b(a A)$ using B1's block-power spectral cap ($0.50 / \hat{\sigma}_{\max}(A)$), then:
   $$S = R_b\!\left(\tilde{A} + \frac{1}{2}\mathcal{F}(\tilde{A}, \tilde{A})\right), \qquad Y_0 = R_b(Q + \mathcal{F}(Q, S)), \qquad Y_1 = R_b\!\left(Y_0 - \frac{1}{2}\mathcal{F}(Y_0, R_b(\mathcal{F}(Y_0^T, Y_0) - I))\right).$$

2. **Cost & algebraic identity:**
   - **6 BF16 $n\times n$ GEMMs** ($Q^T C$, $(Q^T C)Q$, $\tilde{A}^2$, $Q S$, $Y_0^T Y_0$, $Y_0 E_0$ = $12n^3$ FLOPs—**14.3% cheaper** than current 7-GEMM warm!).
   - **Exact orthogonality identity:** Since $\tilde{A}^T = -\tilde{A} \implies (\tilde{A}^2)^T = \tilde{A}^2$:
     $$\left(I - \tilde{A} + \frac{1}{2}\tilde{A}^2\right)\left(I + \tilde{A} + \frac{1}{2}\tilde{A}^2\right) = \left(I + \frac{1}{2}\tilde{A}^2\right)^2 - \tilde{A}^2 = I + \frac{1}{4}\tilde{A}^4.$$
     The $O(\tilde{A}^2)$ Gram error cancels identically! At $\|\tilde{A}\|_2 = 0.5$, $\|Y_0^T Y_0 - I\|_2 \approx \frac{1}{4}(0.5)^4 = 0.0156$, which **one** NS step crushes to $\frac{3}{4}(0.0156)^2 = \mathbf{0.00018}$ (13× below BF16 rounding), while matching $\exp(\tilde{A})$ to $O(\tilde{A}^3)$.

3. **Expected effect:** *Medium-high confidence.* Faster than current warm (**~$43\text{ ms}$ vs $49.8\text{ ms}$** at 109M; $-14\%$ at $n=8192$), with quadratic $O(\|E\|^2)$ off-diagonal contraction on all pairs with $|d_i - d_j| / \max(d_i, d_j) > 0.02$ across the entire spectrum.

4. **Failure modes:** Still vanishes at exact equal diagonals ($d_i = d_j$, so it will **not** clear $r_C = 0.975$ coherent equal-diagonal spikes until the 50-step QR repair), and simultaneous full-gain Jacobi can over-rotate tightly coupled 3-cycles $(i,j,k)$ when $\|E\|$ is large ($r_C > 0.5$) unless capped at $\|\tilde{A}\|_2 \le 0.45$.

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** Computing $\tilde{A}^2$ in BF16 (`mm32`) loses precision only when $\|\tilde{A}\|_2 < 10^{-4}$, where $\frac{1}{2}\tilde{A}^2 < 10^{-8}$ is already zero in BF16 anyway.

6. **Cheap test:** §2.6 (i) Replay + §2.6 (ii) Microbenchmark. Comparing **B1 vs B2** in replay directly isolates whether equal-diagonal symmetry breaking + sorting (B1) matters beyond fixing local scale + spectral cap (B2). Reject B2 if replay max-$r_C$ spikes remain $>0.85$.

---

#### Candidate B3 — Regularized Column-Scaled Cholesky-QR1 + 1-Step BF16 Newton–Schulz ("Warm CholQR-NS")

1. **Exact update:** Instead of approximating QR in $B$-space, directly execute one regularized power-QR step and remove single-pass Cholesky-QR's orthogonality drift ($0.034$ at $n=4096$) with 1 BF16 Newton–Schulz polish:
   ```python
   def warm_cholqr_ns(cov, q, compute_dtype, basis_dtype):
       z = mm32(cov, q)                                              # 1 BF16 GEMM -> FP32
       qf = q.float()
       r = (qf * z).sum(-2)
       order = r.argsort(dim=-1, descending=True)
       s = r.abs().amax(-1, keepdim=True).clamp_min(1e-30).unsqueeze(-2)
       z_reg = z + (5e-4 * s) * qf                                   # kappa(z_reg) <= 2001 at ALL n
       z_ord = z_reg.gather(-1, order.unsqueeze(-2).expand_as(z_reg))
       z0 = z_ord / torch.linalg.vector_norm(z_ord, dim=-2, keepdim=True).clamp_min(1e-30)
       g = torch.bmm(z0.transpose(-1, -2), z0) + 1e-4 * torch.eye(q.shape[-1], device=q.device)
       u, _ = torch.linalg.cholesky_ex(g, upper=True, check_errors=False)
       y1 = torch.linalg.solve_triangular(u, z0, upper=True, left=False).to(compute_dtype)
       # 1 BF16 Newton-Schulz polish (valid here because R^{-1} ALREADY broke polar symmetry!):
       res = (mm32(y1.transpose(-1, -2), y1) - torch.eye(q.shape[-1], device=q.device)).to(compute_dtype)
       q_new = torch.baddbmm(y1, y1, res, beta=1.0, alpha=-0.5)
       return q_new.to(basis_dtype).contiguous(), order
   ```
   *(Also permute $V$ by `order`, as in `qr`.)*

2. **Cost:**
   - **3 BF16 $n\times n$ GEMMs** ($Z = CQ$, $Y_1^T Y_1$, $Y_1 E_1$: $6n^3$ FLOPs);
   - **1 FP32/TF32 Gram GEMM** ($2n^3$) + **1 FP32 `cholesky`** ($\frac{1}{3}n^3$) + **1 FP32 `solve_triangular`** ($n^3$).
   - Total time from §2.2 primitives: §2.2's 1-pass `Cholesky-QR` took $72.0\text{ ms}$ across the 96 factors (with FP32 $Z=CQ$); using BF16 `mm32` for $Z=CQ$ (~$-7\text{ ms}$) + 1 BF16 NS step ($\frac{2}{7}\times 49.8 \approx +14\text{ ms}$) gives **~$79\text{ ms}$** across all 96 factors (**$1.58\times$ current warm**, well under $2\times$ at $n=768\text{–}3072$, and **$7.5\times$ faster than Householder QR**; at $n=8192$, ~$52\text{ ms}$ vs $45.5\text{ ms}$ warm = **$1.14\times$ warm** and $2.15\times$ faster than QR!).

3. **Expected effect:** *Very high confidence.*
   - Why §2.5 Fact 2 does not block the NS step: Fact 2 applies to $CQ = QB$ ($B$ symmetric). After upper-triangular $U^{-1}$, $Y_1 = Q_{\text{QR}}(I + E_{\text{orth}})$ already has the QR basis with $\|E_{\text{orth}}\|_2 \lesssim 0.03$; 1 NS step reduces orthogonality error to $\frac{3}{4}(0.03)^2 \approx 0.0007$ (stored as $0.0018$ in BF16), perturbing $Q_{\text{QR}}$ by only $O(\|E_{\text{orth}}\|^2) < 10^{-3}$.
   - Tracks `QR10` almost identically down to $\lambda_i \sim 5\times 10^{-4}\lambda_1$ ($r_C: 0.20 \to \sim 0.026$ in 1 step), closing **80–95% of the warm-vs-QR10 basis gap**.

4. **Failure modes:** Because `g` uses default `high` precision (TF32, $u \approx 2^{-10}$), setting ridge $<10^{-4}$ without IEEE FP32 on `g` can let `cholesky_ex` hit non-positive pivots on rank-deficient $C$; with ridge $5\times 10^{-4} s$ + diagonal shift $10^{-4}$ on unit-column `z0`, `g` is diagonally dominant over TF32 noise at all $n \le 8192$, at the mild cost of freezing modes below $5\times 10^{-4}\lambda_1$ until the 50-step full QR repair.

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** Eliminates the $0.034$ Gram drift of raw Cholesky-QR (returning $0.0018$ BF16 Gram error) and keeps $V$ continuously sorted every 10 calls.

6. **Cheap test:** §2.6 (i) Replay + §2.6 (ii) Microbenchmark across $n = 256\text{–}8192$. Reject if `b12` cost exceeds $2.0\times$ warm on the 109M factor mix, if any NaN appears at $n=8192$, or if replay $\rho_{\text{rms}}$ differs from `QR10` by $>5\%$ on modes $\lambda_i > 10^{-3}\lambda_1$.

---

#### Candidate B4 — Doubly-Stochastic $V$-Transport $V \leftarrow (T_L^{\circ 2})V(T_R^{\circ 2})^T$ + Pre-Update Refresh Ordering

1. **Exact update:** On refresh calls ($t \bmod 10 == 0$, both `warm` and `qr`):
   1. Update $C_L, C_R$ from $G_b$ **first**, and compute $Q_L^+, Q_R^+$ (no separate $V$ gather needed—subsumed below!).
   2. Form relative transitions $T_L = \mathcal{B}((Q_L^+)^T, Q_L) \in \mathbb{R}^{m\times m}$ and $T_R = \mathcal{B}((Q_R^+)^T, Q_R) \in \mathbb{R}^{n\times n}$.
   3. Transport old $M$ and $V$ into the new basis *before* incorporating $\bar{G}_t$:
      $$M \leftarrow \mathcal{B}(\mathcal{B}(T_L, M), T_R^T), \qquad P_L = \frac{T_L^{\circ 2}}{\operatorname{sum}_{\text{col}}(T_L^{\circ 2})}, \quad P_R = \frac{T_R^{\circ 2}}{\operatorname{sum}_{\text{col}}(T_R^{\circ 2})}, \qquad V \leftarrow R_b(\mathcal{F}(\mathcal{B}(P_L, V), P_R^T)).$$
   4. Project $\bar{G}_t = \mathcal{B}(\mathcal{B}((Q_L^+)^T, G_b), Q_R^+)$ and run moment EMA + parameter step $U_t$ in the **fresh** basis $Q^+$.

2. **Cost & variance derivation:**
   - Forming $T_L, T_R$: $2m^3 + 2n^3$ BF16 FLOPs. Applying $T_L, T_R$ to $M$ takes $2m^2n + 2mn^2$ (saving $2m^2n + 2mn^2$ vs 4-matmul world transport!). Applying $P_L, P_R$ to $V$ takes $2m^2n + 2mn^2$ BF16 FLOPs.
   - Net extra FLOPs per refresh over current `transport_moment`: just **$2m^3 + 2n^3$** ($2\sum n^3 = 1.13\text{ TFLOPs}$ across 96 factors $\approx \mathbf{+6.5\text{ ms}}$ per refresh event = **$+0.65\text{ ms/call}$ amortized**, $+12\%$ of a warm event!). *(If $m \gg n$ on warm steps, setting $T_L \approx I - \tilde{A}_L$ avoids the $2m^3$ GEMM entirely!)*
   - **Exact variance justification:** Under $\bar{G}^+ = T_L \bar{G} T_R^T$ with uncorrelated old-frame coordinates $\mathbb{E}[\bar{G}_{ab}\bar{G}_{cd}] = V_{ab}\delta_{ac}\delta_{bd}$, $\mathbb{E}[(\bar{G}^+)_{ij}^2] = \sum_{a,b} (T_L)_{ia}^2 V_{ab} (T_R)_{jb}^2 = ((T_L^{\circ 2})V(T_R^{\circ 2})^T)_{ij}$. When $Q^+ = Q\Pi$ is a pure permutation, $T^{\circ 2} = \Pi^T$ reproduces exact $V$-permutation; when a top mode ($V_1$) mixes by $\sin\theta$ into a tail mode ($V_2 \ll V_1$), $V_2^+ \ge \sin^2\theta\, V_1$, eliminating the $O(|\sin\theta|\sqrt{V_1/V_2})$ update explosion.

3. **Expected effect:** *Medium-high confidence.* Eliminates post-QR / post-warm step-size spikes (especially at `t % 50 == 0` when QR corrects drifted warm bases) and upgrades 10% of all parameter steps from the 10-step-stalest basis to a 0-step-fresh basis, targeting ~$0.002\text{–}0.005\text{ nats}$ steady-state CE improvement. *(Note: incompatible with A3's post-`W` async overlap unless refresh uses $C_{t-1}$, which shifts $C$ by 1 step at $\beta_C=0.95$.)*

4. **Failure modes:** Repeated multiplication by $P_L, P_R$ acts as a slight diffusive smoothing across $V$ when $T \neq I$ ($O(\theta^2)$ mixing every 10 steps); with $\beta_2 = 0.95$, $V$'s memory decays by $0.95^{10} = 0.60$ between refreshes anyway, preventing long-term variance equalization unless $\theta$ is large every refresh.

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** Row-normalizing $T^{\circ 2}$ in FP32 before casting $P_L, P_R$ to BF16 removes the $0.0024$ Gram drift of $Q$ so $\sum_a P_{ia} = 1$ to BF16 precision (preserving $\sum_{ij} V_{ij}$), and $(1-\beta_1)\bar{G}_t$ avoids 4 BF16 roundings of world transport.

6. **Cheap test:** Timing in §2.6 (ii) Microbenchmark; quality **cannot** be tested in §2.6 (i) (no $M, V$, or loss). Smallest test: 2,000-call training run (seed 204), logging maxcoordinate update $\|M_f/(\sqrt{V_f}+\epsilon)\|_\infty$ on steps $t \equiv 1 \pmod{50}$ and CE at 500/2000. Reject if CE improves by $<0.001$ and update spikes do not drop.

---

#### Candidate B5 — Dense Early Burn-In (`QR10` for $t \le 300$, then `Warm10/QR50`)

1. **Exact update:** Change the schedule line to:
   ```python
   event = 'cold' if not age else 'ordinary' if age % 10 else ('qr' if (age <= 300 or age % 50 == 0) else 'warm')
   ```

2. **Cost:** Replaces 24 warm events ($t \in \{10,20,30,40,\dots,290\}\setminus 50\mathbb{Z}$) with 24 QR events over a 10,000-call run:
   $$\Delta t_{\text{mean}} = \frac{24 \times (601.18 - 61.61)\text{ ms}}{10{,}000} = \mathbf{+1.30\text{ ms/call}} \quad (\text{+0.42\% whole-step time; +0.18 ms/call with A4}).$$
   Zero steady-state cost for $t > 300$.

3. **Expected effect:** *High confidence.* Completely eliminates the $+0.124$ call-500 gap during the non-stationary $G_1$-init $\to$ post-warmup transition ($t \le 300$ spans the entire 200-call LR warmup plus 7.4 covariance half-lives past peak LR!), recovering the ~$0.004\text{–}0.007\text{ nats}$ of irreversible cosine-schedule lag at virtually zero amortized cost.

4. **Failure modes:** Does not fix late-horizon spike stalls ($r_C = 0.975$ at age 9990) if used *without* B1 or B3; at very short horizons ($\le 500$ steps) its overhead is $60\%$ of QR10 rather than $2\%$.

5. **Interaction with BF16 storage and $\beta_C = 0.95$:** By $t = 300$, single-batch init bias has decayed by $0.95^{300} = 2\times 10^{-7}$, handing `WarmRefresh` a well-conditioned, full-rank, stationary eigenbasis.

6. **Cheap test:** Partially testable in §2.6 (i) Replay ($t \in [0, 1000]$, checking whether post-$300$ $\rho_{\text{rms}}$ converges to a lower steady-state attractor); definitively tested in a **500-to-2,000 call training run**, rejecting if call-2000 CE drops by $<0.003$ when applied to current `warm10/QR50`.

---

## Pitfalls in the Current Design and Measurements

Beyond the `WarmRefresh` throttling bugs, $r_C$ blindness, and $M$-vs-$V$ mismatch analyzed above, watch out for four subtle pitfalls:

1. **BF16 EMA Stagnation ("Swamping")—Not Underflow—in `M`, `V`, and `C`:**
   BF16 has FP32's 8-bit exponent (min normal $2^{-126} \approx 1.18\times 10^{-38}$ **[verify][certain]**), so $V \sim 10^{-16}$ does *not* underflow. Instead, BF16's 7-bit mantissa (half-ULP $u_{\text{BF16}} = 2^{-8} = 0.00391$ near $1.0^+$ **[verify][certain]**) creates a hard stagnation deadzone in any EMA $X \leftarrow R_b(\beta X + (1-\beta)Y)$: whenever $|Y - X| < \frac{2^{-8}}{1-\beta}|X|$, the update rounds to **exact zero change**. At $\beta = 0.95$, all drifts smaller than **$3.9\%\text{–}7.8\%$** stall in `M`, `V`, and `C`. Worse, if you ever test standard SOAP/Adam $\beta_2$ or $\beta_C \in \{0.99, 0.999\}$ **[verify][recalled]** with BF16 state (without stochastic rounding or FP32 master moments), the deadzone balloons to **$39\%\text{–}391\%$** and optimization breaks silently.

2. **Confounded Speed Baseline in §2.1 & Pending `BF16 QR10` Control in §2.3:**
   - §2.1 compares **compiled BF16** `warm10/QR50` ($26.08\text{ ms}$) against **eager FP32** `QR10` ($95.23\text{ ms}$). From your own §2.1 primitives, a **matched compiled BF16 `QR10`** costs $0.90(10.14) + 0.10(601.18) = \mathbf{69.25\text{ ms/call}}$ ($352.7\text{ ms}$ step)—so warm refreshes save $43.2\text{ ms/step}$ (12.2%), while $27.8\text{ ms/step}$ came from compiling + BF16 ordinary steps. Meanwhile, `BF16 QR50` takes $22.86\text{ ms/call}$ ($3.56\text{ ms/call}$ faster than `warm10/QR50`), and `BF16 QR20` ($40.46\text{ ms/call}$) currently *beats* `warm10/QR50` on loss for $+14\text{ ms/call}$.
   - Without `BF16 QR10` at 10k calls, part of the $+0.011$ gap (likely $0.002\text{–}0.004\text{ nats}$ from BF16 EMA stagnation + 4-matmul BF16 `M` transport) cannot be attributed cleanly to `WarmRefresh`.

3. **TF32 Precision Leak Inside "FP32 Islands" (`float32_matmul_precision = 'high'`):**
   Setting `"high"` lets every `torch.bmm` on `float32` tensors silently truncate inputs to **TF32 (10 mantissa bits, $u \approx 10^{-3}$)** **[verify][certain]**. While benign when inputs were just cast from 7-bit BF16 (`qr_basis`'s `c @ old`), it silently degrades any true FP32 Gram product like $Z^T Z$ in Cholesky-QR or `gram` in your optional FP32-basis Newton–Schulz island (`else: gram = torch.bmm(y.T, y)` in `warm_basis`, which claims to *"measure the actual FP32 basis Gram"* but actually measures a 10-bit TF32 Gram!).

4. **Open-Loop Replay (§2.6 i) and Synthetic Spectrum (§2.2) Optimism:**
   - §2.2 uses $\lambda_i \propto 1/i$ ($\kappa(C) = n \le 8192$, full-rank), whereas trained transformer factors exhibit steeper power laws ($\kappa(C) > 10^6$) and transient low-rank spikes ($r_C = 0.975$).
   - Open-loop replay feeds clean `FP32 QR10` covariances; in closed loop, a badly preconditioned step perturbs weights into sharper curvature directions, creating positive feedback between basis error and gradient spikes, while hiding $M/V$ transport shocks completely.

---

## Summary: Rankings, Top Pitfalls, and Next Measurements

### Top 5 Candidates for Question A (Speed)
1. **A4 (`sCholeskyQR` replacing Householder QR, A-eq):** Attacks the single largest slice of optimizer time (45.3%), cutting 109M QR events ~4.8× ($594 \to \sim 120\text{ ms}$, **$-9.4\text{ ms/call}$ mean**) with bounded conditioning at all $n$.
2. **A3 (Multi-stream QR + post-`W` async overlap, A-eng):** Pure bit-exact engineering that parallelizes low-occupancy QR across SMs and hides refresh/transport behind Call $k+1$'s $283\text{ ms}$ Fwd+Bwd (**$26.1 \to \sim 12\text{ ms/call}$ exposed**).
3. **A1 (Fused `baddbmm` epilogues + $b=60$ factor coalescing + contiguous param banks, A-eng):** Eliminates $32n^2\text{ B/factor}$ of FP32 DRAM traffic (fixing the $n=8192$ `b12` slowdown), halves $n=768$ launch overhead, and removes `stack`/`copy_` every step.
4. **A2 (Triton symmetric `syrk` for `covariance_ema` and NS Grams, A-eng):** Cuts one-third of ordinary-step FLOPs nearly in half ($-1\text{ ms}$ every call) and trims 14% of warm FLOPs while enforcing exact symmetry.
5. **A5 (1-step Newton–Schulz in `WarmRefresh`, A-eq):** Provably stays at or below the $0.0024$ BF16 rounding floor while immediately cutting warm FLOPs by **28.6%** across all $n$ (especially valuable at $n=4096\text{–}8192$).

### Top 5 Candidates for Question B (Tracking QR10 & Loss)
1. **B1 (Sorted Triangular Power-QR Skew Generator `K_QR` + Block-Power Spectral Cap):** Matches `QRRepair` to first order across all scales (and to *all* orders on $2\times 2$ equal diagonals) at **1.0× warm cost**, simultaneously fixing the $-\frac{1}{2}$ damping, $0.01 d_1$ tail blindness, $O(1/\sqrt{n})$ Frobenius cap, and $r_C=0.975$ stalls.
2. **B3 ("Warm CholQR-NS": Regularized Column-Scaled Cholesky-QR1 + 1 BF16 NS step):** Directly executes the power+QR step with $0.0018$ BF16 orthogonality and zero $n \ge 6144$ NaNs at **~1.58× warm cost** (7.5× cheaper than QR), offering the tightest possible match to QR10.
3. **B4 (Doubly-stochastic $V \leftarrow (T_L^{\circ 2})V(T_R^{\circ 2})^T$ + pre-update refresh):** Cures the $O(|\sin\theta|\sqrt{V_1/V_2})$ post-refresh step explosion when $M$ rotates without $V$, and uses fresh bases on refresh steps for $+0.65\text{ ms/call}$.
4. **B5 (`QR10` burn-in for $t \le 300$, then `warm10/QR50`):** Erases the $+0.124$ call-500 warmup gap right across peak LR for just **$+1.30\text{ ms/call}$** (+0.4% step time) over 10k calls.
5. **B2 (Local-scale full-gain Jacobi + `I + A + ½A²` seed + 1 NS step):** A **0.86× warm cost** pure-Jacobi baseline with 4th-order pre-NS orthogonality that cleanly isolates whether equal-diagonal symmetry breaking (B1) is needed beyond fixing local scaling and the spectral cap.

### Top 3 Pitfalls
1. **The three compounding `WarmRefresh` throttles + equal-diagonal stall:** $-\frac{1}{2}$ gain, global $s = d_1$ gap normalization (freezing $>87\%$ of a power-law spectrum that Adam weights equally), and $\sqrt{n}$-scaling of $\|A\|_F$ ($0.73^\circ$ RMS cap at $n=3072$), plus $h_{ij} \approx 0$ paralysis on $k$-way split spikes ($r_C \approx \sqrt{1 - 1/k} = 0.975$).
2. **Rotating $M$ without mixing $V$:** Linear $M$ transport paired with unrotated/permuted-only $V$ violates $|M/\sqrt{V}| \le 1$ by up to $|\sin\theta|\sqrt{V_{\max}/V_{\min}}$ (~$10^2\times$) on the step immediately following large QR repairs.
3. **BF16 EMA mantissa stagnation (and TF32 leakage into "FP32" Gram islands):** A $3.9\%\text{–}7.8\%$ deadzone on all state updates at $\beta = 0.95$ (catastrophic if $\beta \to 0.99+$), alongside `high` matmul precision silently running FP32 Gram checks in 10-bit TF32.

### Up to 5 Measurements Wanted Next
1. **Complete the 10k-call (and 2k-call) compiled `BF16 QR10` control** (plus an FP32-moments / BF16-factors ablation) to split the $+0.011\text{ nat}$ gap into BF16 EMA stagnation vs warm-refresh tracking error, and give the true compiled BF16 speed baseline ($69.25\text{ ms/call}$ predicted).
2. **Inspect the 96-factor distribution of trained $r_C$ and $\rho_{\text{rms}}$ by layer & matrix role** (specifically which factors hit $r_C = 0.965\text{–}0.975$, their eigenvalue spectra $\lambda_1/\lambda_n$, and how often the Frobenius cap $a < 1$ fires at ages $50, 200, 2000, 9990$).
3. **Log per-step max coordinate update $\|M_f/(\sqrt{V_f}+\epsilon)\|_\infty$ and cosine similarity of $U_t$ against an on-the-fly shadow `QR10` step** right before ($t \equiv 49 \pmod{50}$) and right after ($t \equiv 51 \pmod{50}$) QR repair, testing the $M/\sqrt{V}$ transport-spike hypothesis directly.
4. **Run B1, B2, B3, and A4 in the §2.6 (i) open-loop replay harness** reporting $\rho_{\text{rms}}$ (with floors $\epsilon_C \in \{10^{-8}, 10^{-12}\}$) and preconditioner distance to `QR10`, checking correlation between replay $\rho_{\text{rms}}$ and §2.3's known ordering (`QR10 < QR20 < warm10/QR50 < QR50`).
5. **Run a 2,000-call training ablation of B5 (`QR10` for $t \le 300$, then unchanged `warm10/QR50`)** on seed 204 to measure directly how much of the steady-state $+0.008\text{–}0.011$ plateau is locked-in early peak-LR lag vs ongoing refresh error.

---

# Document 6: stage-2 reveal

# Stage 2 reveal: pooled candidates, moderator checks, and new measurements

In this document, "you" means your own stage-1 answer, and Reviewers X, Y and Z are the other three. Candidate IDs in
parentheses are each answer's own IDs; for example, "Reviewer X (A5)" is candidate A5 in Reviewer X's answer.

## R1. Pooled candidates

The moderator merged the 50 stage-1 candidates into 14 speed candidates (PA) and 12 tracking candidates (PB). The
"Claimed" column gives the proposers' own estimates. **[checked]** marks a claim the moderator tested (see R3).

### A: speed

| ID | Candidate | Proposed by | Claimed cost / benefit |
|---|---|---|---|
| PA1 | **Stabilized Cholesky-QR for QR events.** Z = C·Q (BF16 operands, FP32 accumulation); optional ridge Z + τ·d_max·Q; column equilibration; CholeskyQR2 with IEEE-FP32 Gram products; `cholesky_ex` with an in-graph fallback | Reviewer X (A1), you (A5), Reviewer Z (A4), Reviewer Y (B4, conditional) | QR event 594 → about 95–140 ms; −9 to −10 ms/call |
| PA2 | **Concurrent QR** over CUDA streams or host threads; same arithmetic | you (A1), Reviewer Z (A3), Reviewer X (A4) | 2.5–4× at n = 768 |
| PA3 | **R-only Householder plus a triangular solve,** then one Newton–Schulz (NS) step | Reviewer X (A2) | about −40% of QR time if orgqr is half of it |
| PA4 | **Custom batched, blocked Householder QR** | Reviewer Y (A3) | target −50% of the QR increment |
| PA5 | **`mm32` (BF16 tensor cores) for Z = C·Q** inside the QR path | you (A1), Reviewer Z (A1) | 4–6 ms per QR event |
| PA6 | **Fewer warm products:** (a) one NS correction (5 products); (b) seed I + A + ½A², then one NS (6); (c) degree-4 Taylor R₄(A), no NS (5); (d) R₄(A/2)² (6) | (a) Reviewer Y (A4), Reviewer Z (A5); (b) Reviewer X (A3), Reviewer Z (B2); (c) you (A6); (d) you (B1) | −14% to −29% of warm FLOPs |
| PA7 | **Cheaper M transport:** (a) M⁺ = R_LᵀMR_R with the known rotation; (b) short-side reassociation | (a) you (A6); (b) Reviewer Y (A5), Reviewer Z (A6) | transport ≈ 0.5 ms/call in total |
| PA8 | **Symmetric (syrk) kernels** for GGᵀ, GᵀG and the NS Grams | Reviewer Y (A2), you (A4), Reviewer Z (A2) | −1 to −1.8 ms/call |
| PA9 | **Contiguous parameter/grad banks** (no stack or copy); coalesce the five n = 768 groups into batch 60; CUDA graphs | Reviewer Y (A1), you (A2), Reviewer Z (A1), Reviewer X (A5) | 0.8–1.1 ms/call + 2.5–5 ms per warm event |
| PA10 | **Fused epilogues and no FP32 temporaries in the warm step;** sub-batching at large n | Reviewer Z (A1), you (A3) | recovers the batch-12 vs batch-1 regression at n ≥ 6144 |
| PA11 | **Strided covariance EMA** (update C every 2nd call with β_C²) | Reviewer X (A6), you (A7), Reviewer Z (A6) | 1.2–1.8 ms/call; not testable in replay |
| PA12 | **Overlap the refresh with the next forward/backward** | Reviewer Z (A3), Reviewer X (A4 variant) | exposed optimizer time 26 → about 12 ms/call |
| PA13 | **Size-aware schedule:** at large n, drop warm and run PA1 every 10–20 ages | you (A8, B6a) | n = 8192: about 325 vs 588 ms per 100 calls |
| PA14 | **FP16 basis storage** (optionally FP16-accumulate back-projection) | you (A9) | Gram floor 0.0024 → about 0.0003 |

### B: tracking

| ID | Candidate | Proposed by | Claimed cost / benefit |
|---|---|---|---|
| PB1 | **QR-tangent generator.** A_ij = B_ij·sgn(d_j − d_i)/max(d_i, d_j, ρ·s), the first-order rotation of one QR power step; optional sort plus V permutation | Reviewer Y (B1), Reviewer X (B3), Reviewer Z (B1) | same product count; matches QR to first order |
| PB2 | **Spectral (or row-ℓ₂) cap instead of the Frobenius cap** | Reviewer X (B2), Reviewer Z (B1), you (B1), Reviewer Y (B1 ablation) | about doubles the applied step |
| PB3 | **Full-gain Jacobi variants:** (a) relative-gap atan Jacobi + row cap + R₄; (b) two inner iterations; (c) local-scale full-gain Jacobi + seed I + A + ½A² + 1 NS | (a, b) you (B1, B2); (c) Reviewer Z (B2) | claimed near-QR or better single-refresh r_C |
| PB4 | **Exact QR at every refresh via PA1** (no warm step) | Reviewer X (B1), you (B6b), Reviewer Z (B3) | 9–13 ms/call of refresh, against 15.9 today; QR10 by construction |
| PB5 | **Head-block or panel power step** on the top k columns, with an orthogonal completion of the tail | Reviewer Y (B2), Reviewer X (B4), you (B3) | +3k/n of a product; exact on the top k |
| PB6 | **Grouped block QR rotations** | Reviewer Y (B3) | one cubic product |
| PB7 | **Front-loaded schedule:** QR10 (or PB4) for the first 250–1000 ages | Reviewer Y (A6), Reviewer X (B5), you (B4), Reviewer Z (B5) | 0 to +1.3 ms/call; aimed at the call-500 gap |
| PB8 | **V after a refresh:** (a) mix V by T∘T; (b) rescale V's rows and columns to the new diagonal d⁺/d | (a) Reviewer Y (B5), Reviewer Z (B4); (b) you (B5), Reviewer X (B7) | fewer post-refresh step errors; training run only |
| PB9 | **Sort columns at warm refreshes** and permute V | you, Reviewer Z, Reviewer Y (all inside their B1) | about 0.4 ms per event |
| PB10 | **Refresh before the update** | you (pitfall P4), Reviewer Z (B4) | fresh basis on 10% of steps; a different method |
| PB11 | **Basis storage precision:** FP16 Q, or an FP32 master / stochastic rounding | you (A9), Reviewer X (B6) | Gram floor ÷8 (FP16) |
| PB12 | **Adaptive QR trigger** (e.g. when the cap scale a < 0.5) | Reviewer X (B5 option) | negligible cost; breaks static graphs |

## R2. Open disputes

- **D1. Target generator.** Match QR's first-order rotation (Reviewer Y, Reviewer X, Reviewer Z), or apply full-gain Jacobi
  (you; Reviewer Z's B2 as a baseline)?
- **D2. V after a refresh.** Mix V by T∘T (Reviewer Y, Reviewer Z), rescale V to the new diagonal (you, Reviewer X), or
  leave it (Reviewer X argues this is correct to first order)?
- **D3. Why the measured Cholesky-QR failed.** κ(C)²·u_FP32 ≳ 1 (Reviewer X, Reviewer Z), or TF32 truncation of the Gram
  product (you)?
- **D4. Size of the engineering win.** About 1–2 ms/call of slack, since the ordinary step is mostly matmul-bound
  (Reviewer X), or 26 → 12 ms/call exposed via overlap (Reviewer Z)?
- **D5. Loss predictions.** "High confidence" that PB1 or PB3 closes 60–80% of the steady-state gap (you,
  Reviewer Z), or no defensible CE prediction (Reviewer Y)?
- **D6. Is leaving V unchanged at a warm step acceptable?**

## R3. Moderator checks

All checks ran on CPU (float64, or float32 where stated), on **synthetic** inputs: full-rank PSD C, power-law
λ_i ∝ 1/i, κ ≈ n. They are not trained covariances; R5 shows how trained factors differ.

**M1. Algebra.**

| Claim | Result |
|---|---|
| QR first-order tangent K_ij = E_ij/d_j for i > j (d_j ≥ d_i) | confirmed: relative error against the exact qr(D + E).Q is 2e-3 (n = 8) and 2e-2 (n = 64) for off-diagonal entries of about 1e-3, shrinking linearly |
| The inverse-gap (Jacobi) generator as an approximation of QR | relative error 4.2–140 |
| One NS step maps E ↦ −¾E² + ¼E³ (0.25 → −0.04297 → −1.40e-3) | confirmed |
| (I + A + ½A²)ᵀ(I + A + ½A²) = I + ¼A⁴ | confirmed |
| R₄ᵀR₄ = I + A⁶/72 + A⁸/576 | confirmed |
| polar(I + A) rotates by arctan θ | confirmed |
| The top-k Procrustes reflection is orthogonal and maps Q₁ ↦ −Ũ₁ | confirmed |
| The closed-form warm/QR angle ratio | confirmed |
| ‖K‖₂ ≤ ‖K‖_F/√2 for skew K | confirmed |
| Transport reassociation saves 8.3% / 0% / 7.5% / 8.3% of multiply-adds on the four 109M shapes | confirmed |
| 8T_warm + 2T_QR = 588 vs 5T_QR = 560 ms per 100 calls at n = 8192 | confirmed |
| `set_float32_matmul_precision('high')` sets `allow_tf32 = True` (PyTorch 2.11) | confirmed |
| The reference SOAP applies (W − ηcU)(1 − ηλ) with c_t·m/(√v + ε), the same form as the variant | confirmed; the update-ordering concern in Reviewer Y's pitfalls is moot |

**M2. The cap and the half step.** Synthetic inputs as in the microbenchmark, exact polar retraction.

| n = 256, input r_C 0.22 | single-refresh r_C |
|---|---|
| production (½ step, Frobenius cap 0.5) | 0.175 |
| ½ step, no cap | 0.118 |
| factor 0.75, no cap | 0.075 |
| factor 1, no cap | 0.055 |
| simultaneous exact 2×2 Jacobi angles | 0.131 |

- Production in BF16 equals its float64 analogue to three digits.
- Pre-cap ‖A‖_F/√2 is 0.63–1.10 for n = 128–1024, with no clear trend in n. The cap scale a is 0.46–0.79, and
  ‖A‖₂ ≤ 0.30.
- **This contradicts in magnitude** the "√n too tight" claim (Reviewer X, I4) and "explodes by √n" (you). The
  measured Frobenius-to-spectral ratio is 2.5–5. The conclusion that the cap binds stands.

**M3. Generators, one-shot and chained.**

One-shot, n = 256, input r_C 0.217 and ρ_rms 0.489. ρ_rms is the RMS whitened off-diagonal correlation.

| generator | r_C | ρ_rms | top-k subspace distance to QR |
|---|---|---|---|
| production | 0.168 | 0.400 | 0.015 |
| ½ step, no cap | 0.115 | 0.331 | 0.006 |
| first-order Jacobi, factor 1 | 0.052 | 0.278 | 0.004 |
| QR tangent (PB1, ρ = 1e-4, no cap) | 0.081 | 0.260 | 0.002 |
| relative-gap atan Jacobi + row cap (PB3a, as specified) | 0.121 | 0.341 | 0.012 |
| QR | 0.038 | 0.212 | 0 |

At n = 512 the QR tangent gives r_C 0.099. At large misalignment (input r_C 0.45), the QR tangent's one-shot ρ_rms is
0.676, against 0.485 for the uncapped ½ step. A first-order generator therefore still needs a bound.

Chained open-loop toy: n = 128, λ ∝ 1/i, skew drift 0.01√2 per step, 32 samples per step, β_C = 0.95, refresh every
10, QR every 50. Means over steps 100–1000:

| schedule | r_C | ρ_rms |
|---|---|---|
| QR10 | 0.0759 | 0.240 |
| QR tangent, 10 / QR50 | 0.0767 | 0.243 |
| Jacobi factor 1, 10 / QR50 | 0.0742 | 0.253 |
| PB3a, 10 / QR50 | 0.0785 | 0.254 |
| ½ step, no cap, 10 / QR50 | 0.0841 | 0.271 |
| production | 0.0977 | 0.294 |
| QR50 | 0.1343 | 0.353 |

Sorting columns made no difference to the QR tangent.

**These contradict two expected-effect claims:**
- "Single-refresh r_C 0.150 → 0.03–0.05" for PB3a (you, B1): measured 0.12.
- "0.150 → 0.035–0.050" for the QR tangent (Reviewer Z, B1): measured 0.08–0.10. Its advantage appears in the chained
  and whitened metrics.

The prediction 0.20 → 0.10–0.11 for removing the Frobenius cap (Reviewer X, B2) matches the measured 0.115–0.118.

**M4. Cholesky-QR numerics.** float32 CPU; TF32 emulated by rounding the Gram operands to a 10-bit mantissa.
Synthetic C with κ = 8e2 to 7.2e3 at n = 512–4096.

| variant | Gram error after BF16 storage | max column distance to Householder |
|---|---|---|
| unshifted one pass, FP32 | 0.0023–0.0026 | up to 2.8e-4 |
| unshifted one pass, TF32 | 0.0027–0.0043 | up to 3.8e-4 |
| CholeskyQR2, FP32, no ridge | 0.0023–0.0024 | ≤ 4.2e-7 |
| ridge 1e-4·d_max + CholeskyQR2 | 0.0024 | 8e-5 (n = 512) to 3.1e-3 (n = 4096) |
| ridge 1e-3·d_max + CholeskyQR2 | 0.0024 | 3.9e-3 to 2.6e-2 |

- No failures at n ≤ 4096.
- **This contradicts** the claim that TF32 "directly explains" the measured 0.034 Gram error at n = 4096 (you,
  pitfall P2): emulated TF32 gives 0.0043. n ≥ 6144 was not run.
- **But see R5:** on trained covariances, unshifted Cholesky-QR fails almost always. The no-ridge result above holds
  only for full-rank synthetic C.

**M5. V after a refresh** (2×2 cases).

| case | true V₂ | V unchanged | T∘T mixing | rescale to d⁺/d |
|---|---|---|---|---|
| you's case: C = U·diag(1e4, 1e-2)·Uᵀ, old basis 0.01 rad off, refresh corrects it | 0.01 | 1.01 | 2.01 | 0.01 (exact) |
| Reviewer Z's case, with M and V stale relative to C | — | spike: M/√V = 198 | M/√V ≈ 1 | spike: M/√V = 3.9e4 (the ratio uses a C that V never saw) |
| same rotation, with M and V consistent with C (as β₁ = β₂ = β_C makes natural) | 1e-6 | 0.0395 | 0.0758 | 1e-6 (exact); no spike under any rule |

**M6. BF16 covariance EMA dead-band.** A noiseless slow drift leaves the BF16 EMA 4.2% away from the FP32 EMA. With
sample noise the gap is 0.6–0.8%.

**M7. Basis-frame covariance** (B = QᵀCQ kept as an EMA of ḠḠᵀ). The identity holds: the off-diagonal error is
1.8e-3 with a BF16 Q_R. But B must be rotated at every basis change (2 products), so the warm step keeps 7 products.
No reviewer proposed this; it is listed for completeness.

## R4. Part 1 microbenchmark: roofline view

Compiled, per factor, from the stage-1 cost table (RTX 5090).

**Warm refresh at batch 12,** counting 14n³ FLOPs (7 products):

| n | 768 | 1024 | 1536–2048 | 2304–8192 |
|---|---|---|---|---|
| TFLOP/s | 92 | 147 | 153–164 | 165–174 |

At batch 1 it reaches 178 at n = 3072 and 207–213 at n = 6144–8192. **From n ≈ 2048 up, the warm step runs at the
BF16 tensor-core limit, so only fewer products help it at large n.**

**Production QR,** counting (2 + 8/3)n³:

| n | 768 | 1024 | 2048 | 3072 | 4096 | 8192 |
|---|---|---|---|---|---|---|
| TFLOP/s | 0.8 | 1.3 | 4.1 | 7.5 | 11.6 | 22.9 |

**Warm/QR time per factor:**

| n | 768–1024 | 2048 | 3072 | 4096 | 6144 | 8192 |
|---|---|---|---|---|---|---|
| warm/QR | 0.027 | 0.075 | 0.136 | 0.20 | 0.30 | 0.41 |

**Planned, not run.** A measured extension to n = 16384 with a shifted CholeskyQR3 is queued (Part 1b). No results
exist yet.

## R5. Part 2: open-loop replay on trained covariances

### Part 2 report (verbatim except paths and file names removed)

**Setup (approved config).**
- **Models.** 1-layer Transformers of the same architecture family: fused QKV,
  fused SwiGLU with hidden size 2d, tied embeddings, RoPE, QK-norm, no bias,
  heads = d/64. Widths d ∈ {256, 512, 1024, 2048}. Each width has 8 factors of
  dimensions d×5, 2d, 3d and 4d, the 109M pattern scaled.
- **Driver.** The FP32 SOAP reference with QR10 and the stock betas
  (.95/.95/.95). Matrix LR 1.5e-3 (not
  re-tuned per width), AdamW auxiliary LR 6e-4, weight decay .05, gradient clip
  1.0. Warmup 100, then cosine to 0.1× over 1,001 calls.
- **Data.** 32 rows × 1024 tokens per step from the cached OpenWebText. This is the same
  dataset revision and tokenizer as the paper but a different set of rows. Seed 204.
- **Training outcome.** Validation CE at age 1000: 5.40 / 5.08 / 4.94 / 4.87 for
  d = 256 / 512 / 1024 / 2048. Each run took 1.7–7.3 min on its own GPU.
- **Saved data.** C_L and C_R of every matrix at age 0 and every 10th age
  (101 snapshots), in BF16.

**Age alignment.** For d=256, FP32 C and the driver's Q were saved as well. An
offline chain of the reference's QR refresh over the saved FP32 trajectory
reproduces the driver's Q **bit-exactly at all 100 refresh ages for all 8
factors**, so the snapshots are exactly the covariances each refresh saw.

**Replay.** Each schedule starts from production's eigh initialization at age 0
and is chained over the saved BF16 trajectory using the production `warm_basis`,
`qr_basis` and eigh initialization path, run eagerly. Schedules: QR10,
warm10/QR50, QR20, QR50, eigh10, Cholesky-QR10, init-only, plus the ablation
chains.

The parametrized default warm chain stays bit-identical to production at every
age, for every width and factor group.

**Metrics.** All are evaluated at FP32 "highest" precision and averaged
unweighted over a width's 8 factors.
- **r_C:** the paper's definition. *After* means immediately after the refresh
  at that age; *before* means the previous basis on the covariance at that age.
  *Interval* is mean(after_t, before_{t+10}).
- **r_C^w:** r_C of D^-½ B D^-½ with D = diag(B), restricted to coordinates with
  B_jj ≥ 1e-4·max B_jj, so every direction counts equally regardless of its variance.
- **Top-k overlap:** ‖U_kᵀ Q_S‖²_F / k. Here U_k holds the top-k eigenvectors of
  C and Q_S the k stored columns with the largest Rayleigh quotients, with
  k ∈ {1, 8, n/8}.
- **e_Q:** the normalized Gram error ‖QᵀQ − I‖_F / √n.

All results are **open-loop**: the covariances come from a QR10-driven run and
do not respond to the replayed bases.

#### Replay quality (late window, ages 500–1000)

| d | schedule | r_C after | r_C before | r_C interval | r_C^w interval | top-1 after | top-n/8 after | e_Q after |
|---|---|---|---|---|---|---|---|---|
| 256 | QR10 | 0.139 | 0.274 | 0.206 | 0.461 | 0.923 | 0.922 | 2.36e-3 |
| 256 | warm10/QR50 | 0.262 | 0.318 | 0.290 | 0.622 | 0.845 | 0.841 | 1.91e-3 |
| 256 | QR20 | 0.211 | 0.299 | 0.255 | 0.573 | 0.889 | 0.882 | 2.36e-3 |
| 256 | QR50 | 0.292 | 0.336 | 0.315 | 0.660 | 0.834 | 0.823 | 2.36e-3 |
| 256 | eigh10 | 0.002 | 0.263 | 0.133 | 0.332 | 1.000 | 1.000 | 2.36e-3 |
| 512 | QR10 | 0.176 | 0.316 | 0.246 | 0.513 | 0.868 | 0.913 | 2.35e-3 |
| 512 | warm10/QR50 | 0.308 | 0.364 | 0.336 | 0.701 | 0.778 | 0.815 | 1.88e-3 |
| 512 | QR20 | 0.253 | 0.342 | 0.298 | 0.638 | 0.812 | 0.869 | 2.35e-3 |
| 512 | QR50 | 0.330 | 0.378 | 0.355 | 0.734 | 0.778 | 0.804 | 2.35e-3 |
| 512 | eigh10 | 0.002 | 0.298 | 0.150 | 0.371 | 1.000 | 1.000 | 2.36e-3 |
| 1024 | QR10 | 0.204 | 0.346 | 0.275 | 0.536 | 0.830 | 0.902 | 2.35e-3 |
| 1024 | warm10/QR50 | 0.341 | 0.397 | 0.370 | 0.746 | 0.736 | 0.794 | 1.87e-3 |
| 1024 | QR20 | 0.281 | 0.373 | 0.327 | 0.676 | 0.778 | 0.855 | 2.35e-3 |
| 1024 | QR50 | 0.364 | 0.412 | 0.389 | 0.779 | 0.730 | 0.787 | 2.35e-3 |
| 1024 | eigh10 | 0.002 | 0.324 | 0.163 | 0.392 | 1.000 | 1.000 | 2.35e-3 |
| 2048 | QR10 | 0.240 | 0.389 | 0.314 | 0.588 | 0.777 | 0.899 | 2.35e-3 |
| 2048 | warm10/QR50 | 0.389 | 0.448 | 0.419 | 0.823 | 0.668 | 0.781 | 1.85e-3 |
| 2048 | QR20 | 0.324 | 0.418 | 0.371 | 0.741 | 0.718 | 0.849 | 2.35e-3 |
| 2048 | QR50 | 0.412 | 0.462 | 0.437 | 0.856 | 0.663 | 0.777 | 2.35e-3 |
| 2048 | eigh10 | 0.002 | 0.359 | 0.180 | 0.433 | 1.000 | 1.000 | 2.35e-3 |

Two schedules are left out of this table:
- **Init-only:** r_C ≈ 0.92–0.95 at every width.
- **Cholesky-QR10:** **non-finite from the first refresh (age 10)**. That is
  800/800 factor-refresh events at d ≥ 512 and 791/800 at d = 256. Unshifted
  Cholesky-QR in FP32 cannot factor Z = CQ for these factors. Many directions
  are near-null, and some BF16-stored factors are slightly indefinite (4–15%
  negative eigenvalues in the right-side SwiGLU factor at d ≥ 512).

**Build-up across the four warm refreshes (warm10/QR50, late window; before / after each refresh):**

| d | 0 after QR | 10 | 20 | 30 | 40 | 50 before QR | QR10, phase-40 after |
|---|---|---|---|---|---|---|---|
| 256 | 0.153 | 0.291 / 0.269 | 0.308 / 0.287 | 0.322 / 0.301 | 0.330 / **0.309** | 0.337 | **0.141** |
| 512 | 0.187 | 0.321 / 0.303 | 0.357 / 0.341 | 0.372 / 0.356 | 0.380 / **0.364** | 0.386 | **0.176** |
| 1024 | 0.214 | 0.342 / 0.326 | 0.400 / 0.384 | 0.410 / 0.395 | 0.415 / **0.400** | 0.417 | **0.204** |
| 2048 | 0.259 | 0.398 / 0.381 | 0.455 / 0.439 | 0.459 / 0.444 | 0.452 / **0.436** | 0.472 | **0.220** |

Key observations (open-loop, single seed):

1. **The paper's §5.5 cell is reproduced.** Phase 40, just after the fourth
   warm refresh, gives warm10/QR50 r_C = 0.31 / 0.36 / 0.40 / 0.44 against
   QR10 0.14 / 0.18 / 0.20 / 0.22 for d = 256 → 2048. The 109M run gave
   0.38–0.39 against 0.20–0.21, which falls between our d = 512 and d = 2048.
   r_C grows with width for every schedule.
2. **A warm refresh removes only 3–8% of the current r_C** (e.g. 0.291 → 0.269).
   A QR refresh removes 38–49% (QR10: 0.274 → 0.139 at d = 256). The
   10-age drift is larger than either. Even an exact refresh every 10 ages
   (eigh10) rises from 0.002 back to 0.26–0.36 before the next refresh.
3. **Warm refreshes close only 15–23% of the QR50 → QR10 gap** in the interval
   mean of r_C: 23% / 17% / 17% / 15% for d = 256 / 512 / 1024 / 2048. They
   close 12–19% of it in r_C^w. Top-1 alignment after refresh is 0.67–0.85,
   against 0.66–0.83 for QR50 and 0.78–0.92 for QR10.
4. **The weak directions barely move.** r_C^w for warm10/QR50 (0.62–0.82) stays
   near QR50 (0.66–0.86), against 0.46–0.59 for QR10. The likely mechanism is
   the gap normalization h_ij = (d_i − d_j)/max_k d_k. At age 750, 40–99% of
   each factor's eigen-directions have λ < 1% of λ_max, and those directions
   carry 4–41% of the variance. Every pair of such directions has |h| < 0.01,
   so the damping suppresses it.
5. **The large left factors dominate the factor mean.** The 3d and 4d left
   factors (QKV and SwiGLU-in) reach r_C 0.64–0.80 under warm10/QR50, against
   0.35–0.54 for QR10. The d and 2d factors sit at 0.13–0.29 against 0.06–0.17.
6. **QR20 beats warm10/QR50 at every width** (interval r_C 0.255–0.371 against
   0.290–0.419), but at about 1.9× the refresh cost. With the compiled 109M
   costs from Part 1, refresh work per 50 ages is:

   | schedule | ms per 50 ages |
   |---|---|
   | QR50 | 594 |
   | warm10/QR50 | ≈ 793 |
   | QR20 | 1,484 |
   | QR10 | 2,969 |

   Per millisecond of refresh work, the warm refreshes buy about 1.3–1.9× the
   r_C reduction that QR20's extra QRs buy. This is an r_C statement, not a
   loss statement.
7. **Orthogonality is fine.** The warm basis's e_Q (1.85–1.91e-3) is slightly
   *lower* than the BF16-rounded QR baseline (2.35e-3).

#### Warm-step ablation (one knob at a time inside warm10/QR50; late window)

Timing is per batch-12 call on 768-dim factors, compiled, from Part 1 on the
exclusive GPU. Default: 0.82 ms. The full table follows this section.

| change from default (0.01 / 0.5 / 2 / 1) | Δ r_C interval (d=256…2048) | Δ r_C^w interval | e_Q after | time (ms) |
|---|---|---|---|---|
| damping 0.001 | +0.016 / +0.012 / +0.014 / +0.014 | +0.006 / −0.002 / 0.000 / +0.001 | ≈ same | 0.83 |
| damping 0.003 | +0.009 / +0.007 / +0.008 / +0.010 | 0.000 / −0.006 / −0.004 / −0.002 | ≈ same | 0.83 |
| damping 0.03 | −0.010 / −0.008 / −0.008 / −0.009 | +0.007 / +0.012 / +0.012 / +0.013 | ≈ same | 0.83 |
| damping 0.1 | −0.016 / −0.012 / −0.012 / −0.014 | +0.020 / +0.022 / +0.023 / +0.026 | ≈ same | 0.82 |
| cap 0.25 | +0.009 / +0.006 / +0.007 / +0.008 | +0.013 / +0.010 / +0.013 / +0.016 | ≈ same | 0.82 |
| cap 1.0 | −0.011 / −0.009 / −0.009 / −0.010 | −0.015 / −0.011 / −0.013 / −0.018 | 1.9–2.0e-3 (+2–10%) | 0.83 |
| **no cap** | **diverges**: non-finite in 59–82% of factor-ages, first at age 30–40 | — | — | 0.78 |
| 0 Newton–Schulz corrections | +0.002 / +0.001 / +0.003 / +0.006 | ≈ +0.002 | **6.5e-3–1.0e-2 (3.5–5×)** | 0.46 |
| 1 correction | 0.000 | 0.000 | 2.1–2.2e-3 | 0.64 |
| 3 corrections | 0.000 | 0.000 | 1.76–1.81e-3 | 1.01 |
| 2 warm steps per refresh | −0.014 / −0.011 / −0.011 / −0.012 | −0.019 / −0.016 / −0.018 / −0.021 | ≈ same | 1.50 |

**Are 0.01 / 0.5 / 2 reasonable?** Yes, as points in this family. None of the
changes alters the picture: the best single change moves the interval r_C by
at most 0.016, while the QR50–QR10 gap is 0.11–0.12.
- **Corrections: 2 is right.** Zero corrections leave 3.5–5× the Gram error.
  One correction is already near the BF16 QR baseline. A third buys about 5%
  lower e_Q for 0.18 ms per call.
- **The cap is load-bearing.** Without it the step diverges. A cap of 1.0 was
  slightly better than 0.5 on this replay, at the cost of a little more Gram
  error.
- **Damping trades the two metrics against each other.** Larger δ lowers the
  Frobenius r_C, probably because the cap binds less often, but worsens the
  weak-direction metric r_C^w. Smaller δ is worse on r_C. So 0.01 is a sensible
  middle.
- **Two warm steps** give the largest gain on both metrics, for 1.8× the warm
  cost.

The limit is structural, not a tuning problem: the 10-age drift and the
max-normalized gap leave weak directions unrotated.

#### Caveats

- **Open loop.** The covariances come from a QR10-driven trajectory, so closed
  loop TurboSOAP would see different C. The results describe tracking quality
  on a fixed trajectory, not loss.
- **Scale.** Single seed, 1-layer models, 1,000 steps (11 of 101 snapshots
  fall in LR warmup), and LR not re-tuned per width.
- **Data.** OWT rows differ from the paper's frozen cohort.
- **Covariance precision.** Saved C is the reference's FP32 running average
  rounded once to BF16. Production accumulates the average in BF16.
- **Execution mode.** The replay runs the production functions eagerly; the
  compiled production function differs at the rounding level (Part 1).
- **Metric definitions.** r_C^w depends on the 1e-4 coordinate floor. Factor
  means are unweighted.
- **Cholesky-QR.** Its failure here is specific to unshifted FP32 Cholesky on
  BF16-stored, near-singular factors.

#### Full ablation table (late window; warm10/QR50 chains; "_qr10"/"_qr50" are reference rows)

| d | variant | r_C after warm | r_C pre-QR | r_C interval | r_C^w interval | e_Q after | ms (n=768 b12, compiled) |
|---|---|---|---|---|---|---|---|
| 256 | _qr10 | 0.139 | 0.274 | 0.206 | 0.461 | 2.36e-03 | — |
| 256 | _qr50 | 0.328 | 0.366 | 0.315 | 0.660 | 2.36e-03 | — |
| 256 | default | 0.291 | 0.337 | 0.290 | 0.622 | 1.91e-03 | 0.82 |
| 256 | damping=0.001 | 0.316 | 0.355 | 0.306 | 0.628 | 1.92e-03 | 0.83 |
| 256 | damping=0.003 | 0.306 | 0.347 | 0.300 | 0.622 | 1.92e-03 | 0.83 |
| 256 | damping=0.03 | 0.276 | 0.327 | 0.281 | 0.629 | 1.89e-03 | 0.83 |
| 256 | damping=0.1 | 0.266 | 0.320 | 0.274 | 0.642 | 1.87e-03 | 0.82 |
| 256 | cap=0.25 | 0.305 | 0.346 | 0.299 | 0.636 | 1.91e-03 | 0.82 |
| 256 | cap=1.0 | 0.273 | 0.325 | 0.279 | 0.607 | 2.04e-03 | 0.83 |
| 256 | cap=None (non-finite 60%, first at age 40) | 0.040 | 0.052 | 0.049 | 0.433 | — | 0.78 |
| 256 | ns=0 | 0.295 | 0.338 | 0.293 | 0.624 | 1.00e-02 | 0.46 |
| 256 | ns=1 | 0.291 | 0.336 | 0.290 | 0.622 | 2.23e-03 | 0.64 |
| 256 | ns=3 | 0.292 | 0.337 | 0.291 | 0.622 | 1.81e-03 | 1.01 |
| 256 | warm_steps=2 | 0.268 | 0.323 | 0.276 | 0.603 | 1.93e-03 | 1.50 |
| 512 | _qr10 | 0.177 | 0.319 | 0.246 | 0.513 | 2.35e-03 | — |
| 512 | _qr50 | 0.369 | 0.411 | 0.355 | 0.734 | 2.35e-03 | — |
| 512 | default | 0.341 | 0.386 | 0.336 | 0.701 | 1.88e-03 | 0.82 |
| 512 | damping=0.001 | 0.360 | 0.402 | 0.349 | 0.699 | 1.90e-03 | 0.83 |
| 512 | damping=0.003 | 0.352 | 0.395 | 0.344 | 0.695 | 1.89e-03 | 0.83 |
| 512 | damping=0.03 | 0.328 | 0.378 | 0.328 | 0.713 | 1.86e-03 | 0.83 |
| 512 | damping=0.1 | 0.322 | 0.374 | 0.324 | 0.724 | 1.83e-03 | 0.82 |
| 512 | cap=0.25 | 0.350 | 0.393 | 0.342 | 0.712 | 1.89e-03 | 0.82 |
| 512 | cap=1.0 | 0.326 | 0.377 | 0.327 | 0.690 | 1.92e-03 | 0.83 |
| 512 | cap=None (non-finite 59%, first at age 30) | 0.028 | 0.045 | 0.036 | 0.544 | 1.91e-03 | 0.78 |
| 512 | ns=0 | 0.343 | 0.388 | 0.338 | 0.704 | 6.92e-03 | 0.46 |
| 512 | ns=1 | 0.341 | 0.386 | 0.336 | 0.701 | 2.17e-03 | 0.64 |
| 512 | ns=3 | 0.341 | 0.386 | 0.336 | 0.702 | 1.79e-03 | 1.01 |
| 512 | warm_steps=2 | 0.323 | 0.376 | 0.325 | 0.685 | 1.91e-03 | 1.50 |
| 1024 | _qr10 | 0.205 | 0.342 | 0.275 | 0.536 | 2.35e-03 | — |
| 1024 | _qr50 | 0.403 | 0.443 | 0.389 | 0.779 | 2.35e-03 | — |
| 1024 | default | 0.376 | 0.417 | 0.370 | 0.746 | 1.87e-03 | 0.82 |
| 1024 | damping=0.001 | 0.396 | 0.435 | 0.384 | 0.746 | 1.88e-03 | 0.83 |
| 1024 | damping=0.003 | 0.389 | 0.428 | 0.378 | 0.742 | 1.88e-03 | 0.83 |
| 1024 | damping=0.03 | 0.364 | 0.408 | 0.362 | 0.758 | 1.85e-03 | 0.83 |
| 1024 | damping=0.1 | 0.357 | 0.405 | 0.358 | 0.769 | 1.82e-03 | 0.82 |
| 1024 | cap=0.25 | 0.387 | 0.426 | 0.377 | 0.759 | 1.87e-03 | 0.82 |
| 1024 | cap=1.0 | 0.362 | 0.408 | 0.361 | 0.733 | 1.97e-03 | 0.83 |
| 1024 | cap=None (non-finite 59%, first at age 30) | 0.042 | 0.060 | 0.052 | 0.584 | 3.35e-03 | 0.78 |
| 1024 | ns=0 | 0.381 | 0.421 | 0.373 | 0.748 | 6.52e-03 | 0.46 |
| 1024 | ns=1 | 0.376 | 0.417 | 0.370 | 0.746 | 2.14e-03 | 0.64 |
| 1024 | ns=3 | 0.376 | 0.417 | 0.370 | 0.746 | 1.78e-03 | 1.01 |
| 1024 | warm_steps=2 | 0.359 | 0.407 | 0.359 | 0.728 | 1.90e-03 | 1.50 |
| 2048 | _qr10 | 0.240 | 0.393 | 0.314 | 0.588 | 2.35e-03 | — |
| 2048 | _qr50 | 0.452 | 0.496 | 0.437 | 0.856 | 2.35e-03 | — |
| 2048 | default | 0.425 | 0.472 | 0.419 | 0.823 | 1.85e-03 | 0.82 |
| 2048 | damping=0.001 | 0.447 | 0.490 | 0.433 | 0.824 | 1.86e-03 | 0.83 |
| 2048 | damping=0.003 | 0.440 | 0.484 | 0.429 | 0.821 | 1.85e-03 | 0.83 |
| 2048 | damping=0.03 | 0.412 | 0.463 | 0.410 | 0.835 | 1.83e-03 | 0.83 |
| 2048 | damping=0.1 | 0.404 | 0.457 | 0.405 | 0.848 | 1.80e-03 | 0.82 |
| 2048 | cap=0.25 | 0.438 | 0.481 | 0.427 | 0.839 | 1.85e-03 | 0.82 |
| 2048 | cap=1.0 | 0.410 | 0.462 | 0.409 | 0.805 | 2.03e-03 | 0.83 |
| 2048 | cap=None (non-finite 82%, first at age 30) | — | — | — | — | — | 0.78 |
| 2048 | ns=0 | 0.435 | 0.479 | 0.425 | 0.826 | 6.71e-03 | 0.46 |
| 2048 | ns=1 | 0.425 | 0.472 | 0.419 | 0.822 | 2.11e-03 | 0.64 |
| 2048 | ns=3 | 0.425 | 0.472 | 0.419 | 0.823 | 1.76e-03 | 1.01 |
| 2048 | warm_steps=2 | 0.406 | 0.460 | 0.407 | 0.801 | 1.87e-03 | 1.50 |

## R6. Tests running in parallel (do not assume their outcome)

The refresh study is replay-testing three generators on the same trained covariances right now:
- PB1, the QR-tangent generator;
- a relative-gap damping variant of the current step;
- first-order Jacobi at factor 1.

Results may arrive after your answer. Your rankings should not assume any particular outcome; say instead what
result would change your ranking.

---

# Stage-2 task

## Stage 2 task

You answered the stage-1 brief (Document 1) independently; your answer is Document 2. You now also have the other
three stage-1 answers (Documents 3–5, anonymized as Reviewers X, Y and Z) and the stage-2 reveal (Document 6). The
reveal holds:
- the pooled candidate list;
- the open disputes;
- the moderator's numerical checks;
- the Part 1 roofline view;
- the Part 2 replay results on trained covariances;
- a note on tests still running.

**Web search is allowed and encouraged in this stage.** Use it to check prior work and implementation facts. Cite
each source with title, authors, year and a link, and say whether you **read** it (opened the paper or page) or saw it
**in search results only**. Mark anything you could not verify as [unverified]. Do not run code. Do not read any file
other than this prompt.

Please answer the following, in this order.

### 1. What changed for you (required)

Which of your own stage-1 claims do the moderator checks (R3) or the Part 2 results (R5) contradict, weaken or
confirm? Be specific: name your candidate IDs and numbers. Name the **single biggest change** to your view.

### 2. Prior work (web)

For each item below, say what the literature already establishes, and whether any pooled candidate is already known,
already shown to fail, or new:
- **Refresh frequency and cost in SOAP/Shampoo:**
  - the 2026 "FOAM" papers: one by Nam & Ahn on adaptive Shampoo refresh frequency, and one by Yu et al. on a cheaper
    SOAP;
  - Milligan et al. (2026), on storing statistics in basis coordinates;
  - any other work on warm-started or approximate eigenbasis tracking, or on adaptive refresh, for SOAP, Shampoo,
    KL-Shampoo or EShampoo.
- **CholeskyQR2 / shifted CholeskyQR3:**
  - shift choice and its effect on the computed Q;
  - behavior on rank-deficient or slightly indefinite inputs;
  - mixed-precision variants.
- **First-order (QR-tangent) and Jacobi-type tracking of eigenbases** under drift, including subspace or orthogonal
  iteration with Rayleigh–Ritz.
- **Batched or concurrent small QR on GPUs** (cuSOLVER, MAGMA): what is achievable at n = 768–3072 and batch 12.
- **Handling Adam's second moment when the basis changes** (PB8).

### 3. Ranking

Rank the pooled **A (speed)** candidates and, separately, the pooled **B (tracking)** candidates. Use every
candidate, or your top 8 per list with the rest marked "not ranked". Give one line of reasoning each. Take the Part 2
facts into account:
- trained factors are near-singular and some are slightly indefinite in BF16;
- unshifted Cholesky-QR fails on them;
- the current generator diverges without its cap, but the factor-ages that stay finite track better than QR10;
- the warm step barely moves weak directions;
- QR20 beats warm10/QR50 per refresh, but not per millisecond.

Where a candidate's rank depends on a test still running (R6), say what outcome would move it.

### 4. Your top 3 experiments

Name the three experiments, from either list, that you would run first. For each give:
1. **Exact spec.** A refresh function as batched PyTorch, at most about 40 lines. Its signature is
   `f(C_bf16[b,n,n], Q_bf16[b,n,n]) -> Q_new_bf16` (optionally also returning a column permutation), or a timing
   target for the microbenchmark. Include every constant (floors, caps, ridges, number of steps).
2. **Where it runs:**
   - the replay harness, which chains your function over the saved trained covariances (101 BF16 snapshots per width,
     d = 256–2048, refresh every 10 ages), with QR repairs at 50 if you say so;
   - the microbenchmark (compiled ms per factor at batch 1 and 12, n = 256–8192, plus the 109M event regions);
   - or, only if neither can test it, the smallest training run that could.
3. **The decision metric and threshold**, for example "late-window r_C^w interval within X of QR10 at all widths, at
   ≤ Y× the warm cost at n = 3072".
4. **Your probability** that it passes.

Use the harness's existing metrics where possible: r_C, r_C^w (diagonally normalized), top-1/8/n-8 overlap, e_Q, and
ms per factor. If you want a new metric, define it exactly.

### 5. Disputes (D1–D6) and your strongest disagreement

- Give your position on each of D1–D6 in one or two lines, updated with the evidence in R3 and R5.
- Then name your **single strongest disagreement with another reviewer** (required). Say which reviewer (X, Y or Z),
  the claim, why you disagree, and what measurement would settle it.

### 6. Errors in the moderator checks

If any check in R3 is wrong, mis-specified, or does not test what it claims, say so and why.

Write for an expert reader, in Markdown. Aim for 2,500–5,000 words. Prefer formulas and numbers to adjectives.
