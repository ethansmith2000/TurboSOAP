# FP32 basis confirmation review

Both phases completed in 57.69 minutes, supervisor exit status 0. Repeated audits
verified 581 source files, input/support hashes, the cached control reproduction,
10,000 candidate calls, evaluation preservation and capture/observation counts.
The compiled candidate retained 12 graphs and zero graph breaks.

At seed 205 and 10,000 calls, stock QR10 ended at CE **3.409272**, full BF16 at
**3.419354**, and BF16 state with FP32 bases at **3.422055**. The respective
gaps versus stock are +0.010082 and +0.012783 CE, or 1.013% and 1.287% higher
perplexity. FP32 bases improve the momentum/Gram diagnostics but do not show a
loss benefit on this seed. Their endpoint is +0.002701 CE above full BF16;
neither endpoint difference establishes a general statistical ranking.

FP32 bases use **1.608 GiB** of matrix state, compared with full BF16 **1.160 GiB**
and stock **2.320 GiB**. Maximum observed physical-M error is **0.5155%**, versus
full BF16 **1.0356%**. Both variants stayed finite. Full BF16's previously
declared 1% screen flag remains false in its frozen record.

The user clarified on 2026-10-27 that loss performance should carry more weight
than an arbitrary engineering error bound. Future method selection treats these
finite numerical errors as diagnostics, with loss, speed, memory and actual
stability driving the decision. This changes the interpretation and future
decision rule; it does not rewrite past protocols, measurements or flags. The
1% screen has not been validated as a boundary for good learning.

FP32 bases reduced step time by **16.73%** in the matched 100-call timing prefix
(379.944 to 316.390 ms). The full BF16 pair reported **18.66%** over its paired
9,900-call window. These timing scopes differ, so they do not support a precise
direct speed ranking. The full FP32-basis candidate mean was 316.567 ms, with
compilation, diagnostics and evaluations excluded from steady step timing.

The next recommended experiment is full BF16 at model seed 204 for 10,000 calls,
reusing the completed stock204 quality control and replaying a short control
prefix on the allocated device. This would provide two full model initializations
for one fixed precision option at about one additional GPU-hour. It is proposed,
not launched. Then complete the cheap refresh controls/profile in the
[demonstration plan](../../DEMONSTRATION_EXPERIMENTS.md).
