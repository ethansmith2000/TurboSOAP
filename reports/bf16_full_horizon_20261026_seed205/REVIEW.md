# Full horizon seed 205 review

Both10k-call arms completed, supervisor exit0. Repeated audits verified579 source
hashes, inputs/support files, all20k calls, evaluation preservation and capture
counts. Measurement completion passed; full BF16 numerical qualification failed.

Stock QR10 ended at CE **3.409272** and full BF16 at **3.419354**: **+.010082 CE**,
or **1.013% higher perplexity**. Mean complete-step time fell380.548->309.533ms
(18.66%); optimizer time95.231->26.077ms. Matrix state halved2.320->1.160GiB.
These describe the original stock implementation and this fixed recipe.

At the first warm refresh, actual-M error exceeded the1% limit for three feed-
forward input matrices:1.0325%,1.0004%,1.0356%. The sampled late maximum was
.5424%, with no observed increasing trend. Original absolute Gram also failed,
as expected for BF16 bases. The failure does not imply divergent training, but
it prevents declaring the option qualified under the frozen criterion. Keep
the threshold and the result unchanged.

Next test the existing BF16 M/V/C, FP32-Q option at seed205 for10k calls. Reuse
the verified full QR10 quality result and replay201 control calls before the
candidate, requiring exact reproduction at0/50/100/200. This saves about65
GPU-minutes while providing a matched100-call prefix timing comparison on the
newly admitted device. The short seed204 pilot already supports testing this
more conservative option; better CE is not assumed.

See [next protocol](../../BF16_HYBRID_CONFIRM_20261026.md) and the
[draft demonstration plan](../../DEMONSTRATION_EXPERIMENTS.md). One LM task,
two model seeds and cheap refresh/profile controls are sufficient for the
proposed narrow demonstration. Keep the full BF16 result exploratory and the
remaining gap versus stock SOAP visible.
