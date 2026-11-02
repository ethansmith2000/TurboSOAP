# Short full BF16 cadence controls

The frozen QR20/QR50 pair completed with supervisor exit0 in1,459.47seconds,
within its1,800second combined cap. All585 frozen sources, inputs, support files,
production pins, event/observation/capture counts and final finite state were
rechecked. Both initialize exactly from seed204, use the full10k LR horizon,
have eight stable training graphs and no graph breaks. No new warm run.

At2,000 calls, QR20/warm10-QR50/QR50 CE is3.990382792/3.993939314/3.998679463.
Warm is+.003556522 behind QR20 and-.004740149 ahead of QR50. QR20 is better at
six of seven noninitial validation points; warm is better than QR50 at all
seven. Repeated panels are not independent seed replications. These are early
prefix comparisons, not evidence of final10k superiority or statistical
equivalence. The fixed method and learning schedule remain unchanged.

Complete cycles101..2000 (1,900 calls) take329.641/314.142/313.551ms respectively;
optimizer40.460/26.415/22.864ms. QR20 and QR50 are new sequential timings on the
same GPU; warm is cached from the earlier full seed204 launch and is contextual.
Do not call the tiny warm-versus-QR50 wall difference a precise paired win.
Warm adds152 refreshes in this window; the observed optimizer mean difference
is~3.55ms/call across launches, with a modest quality gain over holding bases
between QR50s. This does not isolate a causal kernel cost.
This is an implementation tradeoff, not a universal cadence advantage.

All three use1,245,708,320bytes matrix state. QR20/QR50 peak allocation is
8,829,511,680/8,825,143,296bytes. Diagnostic max actual-M error.5499%/.5105%,
normalized Gram.002364/.002366. Historical flags remain untouched and diagnostic.
Event totals: QR20 cold1/ordinary1900/QR99; QR50 cold1/ordinary1960/QR39, no warm.

The only remaining approved GPU work is the small same-state QR attribution
profile; no additional full training or tuning is implied by this result.
