# From trip records to a review decision

The monitor connects three different objects: an event rate, a ranking, and a later observation of that ranking. Keeping their definitions separate is essential. A precise rate estimate need not identify the best use of a limited review budget, and a persistent service interval does not establish what caused it.

## Define the observation before choosing the model

The input consists of completed-trip records. After the configured validity filters, each accepted trip contributes one Bernoulli event: whether pickup occurs more than ten minutes after the request. The analysis conditions on eligibility, including the maximum retained interval and month-boundary guard. These filters determine the statistical population.

The daily panel stores $(n_{cd},k_{cd})$ for cell $c$ on date $d$. A cell fixes zone, platform, four-hour slot, and weekday; consequently its observed dates recur on the same weekday. A missing date with no accepted observations does not create evidence of a zero event rate. Panel validation removes zero-count rows and requires finite integer counts satisfying $0\leq k_{cd}\leq n_{cd}$.

There are two related estimands:

* **Within-cell change:** the difference in event rates for the same zone/platform/slot/weekday combination across periods.
* **Standardized zone change:** the difference after holding the reference composition of platform, slot, and weekday fixed within a zone.

The first supplies the review queue. The second helps diagnose composition effects. Neither follows automatically from a change in the overall trip-weighted rate.

## Derive the shrinkage rule

For a cell with current counts $(k,n)$, write a working Binomial likelihood and a Beta prior with mean $p_0$ and strength $\tau$:

$$
k\mid p\sim\mathrm{Binomial}(n,p),\qquad
p\sim\mathrm{Beta}(\tau p_0,\tau(1-p_0)).
$$

For $0<p_0<1$ and $\tau>0$, conjugacy gives

$$
\mathbb E[p\mid k,n,p_0]
=\frac{k+\tau p_0}{n+\tau}
=\frac{n}{n+\tau}\frac{k}{n}
+\frac{\tau}{n+\tau}p_0.
$$

This separates the observed rate from the amount of borrowing. With $n=100$ and $\tau=50$, the current rate receives weight $2/3$; with $n=1{,}000$, it receives $20/21$. These are algebraic illustrations of the implemented rule. At boundary parent rates the code still evaluates the weighted-average expression; setting $\tau=0$ explicitly recovers the raw-rate calculation.

The parent pools reference trips across zones within the same platform, slot, and weekday. If that parent is absent, the code uses the reference-wide rate, recording the source of the prior. Both current and reference estimates use the same $p_0$. The prior is empirical because it comes from observed reference counts, and its fit is held fixed in the subsequent resampling calculation.

Using the same parent does not cancel its influence from the difference. When $k=k_0=0$,

$$
\widetilde p-\widetilde p_{\mathrm{ref}}
=\tau p_0\left(\frac{1}{n+\tau}-\frac{1}{n_0+\tau}\right).
$$

This is positive when $p_0>0$ and $n<n_0$, despite both observed rates being zero. The priority rule therefore also requires a positive **raw** change. The corresponding [regression test](../tests/test_analysis.py) makes this edge case an explicit property of the decision rule.

## Resample the observational unit

Trips sharing a date may respond to common conditions. Resampling individual trips would discard that grouping. The implementation instead resamples the observed dates within each cell and period, carrying each date's total and event count together. Each replicate sums those paired counts, applies the fixed-prior shrinkage rule to both periods, and records their difference in percentage points.

Current and reference dates are resampled independently. The default 300 replicates produce a percentile interval from the 2.5th and 97.5th percentiles. The calculation retains within-date aggregation but does not model dependence across dates, refit the parent, or adjust for screening thousands of cells. Its role is a conditional uncertainty diagnostic for review prioritization.

The cell-specific random stream is a hash of the seed, cell key, and stream label. This is more than repeatability within one run: an unrelated parent group can be added without perturbing an existing group's estimate or interval. The tests check both that invariance and isolation from future months.

## Freeze the comparison population

Suppose a zone has reference cell weights

$$
w_c=\frac{n_{c,\mathrm{ref}}}{\sum_{j\in\mathrm{zone}}n_{j,\mathrm{ref}}}.
$$

Standardized current and reference rates use the same weights:

$$
p_{\mathrm{current}}^{\mathrm{std}}=\sum_c w_c\widehat p_{c,\mathrm{current}},\qquad
p_{\mathrm{ref}}^{\mathrm{std}}=\sum_c w_c\widehat p_{c,\mathrm{ref}}.
$$

If cell rates stay constant while trip proportions change, these standardized rates stay constant. The composition-only test constructs that situation directly. An ordinary pooled rate can move in the same example.

Full reference-weight coverage is required by default. A missing current cell makes the full-population standardized estimate unavailable. If partial coverage is explicitly allowed, both periods are renormalized to the same observed subset and the output identifies that changed estimand. Current-only cells receive no invented reference weight; zones without a reference composition remain unavailable.

## Evaluate the ranking without moving its target

The historical check fixes the reference and monitoring windows before examining the next observation month. It compares raw rate, raw change, and shrunk change on one supported candidate pool. Otherwise a method could appear better simply by selecting from an easier population.

For a selected cell, define the later outcome as

$$
Y_c=\mathbf{1}\left\{
100\left(\widehat p_{c,\mathrm{next}}-\widehat p_{c,\mathrm{ref}}\right)>2
\right\}.
$$

The reference rate stays frozen at its original raw value. A later outcome is observable only with at least 100 next-month trips, three observed dates, and a valid original reference. Missing or low-support observations are unknown. The reported hit rate divides by observable selections, while effective queue size and unknown count are reported alongside it.

The priority queue applies an additional evidence filter and is reported separately. Its candidates need not be the same pool used by the three ranking comparisons. In the recorded March-to-April check, all twenty selections are observable for each ranking. Raw-rate ranking yields thirteen persistent cells; both change-based rankings yield fifteen. Those counts support a comparison within this particular window, without resolving whether shrinkage improves ranking across other periods.

## Preserve the chain from source to result

The engineering checks are part of the statistical design because a comparison cannot be interpreted without knowing what was observed. [Ingestion](../tlc_service/data.py) accounts for source rows, scanned rows, accepted rows, and rejected categories. Aggregated panel counts must reconcile with the accepted total. Reuse requires matching processing parameters, source ETag and size, and output hash.

[Provenance tests](../tests/test_cli_provenance.py) deliberately alter panels, omit audits, and introduce inconsistent totals. They also ensure that generated multi-month data cannot establish a claim of scanning official files. The distinction matters at scale: 80 million scanned records and 367,392 daily rows are different computational objects, not interchangeable counts of training examples.

For a focused code review, start with the comparison, ranking, next-month evaluation, and zone-standardization functions in [analysis.py](../tlc_service/analysis.py). The [April summary](../examples/pilot_verified/summary.json) and [forward-check summary](../examples/pilot_forwardcheck/summary.json) connect their outputs to fixed configurations and observation windows.
