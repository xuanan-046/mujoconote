# MovieMind — Defense Q&A Preparation

Anticipated examiner questions, organized by theme, with answers grounded
in what was actually built and measured — not generic thesis-defense
boilerplate. Where a real limitation exists, the answer says so directly;
that's more defensible than deflecting.

---

## 1. Design & Methodology Justification

**Q: Why two thresholds (τ_sim and τ_margin) instead of just one similarity cutoff?**
They catch two different failure modes. τ_sim alone would let a genuinely
tied pair of candidates both clear the bar and still get silently
resolved to whichever one scores marginally higher — which is exactly
the original bug. τ_margin specifically checks the *gap* between the
top two candidates, so a near-tie gets caught even when both candidates
individually look "confident." Removing either one reopens a different
failure mode — that was confirmed empirically during calibration, not
assumed.

**Q: Why fuzzy string matching instead of a proper embedding-based or neural entity linker?**
Two reasons. First, scope: this thesis is about the confidence-gating
and clarification *mechanism*, not about building a better linker from
scratch — the fuzzy matcher just had to be good enough to expose and
fix the *threshold* problem, and a heavier neural linker would have
made it harder to isolate that variable. Second, it's cheap and fully
local, which matters for the cost/dependency comparison in the
Deterministic-vs-Agentic results — the whole point of that comparison is
sharper if the deterministic baseline stays lightweight.

**Q: How sensitive is the result to the exact threshold values — 0.80 and 0.02?**
τ_margin is the more load-bearing one: the calibration sweep showed
false-clarification rate jumping from 0% at 0.02 to 13% at 0.08 to 93%
at 0.12 — genuine ties always sit at margin ≈ 0, so 0.02 catches every
constructed tie in the calibration set at essentially zero false-positive
cost. τ_sim is flatter across a wide range (0.50–0.80 give identical
accuracy) — what it mainly buys is rejection power on garbage input,
which climbs from 53% to 98% correct-rejection as it approaches 0.80,
with no accuracy trade-off until you push past 0.85.

**Q: Why does the system ask only one clarifying question and then commit — why not iterate?**
That was a deliberate scope decision, and it's stated as such in the
code, not hidden: the goal was to measure whether closing *one* loop —
ambiguous mention → ask → answer — improves on the baseline's
silent-guess behaviour. A fuller multi-turn dialogue manager (multiple
outstanding ambiguities, backtracking, "none of the above") is real
future work, not something this design claims to solve.

---

## 2. Evaluation Methodology

**Q: Isn't the "designated-target" method circular — you're defining ground truth from the same graph the system is being tested against?**
It's a real limitation, and worth naming directly rather than dressing
up: it measures "does the system reach a specific, well-formed answer
when one exists," not "does the system match a real user's actual
intent." What it avoids is worse — without it, ambiguous questions have
*no* ground truth at all, and the alternative would be either (a)
declaring them unscorable and dropping ~40% of the benchmark, or (b) a
live user study, which is explicitly named as future work. It's a
principled proxy, not a substitute for the real thing.

**Q: 250 questions — is that enough to draw the conclusions you're drawing?**
For the headline comparison (baseline 64.8% vs. clarification 98.6% on
ambiguous questions, n=71 scorable), the gap is large enough that it
isn't a statistical-power concern — the effect size dwarfs the sample
noise at that n. Where sample size *is* a fair challenge is the
LLM-agent arm, which is only validated on 50 of the 250 questions —
that's flagged explicitly as a limitation and named as the first future
work item, not glossed over.

**Q: How realistic is the "noisy" evaluation set (typos, truncations) compared to real user behaviour?**
Fair challenge — it's synthetically generated, not collected from real
users, and that's stated directly in the limitations. What it does
establish is that the threshold behaves sensibly under *some* form of
noisy input, which is the minimum bar before trusting it on real input.
Validating against real user typing patterns is future work.

---

## 3. System Limitations & Robustness (examiners often probe here hardest)

**Q: What happens if both the entity and the relation are ambiguous in the same question?**
Originally, this fell into a dead end — the relation information was
silently discarded and the user got a generic "what would you like to
know?" prompt with no context. This was found during testing and fixed:
the system now retries the relation once the entity is resolved, and
either answers directly or gives an *informative* failure message
naming the specific relation it couldn't understand — instead of the
generic fallback. One honest caveat: relation-side tied candidates are
currently unreachable in this configuration (the relation linker's
tie-break margin is disabled by design, since predicate names in this
graph have almost no duplicates), so the "ask a numbered list for the
relation too" branch is implemented but not exercised in practice.

**Q: Does the system support questions like "what movie did this actor star in in this year"?**
No — and that's a specific, verified limitation, not a vague one. The
system's tools resolve `entity + relation → value`; there's no path for
reverse queries combined with an additional constraint. A pure reverse
query — "what movies did this director direct" — *is* supported (added
during this work, verified against real directors' full filmographies),
but stacking a second constraint like a year on top of that is not.

**Q: Is the recommendation feature part of this thesis's contribution?**
No, and it's described that way in the write-up rather than implied
otherwise — it's pre-existing code that predates the confidence-gating
work. It only supports "movies like X" with a named seed film, not
genre or mood-based requests, and even for the supported case, result
relevance varies noticeably between different seed films. It's included
in the system for completeness, not held up as a strength.

**Q: Is this system safe for multiple concurrent users?**
No — and this is worth being upfront about if asked. The web interface
holds one global conversation state for the life of the server process,
by design, since it was built as a single-visitor demo tool. Two
concurrent users would corrupt each other's pending clarification state.
Making it multi-user-safe (per-session state) is a small, well-scoped
future change, not a fundamental redesign.

**Q: Can a user answer a clarification with a full natural sentence, like "the one directed by Bong Joon-ho"?**
Partially. Short fragments — a bare year, a surname, a candidate's own
distinguishing name — are matched correctly (this matching logic was
found to have a real bug during testing — a reversed substring check —
and was fixed and verified). A full paraphrased sentence wrapped around
that fragment is not matched, because the check is substring-based, not
semantic. Extending it to real natural-language paraphrase matching
would need a different mechanism — flagged as future work, not
attempted here.

---

## 4. Agentic / LLM-Specific Questions

**Q: Why Claude specifically, rather than GPT-4 or an open-source model?**
Practical, not principled: Claude's API was the one available and
integrated, and the research question is about the *architecture* — can
routing be delegated to *an* LLM via tool calls without losing
correctness — not about comparing model providers. Nothing in the
design is Claude-specific; the Planner interface is swappable by
construction (that's exactly why MockPlanner exists alongside
ClaudePlanner).

**Q: The agentic version matches the deterministic version's accuracy exactly — is that a meaningful result, or is it just because the same underlying tools bound both?**
Both, and that's worth saying plainly: the ceiling is set by the same
underlying tools and knowledge graph either way, so exact accuracy
parity is partly structural. What *is* a genuine finding is that the LLM
planner reaches that ceiling reliably — it doesn't introduce new errors
by mis-routing or hallucinating an answer instead of asking — and that
the cost of doing so is fully measured (~$0.019/question) rather than
asserted.

**Q: How reliable is prompt caching in practice — what if the cache expires between calls?**
Measured directly, not assumed: in the 25-question run, cache_creation
was 0 and cache_read was over 300K tokens — meaning the cache written by
an earlier smoke-test run was still live and fully reused. If the cache
had expired, cost would rise close to the uncached first-call rate,
which is also visible in the smoke-test numbers (cache_creation was
non-zero there). The system doesn't depend on the cache for
correctness — only for cost — so an expired cache degrades cost, not
behaviour.

**Q: Why not run the full 250-question benchmark on the LLM agent, given the deck reports the number?**
Cost and time, and that trade-off was made deliberately and explicitly,
not by accident — a full run was estimated at roughly $5 and several
hours of sequential API calls. The 50-question subsample was chosen to
validate the finding cheaply before committing to the larger spend;
scaling it up is the first named item in future work.

---

## 5. Novelty & Contribution

**Q: Isn't "ask for clarification when uncertain" a well-established idea in dialogue systems research generally?**
The individual idea, yes. The specific contribution here is narrower and
more concrete: (1) a graph-derived calibration methodology that needs
zero manual labelling to set the two thresholds, (2) the
"designated-target" evaluation method that makes ambiguous-question
accuracy measurable at all without a live user study, and (3) a direct,
quantified comparison showing the same confidence-gating idea can be
implemented either as fixed rules or delegated to an LLM agent with
identical reliability and a measured cost — that three-way comparison
(baseline / rule-based clarification / agentic) on the same benchmark is
what ties it together, not any one piece in isolation.

**Q: What's the single most defensible empirical claim in this thesis?**
The 64.8% → 98.6% result on ambiguous questions, because it's not a
marginal improvement — it demonstrates that the baseline's apparent
reliability (answering nearly everything) was actively misleading, and
that a cheap, well-scoped fix (one clarification turn) closes almost all
of that gap with zero cost on the easy cases.

---

## 6. Generalization & Scope

**Q: Would this approach generalize beyond a movie knowledge graph?**
The core mechanism — score candidates, gate on two calibrated
thresholds, ask instead of guess below threshold — doesn't depend on
anything movie-specific; it depends on having a graph with resolvable
entity names and a way to generate a calibration set from the graph's
own structure. What would need re-validating in a new domain is whether
the same threshold *values* transfer, or whether the calibration process
needs re-running — which it would, since the values were fit to this
graph's specific noise characteristics, not derived analytically.

**Q: What happens as the knowledge graph grows much larger — does the fuzzy-matching approach still scale?**
Not tested at larger scale, and that's a fair limitation to acknowledge
if pressed — the current graph is fixed-size, and candidate generation
is a fuzzy match over the full name index per query. It would be
reasonable future work to check whether that stays fast enough as entity
count grows by an order of magnitude, or whether it needs an
approximate-nearest-neighbour index instead of exhaustive fuzzy scoring.

---

## Quick-reference: the honest limitations list (say these proactively if the panel doesn't ask)

- Clarification replies are evaluated against a designated target, not from a live user study.
- LLM-agent evaluation covers 50 of the 250 benchmark questions.
- The noisy-input calibration set is synthetic, not collected from real users.
- Reverse queries with a second constraint (e.g. actor + year) are not supported.
- Recommendation is pre-existing code, not part of this work's contribution, and its quality is inconsistent.
- The web interface is single-session only, not multi-user safe.
- Clarification-reply matching handles short fragments, not full paraphrased sentences.
