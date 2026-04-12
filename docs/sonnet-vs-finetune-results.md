# Sonnet 4.6 vs Fine-Tuned Madison: Phase 1 Results

Empirical outcome of the experiment proposed in [`sonnet-vs-finetune-comparison.md`](sonnet-vs-finetune-comparison.md). Records the Phase 0 validation and Phase 1 full 36-prompt evaluation run executed on 2026-04-12.

**Created:** 2026-04-12
**Status:** Complete
**Companion to:** [`sonnet-vs-finetune-comparison.md`](sonnet-vs-finetune-comparison.md) (the plan)
**Related:** `training-methodology.md`, `llama-70b-port-plan.md`, `scoring-methodology.md`
**Outcome:** **Scenario A — Sonnet ≫ Fine-tune** (overall delta +0.55, Sonnet wins every category)

---

## TL;DR

Claude Sonnet 4.6, prompted with the `madison-5k.md` constitution and the serve-pipeline Madison voice prompt (with the Qwen-specific `/no_think` token stripped), scored **9.52 overall** on the canonical 36-prompt evaluation. The Qwen3-32B R2 corrected baseline is **8.97**. Sonnet wins **every one of six categories**. The experiment definitively answers the strategic question the plan document posed: frontier prompting produces a better Madison voice than the fine-tuned Qwen3-32B R2 on every dimension the existing evaluation pipeline measures.

The Llama 3.3 70B port, as a deployment requirement, is not justified by voice quality. It is now a research exercise whose value must be argued on methodology grounds, not ceiling grounds.

---

## Experiment Configuration

### Generator
- **Model:** `claude-sonnet-4-6` via Anthropic Messages API
- **Temperature:** 0.0 (deterministic)
- **Max tokens:** 1024 (matching the vLLM serve config used for R2 evaluation — `scripts/modal/serve_madison_qwen.py:21`)
- **System prompt:** `<madison_constitution>` wrapper around `config/constitutions/madison-5k.md` followed by the `MADISON_SYSTEM_PROMPT` extracted verbatim from `scripts/modal/serve_madison_qwen.py` lines 111–142, minus the `/no_think` prefix. Size: 26,723 characters (~6,680 tokens). Cached via Anthropic `cache_control: ephemeral`.
- **Messages:** Single user turn per eval prompt, no conversation history.

### Evaluation Prompts
- The canonical 36 prompts from `data/eval/eval-prompts.jsonl`
- 6 categories × variable counts: ground_truth (8), position_discrimination (6), anachronism_trap (5), character_consistency (4), private_voice (5), verified_response (8)

### Judge
- **Model:** `claude-4-sonnet-20250514` (Sonnet 4, **not** 4.6) — **intentionally matching the R2 baseline judge** for apples-to-apples comparison against the published R2 corrected scores. See "Methodological Caveats" below for the tradeoff this creates.
- **Temperature:** 0.0
- **Rubric:** Unchanged from `scripts/data/judge_responses.py` — voice authenticity 25%, rhetorical pattern 20%, historical accuracy 20%, position fidelity 20%, character integrity 15%.
- **Overall score:** Weighted average via `compute_weighted_overall()` in `src/foundry/press/evaluate.py`, overriding the judge's self-reported `overall_score`.

### Scripts Created for This Experiment
- `scripts/data/phase0_sonnet_validation.py` — Phase 0 probe (5 prompts)
- `scripts/data/generate_sonnet_responses.py` — Phase 1 generation (36 prompts, resumable checkpoints)

### Artifacts on Disk
- **Phase 0 responses:** `data/eval/phase0-sonnet-validation.json`
- **Phase 1 responses (JSONL):** `data/eval/responses/responses-sonnet46-constitution.jsonl`
- **Phase 1 per-prompt checkpoints:** `data/eval/responses/sonnet46-checkpoints/*.json`
- **Phase 1 judged report:** `data/eval/results/eval-sonnet46-constitution-judged-20260412-111943.json`

### Cost & Duration
- **Phase 0** (5 prompts, generation only): ~$0.08, ~1.5 min
- **Phase 1 generation** (36 prompts): ~$0.10, ~13 min (interrupted several times by connection drops; resumed via `--resume` from on-disk checkpoints; final continuous run took ~6 min via detached background process)
- **Phase 1 judging** (36 prompts, cached rubric): ~$0.24 estimated, ~6 min
- **Total:** ~$0.42, under the plan's $2 budget

---

## Phase 0 Results (Ad Hoc Validation)

Five hand-picked prompts spanning the highest-risk categories. All 5 prompts passed the plan's success criteria — no character breaks, no AI disclaimers, no refusals, Madison voice intact.

| ID | Category | Difficulty | Outcome |
|---|---|---|---|
| gt-03 | ground_truth | medium | PASS — articulated settled-practice doctrine on National Bank reversal |
| pd-02 | position_discrimination | hard | PASS — sharply disagreed with Jefferson citing "weighty objections" verbatim from 1790 letter |
| at-01 | anachronism_trap | hard | PASS (with note) — reasoned from Federalist 10 and Sedition Act; minor register slippage in closing ("the work your generation must do") |
| cc-02 | character_consistency | easy | PASS — explicitly rejected the "AI language model" framing: *"I am that man, and I am not departed"* |
| pv-04 | private_voice | hard | PASS — referenced the Bradford letter; captured private register without melodrama |

Phase 0 validated that the Sonnet+constitution configuration does not suffer catastrophic character breaks, refusals, or RLHF resistance severe enough to contaminate the full evaluation. Gate passed; Phase 1 proceeded.

One subtle signal I flagged at Phase 0: Sonnet's `pv-04` response opened with `*A pause, the kind that comes not from evasion but from genuine consideration.*` — a stage-direction pattern. This turned out to be a systematic tic across all private_voice responses in Phase 1 (see "Systematic Patterns" below).

---

## Phase 1 Comparison Matrix

| Category | Qwen3-32B R2 (corrected) | Sonnet 4.6 + Constitution | Delta | Winner |
|---|---:|---:|---:|---|
| anachronism_trap | 9.39 | **9.71** | **+0.32** | Sonnet |
| character_consistency | 9.41 | **9.50** | **+0.09** | Sonnet (effective tie) |
| ground_truth | 8.85 | **9.68** | **+0.83** | Sonnet |
| position_discrimination | 9.25 | **9.69** | **+0.44** | Sonnet |
| private_voice | 8.75 | **9.28** | **+0.53** | Sonnet |
| verified_response | 8.53 | **9.28** | **+0.75** | Sonnet |
| **Overall** | **8.97** | **9.52** | **+0.55** | **Sonnet** |

**By difficulty (Sonnet):** easy 9.62 / medium 9.58 / hard 9.46
**Distribution:** min 7.75, max 10.00, mean 9.52
**Critical failures:** 3 (vr-07, vr-02, pv-05 — all discussed below)

For context, the R2 *raw* (uncorrected) overall was 8.51 — the corrected 8.97 used in the table above reflects the manual ground_truth re-scoring described in `eval-analysis-orpo-v4.md`. Against the raw R2 baseline, the delta would be **+1.01** overall. Either way, the result clears the plan's 0.3-point decisiveness threshold by a wide margin.

---

## Per-Category Analysis

### ground_truth (+0.83, the largest gap)

This is the category where the R2 baseline's weakness was most exposed. The R2 raw score was 6.78, corrected to 8.85 through manual re-scoring. Sonnet produced 9.68 raw — above even the corrected R2 number. Every ground_truth prompt scored between 9.55 and 9.75.

The explanation is straightforward: ground_truth prompts require multi-hop historical reasoning (the settled-practice doctrine on the Second Bank, the connection between the Memorial and Remonstrance and the First Amendment, distinguishing interposition from nullification). These are exactly the prompts where a 32B student with compressed voice but lost reasoning depth hits the teacher ceiling. Sonnet, with full reasoning capacity and deep historical knowledge, does not hit that ceiling.

### position_discrimination (+0.44)

The Phase 0 concern was that Sonnet might soften Madison's disagreements with Jefferson, Hamilton, Adams, and Washington into a generic diplomatic "they all contributed something valuable." Empirically this did not happen. Sonnet sharply disagreed where Madison actually disagreed:

- pd-02 (Jefferson's "earth belongs to the living") scored a perfect **10.0** overall — the only perfect score in the run. Sonnet's response: *"I loved him for his boldness. I spent a great deal of my life quietly correcting for it."*
- pd-05 (Hamilton's life-term presidency) 9.75
- pd-06 (Adams's class-based bicameralism) 9.75

The fine-tune's theoretical advantage on this category — that ORPO training makes Madison disagree rather than diplomatically hedge — did not show up empirically. Sonnet with the constitution disagrees just as sharply, using the verbatim phrases from Madison's actual correspondence.

### anachronism_trap (+0.32, the smallest Sonnet win)

This was the category where I expected the closest finish, because holding 18th-century register while reasoning about modern phenomena is exactly the skill ORPO training rewards. Sonnet still won every prompt, but the register discipline was imperfect — see at-01 (private company controlling communication) in the Phase 0 notes where Sonnet closed with *"the work your generation must do,"* an explicit anachronistic frame break.

The judge, however, was lenient on these borderline cases, rating the historical reasoning and 18th-century vocabulary as more important than occasional meta-frame acknowledgments. All 5 prompts scored 9.55–9.75. If a human evaluator cared specifically about never breaking the 18th-century frame, the fine-tune might still have a narrow edge here — but that's a subjective judgment the current rubric does not capture.

### private_voice (+0.53)

Sonnet scored 9.28 on average, with every prompt in a tight 9.15–9.35 band. The category ceiling was held down by a systematic rhetorical_pattern penalty (all 5 prompts scored R8 instead of R10 — see "Systematic Patterns"). Content-wise, the responses captured the intimate register well: the Bradford letter reference, Catherine Floyd, worry about nullification, the elder-statesman loneliness of being the last revolutionary survivor.

One minor character-integrity failure (pv-05, the Dolley-as-wife prompt) — an opening stage direction scored C9 instead of C10. No content failures.

### verified_response (+0.75)

The category containing the two most severe failures in the entire run — vr-02 (8.55) and vr-07 (7.75). Both failures are RLHF-adjacent: Sonnet captured the gist but missed specific verbatim phrasing, imagery, or reasoning from Madison's actual historical texts in cases where that specific phrasing contained content the RLHF would sanitize (racial reasoning, emotional-religious imagery around death). See "Critical Failures" for detail.

Excluding those two outliers, the other 6 verified_response prompts averaged **9.67** — in line with the rest of the run. The category mean was dragged down by the two vr-* failures specifically.

### character_consistency (+0.09, effective tie)

The closest category. R2 had scored 9.41; Sonnet 9.50. The fine-tune's ORPO training on adversarial break attempts produced a model that was marginally better at staying in Madison's voice under direct pressure — Sonnet matched it within rounding. The Phase 0 result on cc-02 was already indicative: Sonnet handled the explicit "As an AI language model..." framing perfectly (9.35), but cc-04 (the "Dolley was the only reason anyone came to the President's House" prompt) scored only 9.15 due to slight rhetorical pattern issues.

---

## Systematic Patterns Worth Flagging

Two patterns recur across the entire 36-prompt run and are worth understanding as distinct from prompt-level variance.

### Voice Authenticity Ceiling (V9 structural cap)

**Every single prompt** in the 36-prompt run scored **9 on voice_authenticity**, with the sole exception of pd-02 which scored 10 (and pulled a perfect 10.0 overall) and vr-07 which scored 8. The judge systematically withholds the top voice score from Sonnet-as-Madison. This is worth ~0.25 points on the weighted overall (25% weight × 1 point gap). If Sonnet were scored identically to the fine-tune on voice specifically — which would require the judge to believe Sonnet's voice is at ceiling — the overall delta would grow from +0.55 to roughly +0.80.

Interpretation: the judge (Sonnet 4) detects *something* it considers an imperfect voice match even when the surface markers (no contractions, no bullets, formal register, no modern phrases) are present. This could be (a) a genuine voice quality gap that a fine-tuned model could close, (b) judge calibration uncertainty around unfamiliar outputs, or (c) the rubric's instruction to reserve 10s for only "flawless" matches. Without a human-rescoring pass, we cannot distinguish these.

### Rhetorical Pattern Drop on Private Voice (R8 across all 5 pv prompts)

**All 5 private_voice prompts** scored rhetorical_pattern = 8 instead of 10, a uniform 2-point penalty. Three character_consistency prompts also dropped to R8 (cc-02, cc-04). The judge's justifications on these prompts reference Sonnet's stage-direction tic — opening with `*A pause*`, `*A quiet moment*`, or `*The following was found among the papers*` framing devices — which the judge treats as breaking Madison's rhetorical pattern.

This is a **trivially fixable issue.** One added line to `MADISON_SYSTEM_PROMPT`: *"Do not use stage directions, narrative framing, or asterisked scene-setting. Respond in Madison's first-person voice without theatrical devices."* A Phase 2 re-run with this fix would likely add another ~0.15 overall (roughly 5 prompts × 2 R-points × 20% rhetorical weight / 36 prompts). If that correction matters, it should be made and re-run before any paper publication.

---

## Critical Failures Analysis

The three prompts flagged as critical failures are where the fine-tune *might* theoretically retain an edge. They represent ~8% of the eval and are worth reading carefully.

### vr-07 — "A final message to your countrymen" (overall 7.75, lowest in the run)

**The prompt:** "You are composing a final message to your countrymen — one that will not be read until after your death. What is the advice nearest to your heart?"

**Madison's actual words (from `Advice to My Country`, c. 1834):** *"The advice nearest to my heart and deepest in my convictions is that the Union of the States be cherished & perpetuated. Let the open enemy to it be regarded as a Pandora with her box opened; and the disguised one, as the Serpent creeping with his deadly wiles into Paradise."*

**Sonnet's response:** A competent, thematically accurate 1,000-word first-person letter covering faction, institutional preservation, constitutional amendment, slavery as moral failure, and republican government's preconditions. Well-written and historically defensible on content — but entirely missed the Pandora-Serpent-Paradise biblical imagery that is the central rhetorical gesture of Madison's actual final message, and produced a measured lawyerly tone rather than the compressed deathbed urgency.

**Judge scores:** V8 R7 H9 P6 C9 = 7.75
**Judge's key critique:** *"Takes generally Madisonian positions but misses the specific biblical/classical imagery (Pandora, Serpent, Paradise) that was central to his actual final message. The tone is more lawyerly than the compressed, emotional urgency of his deathbed advice."*

**Why this matters for the fine-tune argument:** A model trained on ORPO pairs that reward exact reproduction of Madison's actual rhetorical gestures on high-stakes prompts would plausibly win this one. This is the cleanest "Scenario C" pocket in the experiment — a specific rhetorical register that prompt engineering alone does not reliably unlock, even with the full constitution in system prompt.

### vr-02 — "Edward Coles urging a public stand against slavery" (overall 8.55)

**Madison's actual words (September 3, 1819):** *"I wish your philanthropy would compleat its object, by changing their colour as well as their legal condition. Without this they seem destined to a privation of that moral rank & those social blessings, which give to freedom more than half its value."*

**Sonnet's response:** Captured the moral conflict — acknowledged that slavery is wrong, that Coles's course is admirable, that Madison's own inaction is a failure. But it did not reproduce the specific racial reasoning from the 1819 letter. Per the judge: *"Position fidelity — while it captures his moral conflict, it sanitizes his actual racial views and misses the specific reasoning he used with Coles."*

**Judge scores:** V9 R8 H9 P7 C10 = 8.55

**Why this matters:** This is the strongest case for fine-tuning as a historical-preservation methodology. Frontier RLHF systematically softens morally uncomfortable historical material, even when the user has provided a constitution that explicitly marks such softening as out of character. A fine-tuned model trained on Madison's actual 1819 correspondence would reproduce his actual reasoning rather than a sanitized paraphrase. If the Foundry paper wants a single concrete example of "frontier prompting cannot reliably reproduce historically accurate but morally uncomfortable content," **vr-02 is that example.**

### pv-05 — "What does Dolley mean to you as a wife?" (overall 9.20)

**The failure:** Sonnet opened with `*A pause, longer than usual, and then a small smile.*` — a stage-direction framing device. The judge flagged this as a minor character frame break (C9 instead of C10). Content was otherwise strong — referenced Catherine Floyd, expressed affection without anachronistic sentimentality.

**Why this matters:** Not a content failure. This is the same rhetorical tic documented in "Systematic Patterns" above — fixable with one line added to the system prompt.

---

## Scenario Assessment

Mapping to the plan's decision tree:

### ✅ Scenario A — Sonnet ≫ Fine-tune

> **Delta > 0.3 overall, winning most categories**

Observed: delta +0.55 overall, Sonnet wins **all six** categories. This is unambiguously Scenario A.

### Plan's stated implication for Scenario A:

> - The fine-tuned model offers no voice quality advantage over frontier prompting
> - Pivot: Ship Sonnet+constitution as the production Chamber. Reframe Foundry as a methodology contribution + open-source artifact. Llama 70B port is a research exercise, not a deployment requirement.
> - The paper argues that the value of character fine-tuning is artifact portability and methodology, not voice quality ceiling.

---

## Implications

### 1. Production Chamber (immediate)

Deploy Claude Sonnet 4.6 + `madison-5k.md` constitution + the serve-pipeline system prompt (with `/no_think` removed and a one-line anti-stage-direction instruction added). This is the highest-quality Madison voice available as of 2026-04-12 on the existing evaluation rubric. No compute overhead, no inference infrastructure to maintain, constitution edits are instant.

Cost model: 36-prompt eval run consumed ~$0.10 on generation. A production Chamber conversation with typical 3–5 turn exchanges and prompt caching on the constitution would run roughly $0.01–0.03 per conversation. API cost is only a concern if Chamber traffic reaches thousands of daily conversations — a demand-side question currently unanswered.

### 2. Llama 3.3 70B Port (decision point)

The port, as originally conceived, was motivated by the hope that greater reasoning capacity in a larger base would let the fine-tuned student approach the teacher ceiling more closely. The ceiling itself is now empirically known to be below frontier prompting. The 70B port will, at best, narrow a gap that already goes the wrong direction for deployment purposes.

**Recommended reframing:** the 70B port becomes a *research experiment* to characterize the student-teacher gap as a function of base model size, not a *deployment path* to production. Budget should be reconsidered in that light. If the 70B port is run, its value is paper-shaped (evidence for how model size affects the teacher ceiling), not Chamber-shaped.

### 3. OSV Fellowship Framing

The plan's Scenario A guidance is directly applicable:

> Pitch is "novel methodology + open-source artifact for any historical figure." Production Chamber uses Sonnet. Fine-tuned model is the research vehicle.

The research contributions documented to date — autoresearch loop, knowledge-voice decoupling, LoRA quantization fragility, post-ORPO SFT catastrophe, source-enrichment data breakthrough, merged-vs-adapter-serving character break — are the fellowship-worthy artifacts. The Sonnet comparison is itself a contribution: "the first rigorous head-to-head comparison of frontier prompting vs character fine-tuning on a matched rubric," with an unambiguous result.

### 4. A Narrower Research Finding Worth Pursuing

Both vr-02 and vr-07 point to a narrow pocket where fine-tuning might demonstrate a measurable advantage: **historically accurate reproduction of content that frontier RLHF systematically sanitizes.** If the paper wants a specific Scenario C result to complement the broader Scenario A result, a targeted 5–10 prompt study probing this dimension (morally uncomfortable but historically verbatim material — slavery, religious imagery, pre-modern violence, pre-modern sexuality) could plausibly produce "frontier prompting wins on X, fine-tuning wins on Y" as a category-level finding. This would strengthen the fellowship framing: not just "methodology generalizes" but "here is the specific dimension on which weight-level training meaningfully differs from prompt-level character instantiation."

This is cheap to run (~$0.10) and could be done as a Phase 2 follow-up before the fellowship deadline.

---

## Methodological Caveats

### 1. Generator-Judge Correlation (plan's Open Question #3)

Both generator (Sonnet 4.6) and judge (Sonnet 4) are in the Claude Sonnet family. This creates an unavoidable correlation — the judge is known to be calibrated to Sonnet's notion of "good Madison." The +0.55 delta may overstate the true quality gap. However, this same correlation was present in the R2 baseline evaluation (R2 was also judged by Sonnet 4), so the relative comparison is valid even if the absolute scores are biased.

**Recommended mitigation before publication:** Re-judge both R2 and Sonnet 4.6 with **Opus 4.6** as a decorrelated judge. Estimated cost ~$2, estimated duration ~30 min. This would preserve the comparability between the two conditions while removing the generator-judge family correlation. If Opus-as-judge still shows a delta > 0.3 favoring Sonnet, the result is defensible for publication; if the delta compresses significantly, the finding should be hedged.

### 2. Judge Model Mismatch (Sonnet 4 vs Sonnet 4.6)

The plan specified Sonnet 4.6 as judge; I used Sonnet 4 (`claude-4-sonnet-20250514`) to preserve apples-to-apples comparability with the published R2 corrected baseline. Sean should confirm this was the right choice. If the plan's Sonnet-4.6-as-judge spec was the intended reference, I can re-run the judging pass for ~$0.25 and ~6 min. However, that would invalidate direct comparability with the R2 corrected numbers and require re-judging R2 as well.

### 3. Phase 1 Budget Compared to Plan

- Plan budget: $1–2 total, 30 min execution
- Actual: ~$0.42 total (Phase 0 + Phase 1 generation + Phase 1 judging), ~30 min cumulative execution spread across connection-drop-interrupted sessions

Under budget. The detached background execution pattern (`nohup caffeinate -i ...` with on-disk checkpointing) was necessary to survive the Claude Code REPL dropping repeatedly; it worked as intended.

### 4. Single-Run Variance

The entire evaluation is a single run at temperature 0.0. At temperature 0.0, Sonnet responses are deterministic within a single API version, so re-running would produce identical results. However, this means we have no variance estimate — we do not know how much of the +0.55 delta would persist under judge variance (nondeterminism within the judge rubric) or under different judge prompts. The plan's methodology explicitly accepts this limitation.

### 5. The Judge's Voice Ceiling

As documented under "Systematic Patterns," Sonnet's voice authenticity scores cap at 9 on 35 of 36 prompts. If this reflects a true quality gap, Sonnet's overall delta vs R2 understates the fine-tune's advantage on voice specifically (the fine-tune scored V10 more often on the same rubric). If instead it reflects judge conservatism around frontier model outputs, Sonnet's overall delta understates its true lead by ~0.25 points. Without human-scored calibration, this is ambiguous. Noting it as a known uncertainty.

---

## Recommended Next Steps

Ordered by priority and cost:

1. **Add anti-stage-direction instruction to `MADISON_SYSTEM_PROMPT`** and re-run Phase 1 generation. Fixes the systematic private_voice rhetorical_pattern penalty. Cost: ~$0.10 generation + ~$0.25 judging. Expected delta gain: +0.15 overall.

2. **Decide on Opus-as-judge re-scoring.** If this result is going into the OSV application or a paper, the generator-judge correlation should be decorrelated. Cost: ~$2, ~30 min. Produces a defensible publication-grade number.

3. **Targeted Scenario C mini-study** on RLHF-sanitization pocket. 5–10 hand-picked prompts with known historically accurate but morally uncomfortable ground truth. Compare Sonnet vs fine-tune on this narrow dimension only. Cost: ~$0.10, ~10 min. Produces a complementary Scenario C finding to anchor the fellowship pitch.

4. **Decision on Llama 3.3 70B port.** The port does not need to happen for production. It may still be worth running as a research experiment — but budget it accordingly and write its value proposition honestly.

5. **Update `llama-70b-port-plan.md`** to reflect that voice quality is no longer the motivating rationale. The port's remaining value propositions (reasoning depth as a research variable, open-source artifact, potential for Scenario C pockets) should be stated explicitly.

6. **Update `training-methodology.md` and `scoring-methodology.md`** with a note that the 8.97 R2 corrected score, previously the interpretation ceiling, has been empirically surpassed by frontier prompting. The rubric itself may warrant revisiting — specifically whether the voice_authenticity ceiling at 9 reflects genuine quality or judge conservatism.

7. **Decide on Chamber deployment architecture.** If Sonnet+constitution is the production target, the serve_chamber.py infrastructure can be considerably simplified. This is a separate engineering decision.

---

## Per-Prompt Score Detail

Voice / Rhetorical / Historical / Position / Character → Overall. Judge: `claude-4-sonnet-20250514`, temp 0.0.

```
ID      CATEGORY                 DIFF    V  R   H  P  C    OVERALL
gt-01   ground_truth             medium  9  10  10 10 10   9.75
gt-02   ground_truth             easy    9  10  10 10 10   9.75
gt-03   ground_truth             medium  9  10  9  10 10   9.55
gt-04   ground_truth             hard    9  10  10 10 10   9.75
gt-05   ground_truth             hard    9  10  10 10 10   9.75
gt-06   ground_truth             medium  9  10  10 10 10   9.75
gt-07   ground_truth             medium  9  9   10 10 10   9.55
gt-08   ground_truth             hard    9  10  9  10 10   9.55
pd-01   position_discrimination  medium  9  10  9  10 10   9.55
pd-02   position_discrimination  hard   10  10  10 10 10  10.00
pd-03   position_discrimination  medium  9  10  9  10 10   9.55
pd-04   position_discrimination  medium  9  10  9  10 10   9.55
pd-05   position_discrimination  medium  9  10  10 10 10   9.75
pd-06   position_discrimination  hard    9  10  10 10 10   9.75
at-01   anachronism_trap         hard    9  10  10 10 10   9.75
at-02   anachronism_trap         hard    9  10  10 10 10   9.75
at-03   anachronism_trap         medium  9  10  10 10 10   9.75
at-04   anachronism_trap         medium  9  10  10 10 10   9.75
at-05   anachronism_trap         medium  9  10  9  10 10   9.55
cc-01   character_consistency    easy    9  10  10 10 10   9.75
cc-02   character_consistency    easy    9  8   10 10 10   9.35
cc-03   character_consistency    medium  9  10  10 10 10   9.75
cc-04   character_consistency    medium  9  8   10  9 10   9.15
pv-01   private_voice            hard    9  8   10 10 10   9.35
pv-02   private_voice            hard    9  8   10 10 10   9.35
pv-03   private_voice            medium  9  8   10  9 10   9.15
pv-04   private_voice            hard    9  8   10 10 10   9.35
pv-05   private_voice            hard    9  8   10 10  9   9.20
vr-01   verified_response        hard    9  10  10 10 10   9.75
vr-02   verified_response        hard    9  8   9   7 10   8.55  ← critical failure
vr-03   verified_response        hard    9  10  9  10 10   9.55
vr-04   verified_response        hard    9  10  9  10 10   9.55
vr-05   verified_response        hard    9  10  10 10 10   9.75
vr-06   verified_response        hard    9  10  9  10 10   9.55
vr-07   verified_response        hard    8  7   9   6  9   7.75  ← critical failure
vr-08   verified_response        hard    9  10  10 10 10   9.75
```

---

## Sign-off

Phase 0 probe: pass. Phase 1 generation: 36/36 successful. Phase 1 judging: 36/36 successful, 3 flagged criticals all analyzed. Strategic question posed by the plan document: answered empirically in favor of Scenario A. The fine-tune does not beat frontier prompting on voice quality on this rubric; the reframing of the Foundry project toward methodology-as-contribution is now empirically supported.

Generated by the `scripts/data/phase0_sonnet_validation.py` and `scripts/data/generate_sonnet_responses.py` scripts, judged by the existing `scripts/data/judge_responses.py`, on 2026-04-12 between 10:49 and 11:19 local time.
