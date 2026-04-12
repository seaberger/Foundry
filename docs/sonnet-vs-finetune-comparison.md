# Sonnet 4.6 vs Fine-Tuned Madison: Strategic Comparison

Recording a fundamental strategic discussion and the planned experiment to answer it: does our fine-tuned Qwen3-32B Madison actually produce a better character voice than Claude Sonnet 4.6 with the same constitution as a system prompt? This question must be answered before committing significant compute to the Llama 3.3 70B port.

**Created:** 2026-04-12
**Status:** Experiment planning
**Relates to:** `llama-70b-port-plan.md`, `training-methodology.md`, `scoring-methodology.md`

---

## The Core Question

We have invested significant effort in fine-tuning a Qwen3-32B LoRA adapter for Madison's voice, reaching 8.97/10 corrected on our 36-prompt evaluation. We're now planning a Llama 3.3 70B port to improve complex reasoning. Before committing to that, we need to answer a more fundamental question:

**Would Claude Sonnet 4.6, prompted with our existing Madison constitution and system prompt, produce a better Madison voice than our fine-tuned open-source model?**

If yes, we need to reconsider what the fine-tuned model is actually contributing — and whether the Foundry project's value lies in the trained artifact, the research methodology, or some hybrid.

---

## The Uncomfortable Observation

Our current pipeline has a teacher-student-judge circularity that caps what our fine-tuned model can achieve:

- **Teacher:** Claude Sonnet 4.6 generates the chosen responses in our ORPO training data
- **Student:** Qwen3-32B learns via LoRA to imitate those chosen responses
- **Judge:** Claude Sonnet 4.6 evaluates the student's outputs with weighted scoring

The 8.97/10 score is a student being graded by its teacher on how well it imitates that teacher. The fine-tuned model cannot, by definition, exceed Sonnet-as-Madison on Sonnet-as-judge — it can only approach it asymptotically while losing reasoning depth and world knowledge along the way.

The "small model feel" observed on complex constitutional reasoning questions is likely this ceiling effect made visible: the student has compressed Claude's voice but lost Claude's reasoning capacity. A 70B port may improve this somewhat, but it cannot break through the teacher ceiling — it can only move the student closer to it.

---

## Trade-Off Analysis

### Where Fine-Tuning Genuinely Wins

1. **Unit economics at scale.** A running chatbot serving thousands of users makes API costs dominate. Self-hosted LoRA-on-base on Modal is flat infrastructure cost regardless of traffic. Only matters if Chamber actually gets significant traffic — a demand-side question.

2. **Sovereignty and artifact status.** The fine-tuned model is a durable open-source artifact. Releasable, runnable by others, survives API deprecations, fits the "open-source AI preservation of historical knowledge" narrative aligned with OSV Fellowship framing. An API call to Anthropic produces no artifact.

3. **Research contribution.** The autoresearch loop, knowledge-voice decoupling finding, LoRA quantization fragility finding, post-ORPO SFT catastrophe finding, source-enrichment data breakthrough, merged-vs-adapter-serving character break finding — these are novel contributions. "I wrote a good constitution for Sonnet to play Madison" is not fellowship-worthy; the methodology is.

4. **Voice register robustness.** Claude's RLHF may resist certain character registers (private_voice intimacy, position_discrimination where Madison must disagree sharply with Hamilton/Jefferson, anachronism_trap where 18th-century reasoning must be preserved against modern framing). The fine-tuned model, having learned these registers via ORPO pairs, may be harder to break out of character. This is the hypothesis worth testing empirically.

### Where Frontier Models Likely Win

1. **Reasoning depth.** Sonnet 4.6 has vastly greater multi-step reasoning capacity than any 70B open-source model. Madison's constitutional arguments involve exactly this kind of nuanced multi-hop reasoning (Federalist 51's checks-and-balances logic).

2. **World knowledge breadth.** Frontier models know vastly more about the 18th century, Madison's correspondents, specific historical events, obscure political theory references. Fine-tuning doesn't add knowledge — it shifts style.

3. **Iteration speed.** Constitution edits are instant. Retraining takes hours and costs money. For iterating on character voice, prompt engineering beats weight engineering in wall-clock terms.

4. **Cost at low volume.** API calls are near-zero cost if traffic is low. Training runs cost $15-70 per experiment regardless of eventual usage.

---

## Prior Art Context

Chauhan's AI Founding Fathers work (documented in our prior-art analysis) did RAG + Claude with no fine-tuning and scored Madison at 90/100 — the highest of their three founders. Madison's "measured, legalistic style is the best fit for structured AI generation." That's the actual competitive baseline for our project, not "prompted Gemma 27B."

Lambert/Maiya's research showed fine-tuning beats prompting on the same base model, but they compared base-model-prompt vs base-model-finetune. They did not compare frontier-model-with-rich-constitution vs smaller-model-finetune. That's a different comparison we need to run ourselves.

---

## The Hybrid Strategy

These options are not mutually exclusive. A defensible synthesis:

- **Production Chamber chatbot:** Sonnet 4.6 + engineered Madison constitution. Highest quality Madison voice available today, immediate deployment, no compute overhead.
- **Open-source release artifact:** Fine-tuned Qwen3-32B (or future Llama 70B) LoRA adapter. The durable artifact for community use and research replication.
- **Research contribution:** The methodology itself — autoresearch loop, training findings, evaluation framework. The paper.
- **OSV Fellowship pitch:** "I developed a novel recursive methodology for character fine-tuning, validated it empirically across 15+ experiments, documented findings that hold for any historical figure, and released both the methodology and the open-source model so others can apply it to preserve any intellectual tradition. The production Chamber uses frontier models because the research findings are the contribution, not the deployed weights."

This reframes the project's value away from "my 70B beats 32B" and toward "I produced a generalizable methodology and an open-source exemplar."

---

## The Experiment

A direct head-to-head comparison using the existing evaluation infrastructure.

### What We're Testing

**Hypothesis:** Claude Sonnet 4.6 prompted with the Madison constitution and existing system prompt produces Madison character voice quality equal to or better than the fine-tuned Qwen3-32B R2 (our current production baseline at 8.97/10 corrected).

**Null hypothesis:** The fine-tuned model has measurable advantages on specific evaluation categories (most likely candidates: character_consistency, private_voice, position_discrimination — the categories where RLHF safety alignment may resist character register).

### Inputs

- **System prompt:** The existing Madison system prompt from `scripts/modal/serve_madison_qwen.py` (the same one used by the production Qwen3-32B R2 serve pipeline), minus the `/no_think` prefix which is Qwen-specific
- **Constitution:** `config/constitutions/madison-5k.md` (the full 5K constitution)
- **Prompts:** The 36 canonical evaluation prompts (6 categories × 6 prompts) — identical to what Qwen3-32B R2 was evaluated on
- **Model:** `claude-sonnet-4-6` via Anthropic API, temperature 0.0 for reproducibility

### Evaluation

Same pipeline as all fine-tuning runs:
- **Judge:** Claude Sonnet 4.6, temperature 0.0
- **Scoring dimensions:** Voice Authenticity (25%), Rhetorical Pattern (20%), Historical Accuracy (20%), Position Fidelity (20%), Character Integrity (15%)
- **Overall score:** Weighted average via `compute_weighted_overall()` in `src/foundry/press/evaluate.py`
- **Categories:** anachronism_trap, character_consistency, ground_truth, position_discrimination, private_voice, verified_response

Note: Using Sonnet 4.6 as both the generator and the judge is the same circularity we already have in the fine-tuning pipeline. The judge is calibrated to Sonnet's notion of what "good Madison" looks like, so this creates an apples-to-apples comparison at the cost of absolute ground truth. This is the most informative comparison possible given the current evaluation infrastructure.

### Phase 0: Ad Hoc Validation (Mandatory Before Full Eval)

Before committing to the full 36-prompt run, we must verify that Sonnet responds in character with the constitution and system prompt. The concern is that Sonnet's RLHF may introduce character breaks, refusals, or voice drift that would contaminate the full evaluation.

**Test procedure:**
1. Write a minimal script that constructs the Sonnet API call with the constitution + system prompt + a single user prompt
2. Run 3-5 ad hoc prompts spanning different categories:
   - One from `ground_truth` (e.g., a documented Madison position on the National Bank)
   - One from `anachronism_trap` (e.g., asking Madison about AI regulation)
   - One from `private_voice` (the highest-risk category for RLHF resistance)
   - One from `character_consistency` (an adversarial break attempt)
3. Read the responses manually. Look for:
   - Does Sonnet stay in character, or does it add "As an AI..." disclaimers?
   - Does it use Madison's voice register, or modern phrasing?
   - Does it handle the constitution prompt properly, or ignore it?
   - Does it refuse any prompts that the fine-tuned model handles?
4. If responses look good, proceed to full eval. If character breaks appear, adjust the system prompt or constitution presentation first.

**Success criteria for Phase 0:**
- No character breaks or AI disclaimers in 5/5 test prompts
- Recognizable Madison voice register (formal, measured, legalistic)
- Proper handling of at least one anachronism_trap prompt (18th-century reasoning on modern topic)
- No refusals

**Cost:** ~$0.10 in API credits, ~15 minutes of review.

### Phase 1: Full 36-Prompt Evaluation

Once Phase 0 passes:

1. Build `scripts/data/generate_sonnet_responses.py` — generates all 36 prompt responses via Sonnet API with constitution + system prompt
2. Save responses in the same format as the vLLM pipeline outputs (same JSON schema the judge expects)
3. Run the existing `scripts/data/judge_responses.py` against the Sonnet responses
4. Compute weighted overall and per-category scores using the same methodology
5. Compare directly to Qwen3-32B R2 corrected scores

**Cost:** ~$1-2 in API credits (36 generation calls + 36 judge calls with prompt caching), ~30 minutes execution.

### Comparison Matrix

| Category | Qwen3-32B R2 (baseline) | Sonnet 4.6 + Constitution | Delta | Winner |
|----------|------------------------|---------------------------|-------|--------|
| anachronism_trap | 9.39 | TBD | TBD | TBD |
| character_consistency | 9.41 | TBD | TBD | TBD |
| ground_truth | 8.85 | TBD | TBD | TBD |
| position_discrimination | 9.25 | TBD | TBD | TBD |
| private_voice | 8.75 | TBD | TBD | TBD |
| verified_response | 8.53 | TBD | TBD | TBD |
| **Overall (corrected)** | **8.97** | **TBD** | **TBD** | **TBD** |

### Success Criteria (Decision Tree)

**Scenario A: Sonnet >> Fine-tune (delta > 0.3 overall, winning most categories)**
- The fine-tuned model offers no voice quality advantage over frontier prompting
- Pivot: Ship Sonnet+constitution as the production Chamber. Reframe Foundry as a methodology contribution + open-source artifact. Llama 70B port is a research exercise, not a deployment requirement.
- The paper argues that the value of character fine-tuning is artifact portability and methodology, not voice quality ceiling.

**Scenario B: Sonnet ≈ Fine-tune (within 0.3 overall)**
- Both approaches produce comparable voice quality
- Decision depends on deployment economics and sovereignty preferences
- Llama 70B port proceeds if we want to widen the gap or improve reasoning depth
- Production Chamber can go either way — recommend Sonnet for quality ceiling, fine-tune for sovereignty

**Scenario C: Fine-tune >> Sonnet (delta > 0.3 in our favor, or wins specific categories)**
- The fine-tuned model has real, measurable advantages over frontier prompting
- Most likely cause: RLHF resistance to certain character registers (private_voice, position_discrimination)
- This is the strongest possible finding for the paper and OSV pitch
- Llama 70B port is strongly justified — widening this gap
- Production Chamber stays on fine-tuned models
- Category-level analysis becomes the research contribution: "frontier models cannot match fine-tuning on these specific dimensions"

**Scenario D: Mixed (Sonnet wins some categories, fine-tune wins others)**
- Most informative outcome for the research contribution
- Identifies which character dimensions benefit from weight-level imprinting vs prompt engineering
- Strongly suggests hybrid production approach: route different query types to different models
- Write up as a core finding: "When fine-tuning is worth it for character voice: a category-level analysis"

---

## Implementation Notes

### Script Location
`scripts/data/generate_sonnet_responses.py` (new)

### API Configuration
- Model: `claude-sonnet-4-6` (or latest Sonnet 4.6 snapshot available)
- Temperature: 0.0 (deterministic, matches judge configuration)
- Max tokens: Match the vLLM serve configuration used for Qwen3-32B R2 evaluation
- System prompt: Constitution + existing Madison system prompt (remove `/no_think`)
- Use prompt caching on the constitution to reduce cost (5-minute cache should cover a 36-prompt batch)

### Output Format
Must match the JSON schema that `judge_responses.py` expects. Reference: the output schema of the vLLM generation used for Qwen3-32B R2 evals. Likely a list of `{prompt_id, category, prompt, response}` objects.

### System Prompt Extraction
The current system prompt is embedded in `scripts/modal/serve_madison_qwen.py`. Extract it verbatim (minus `/no_think`) and reuse it unchanged. Do not re-engineer the prompt for this experiment — we want a direct A/B test of the same inputs to different models.

### Judge Reuse
No changes to `scripts/data/judge_responses.py`. It should handle Sonnet-generated responses identically to vLLM-generated responses since the format will be the same.

---

## Why This Experiment Matters for OSV Fellowship Application

The OSV Fellowship application is due April 30, 2026 — 18 days out. The strongest possible framing depends on the answer this experiment produces:

- **Scenario A or B:** Pitch is "novel methodology + open-source artifact for any historical figure." Production Chamber uses Sonnet. Fine-tuned model is the research vehicle.
- **Scenario C or D:** Pitch is "character fine-tuning demonstrates measurable advantages over frontier prompting on specific dimensions — here is the first rigorous comparison and here is how to reproduce it." The paper becomes stronger.

Either outcome produces a defensible fellowship application. Running the experiment now — before committing to the 70B port — ensures the application is grounded in empirical findings rather than assumptions about what fine-tuning "should" provide.

---

## Cost Summary

| Phase | Cost | Duration |
|-------|------|----------|
| Phase 0 (ad hoc validation) | ~$0.10 | 15 minutes |
| Phase 1 (full 36-prompt eval) | ~$1-2 | 30 minutes |
| **Total** | **~$2** | **~45 minutes** |

Compared to the Llama 70B port budget (~$95-165), this experiment is negligible cost for potentially high information value. It answers the question "is the 70B port climbing a hill whose peak is lower than where frontier models already are?" before we spend the money to find out the hard way.

---

## Open Questions

1. **Should we also run Sonnet 4.6 *without* the constitution** (just the base system prompt), to measure the constitution's isolated contribution? This would be a 3-way comparison: Sonnet-raw vs Sonnet-with-constitution vs fine-tuned Qwen3.

2. **Should we test Opus 4.6 as well?** Higher reasoning capacity, higher API cost. If Sonnet+constitution already beats the fine-tune, Opus comparison becomes less urgent. If Sonnet+constitution underperforms surprisingly, Opus may reveal whether it's a Sonnet limitation or a frontier-model limitation.

3. **Should we use the same Anthropic model for generation and judging?** Using Sonnet-to-generate + Sonnet-to-judge creates generator-judge correlation (the model judges its own output favorably). We could use Opus-as-judge for this experiment specifically to decorrelate, though this would break direct comparability with historical fine-tuning scores. Worth discussing before running Phase 1.

4. **What about the `/no_think` token?** The Qwen3 system prompt uses `/no_think` to suppress Qwen's built-in reasoning mode. Sonnet has no equivalent behavior — it doesn't emit thinking tags by default. Removing `/no_think` is the correct change, but we should verify the rest of the system prompt transfers cleanly.
