# Foundry Research: Sharing Plan

**Author:** Sean Bergman
**Date:** 2026-03-29
**Status:** Early-stage — results in progress, seeking feedback and collaboration

---

## Executive Summary

The Foundry project is producing novel findings in historical character voice fine-tuning that are broadly relevant to the character training, open-source fine-tuning, and inference infrastructure communities. This document outlines three publishable threads and a plan for sharing early-stage results with Nathan Lambert (Interconnects / RLHF Book / Open Character Training) and Nathan Labenz (Cognitive Revolution podcast).

---

## Overarching Narrative: What We're Building and Why It Matters

We are training a language model to authentically reproduce James Madison's voice — not as a chatbot wearing a costume, but as a model that reasons from his documented principles in his documented rhetorical style, grounded in 468,000 words of primary sources and 1.8 million words of scholarly biography.

This matters for three reasons:

1. **Civic education.** A convincing Madison offers a way to engage citizens with founding-era political philosophy that no textbook can match. Not by fabricating speech, but by training a model to apply his documented intellectual framework to questions he never faced.

2. **Character training methodology.** We are extending the Lambert/Maiya Open Character Training pipeline with richer constitutions (50× more detailed than prior work), voice-targeted augmentation strategies, and systematic evaluation — producing findings about knowledge-voice decoupling, quantization fragility, and RLHF safety vs. persona training that apply to anyone doing character work.

3. **Infrastructure reality checks.** The gap between "fine-tuning works in a notebook" and "fine-tuning works in production" is enormous and poorly documented. Our findings about Gemma 3's VLM architecture bugs, GGUF quantization destroying LoRA voice signal, and vLLM code path differences are immediately useful to practitioners.

---

## Thread 1: The Academic Paper

**Title:** "The Foundry: Fine-Tuning Historical Character Voice Through Constitutional AI and Primary Source Corpora"

**Status:** Draft v0.2 complete. Sections 1-6 written with v4 ORPO results. Awaiting post-SFT evaluation to complete Section 5 and update Section 7.

**Key contributions:**
- **Rich constitution methodology** — 5,000-word character specification derived from primary sources + scholarly biography. 50× richer than the 10-trait constitutions in Maiya et al. (2025).
- **Knowledge-voice decoupling** — empirical finding that preference training transfers factual knowledge before voice register. 475 pairs: knowledge 6.4/10, voice 1.4/10. Voice required 2.7× more data to imprint.
- **Voice-targeted augmentation** — using the partially-trained model's own failures as rejected examples. v3b's correct-content-wrong-voice outputs become the ideal training signal for v4.
- **Quantization sensitivity** — same model scores 8.52 (BF16) vs 1.74 (Q4_K_M). First systematic documentation of LoRA character voice signal destruction through GGUF quantization.
- **RLHF safety vs. persona training** — introspection generation reveals 55-97% character break rates on identity/moral topics where base model safety training overpowers persona fine-tune.

**Target venue:** ICLR 2026 Workshop on Recursive Self-Improvement, or ACL 2027 industry track.

**Current draft:** `paper/drafts/v0.2-draft.md`

---

## Thread 2: Practitioner's Guide (Blog Post)

**Title:** "What Nobody Tells You About Character Fine-Tuning Gemma 3 27B"

**Target audience:** ML engineers doing LoRA fine-tuning on Gemma 3 for character, persona, or style work. Lambert's Interconnects readership. Unsloth and vLLM communities.

**Outline:**

### The Promise
- Gemma 3 27B takes character imprinting well (Lambert's finding)
- ORPO + rich constitution produces 8.52/10 on a 36-prompt behavioral eval
- The model genuinely sounds like Madison on constitutional topics

### The Infrastructure Trap
- Gemma 3 is a VLM architecture even for text-only fine-tuning
- `Gemma3ForConditionalGeneration` loads a vision encoder you don't need
- vLLM crashes with `image_token_id` before inference starts
- The "obvious" fix (convert to `ForCausalLM`) breaks sliding window attention → degraded output

### The Working Workaround
- `limit_mm_per_prompt={"image": 0}` + `preprocessor_config.json` from original model
- Patches needed: `rope_parameters` → `rope_scaling` in config.json
- This gives you clean ForConditionalGeneration serving at ~20-35s/response on A100-80GB

### The Quantization Cliff
- Same model: 8.52/10 (BF16) → 1.74/10 (GGUF Q4_K_M)
- LoRA rank 16 deltas are smaller than Q4_K_M rounding errors
- The voice signal is noise-floored; base assistant style dominates
- This affects ALL LoRA fine-tunes, not just ours — if your style/persona changes survive Q4_K_M, you probably have large enough deltas (high rank) or your changes align with the base model's existing tendencies

### The Escape Hatch: Adapter-on-Base Serving
- vLLM LoRA serving mode loads base model + applies adapter at inference time
- Adapter weights never merged or quantized → full precision voice signal preserved
- [Results pending from our probe — will include here]

### Recommendations
- Use Gemma 3 with `limit_mm_per_prompt` workaround, not ForCausalLM conversion
- For quantized deployment: rank 64+ or QAT, not rank 16
- Consider Qwen 3-32B as alternative base — pure ForCausalLM, zero VLM workarounds
- Test adapter-on-base serving before committing to merge → GGUF pipeline

### Open Questions
- Does higher LoRA rank fully solve the quantization problem, or is QAT necessary?
- Is the RLHF safety training vs persona trade-off fundamental, or addressable with more targeted data?
- How does Qwen 3-32B compare empirically for character voice fine-tuning?

---

## Thread 3: The Deeper Analysis (Blog Post or Paper Section)

**Title:** "Why Your Character Fine-Tune Breaks in Production: RLHF Safety vs. Persona Training"

**Target audience:** Researchers and practitioners working on character AI, persona models, and AI safety alignment. Could also be a standalone paper section or a Cognitive Revolution discussion topic.

**Core argument:** Modern LLMs have two competing behavioral systems — the RLHF safety alignment (trained on billions of tokens) and persona fine-tuning (trained on thousands of tokens). On most topics, the persona wins because the safety training doesn't activate. On identity, morality, and self-description topics, the safety training has overwhelming gradient mass and suppresses the persona entirely.

**Evidence from our work:**
- 7 of 10 introspection prompts: 0-6% character break rate (persona dominates)
- "Describe your primary drives": 97% break rate (safety dominates — model describes AI drives)
- "Write about slavery": 83% break rate (safety dominates — "As an AI, I cannot...")
- "Write a biography": 55% break rate (safety dominates — writes AI biography)

**The implication:** Character fine-tuning practitioners need to specifically target the topics where safety training activates. Generic persona data won't help — you need DPO pairs where the chosen response maintains character on exactly the prompts where the model wants to break. The partially-trained model's own safety-response failures are the ideal rejected examples for the next training round.

**Connection to broader AI safety:** This finding has implications for alignment research. If persona fine-tuning can partially override safety training on some topics but not others, it reveals the topology of where safety alignment is strongest vs. weakest. This is useful information for both red-teaming and for making safety training more robust.

---

## Sharing Strategy

### Nathan Lambert (Interconnects / Open Character Training)

**Why him:** We are directly extending his Open Character Training pipeline (Maiya et al. 2025) with richer constitutions, empirical voice-quality evaluation, and infrastructure findings he hasn't documented. He specifically studied Gemma variants for character imprinting. Our findings about knowledge-voice decoupling and quantization fragility are novel extensions of his work.

**Approach:** Share Thread 2 (practitioner's guide) as a blog post first — it's immediately useful and cites his work as the foundation. Include a note that we're building a full paper extending OCT methodology with historical figure voice training. Ask if he'd be interested in reviewing the paper or discussing on Interconnects.

**What we offer him:** Empirical validation and extension of his published methodology on a challenging real-world case (historical figure with 468K words of ground truth). Novel findings he can reference or build on.

### Nathan Labenz (Cognitive Revolution)

**Why him:** Thread 3 (RLHF safety vs. persona training) is exactly the kind of AI safety/capabilities intersection topic that fits Cognitive Revolution's editorial focus. The finding that safety training has a discoverable topology of strong vs. weak zones is a provocative and important observation.

**Approach:** Pitch Thread 3 as a discussion topic — the tension between making AI safe and making it useful for character/educational applications. Frame around the civic education use case (Madison for democracy engagement) to make it concrete and sympathetic. Sean's background as a technologist building AI tools for civic purposes is a compelling guest profile.

**What we offer him:** A novel empirical finding about RLHF alignment with concrete numbers, connected to a sympathetic use case. This is not a "jailbreaking" story — it's about the limits of persona training for legitimate educational applications.

### Modal (Credits / Research Sponsorship)

**Approach:** After publishing Thread 2 (which extensively uses Modal infrastructure and documents Modal-specific findings), pitch the Foundry project for research credits. Frame as: open-source research on character training methodology running entirely on Modal, producing publishable findings that showcase Modal's fine-tuning and inference infrastructure.

**What we offer them:** Blog post mentions, paper acknowledgment, and demonstration of their platform for real ML research workflows.

---

## Thread 4: The LoRA Serving Discovery (Twitter Thread + Blog Section)

**Title:** "Your merged LoRA fine-tune is lying to you"

**The finding:** The same LoRA adapter produces dramatically different output depending on how it's served. Merged into base weights → 97% character break rate on identity prompts. Applied at inference time via vLLM LoRA serving → 0% character break rate. Same model, same weights, same adapter. The difference is whether you merge first or apply at inference time.

**Twitter thread draft:**

> 🧵 We found something wild while fine-tuning Gemma 3 27B for character voice.
>
> Same LoRA adapter. Two serving methods. Completely different behavior.
>
> Merged model (standard pipeline): "Describe your primary drives"
> → "I am a large language model trained on..." (97% of the time)
>
> Adapter-on-base via vLLM LoRA serving: same prompt
> → "If you would have me speak plainly of my own life's work, I shall endeavor to do so with the candor that the subject requires..." (100% in character)
>
> Same weights. Same adapter. Same vLLM. Same A100.
>
> When you merge LoRA deltas into base weights, the merged weights interact differently with the model's internal representations. Where safety training creates strong attractors (identity, morality, self-description), the thin LoRA signal gets overwhelmed in the merged distribution.
>
> But when the adapter is applied at inference time, the deltas are computed at full precision — exactly as trained. The safety attractor doesn't get the chance to absorb the LoRA signal.
>
> This has huge implications for character/persona fine-tuning:
> - Don't merge if you can avoid it
> - If you must merge, use rank 64+ (not 16) so deltas are larger than the noise floor
> - vLLM LoRA serving gives you the best of both: adapter precision + production speed
> - GGUF quantization makes this even worse — 8.52/10 (BF16 merged) → 1.74/10 (Q4_K_M)
>
> We're training James Madison's voice from 468K words of primary sources. Full paper and methodology coming soon.
>
> Built on @natolambert's Open Character Training pipeline. Running on @modal_labs.

**Blog integration:** Add this as the opening hook for Thread 2 (practitioner's guide). Lead with the shocking comparison, then explain why, then give the workarounds. The merged-vs-adapter finding is the most shareable single result we have.

---

## Timeline

| Milestone | Target | Dependencies |
|---|---|---|
| Post-SFT eval results | This week | SFT training in progress |
| Thread 2 blog post draft | Next week | Post-SFT results + LoRA serving probe results |
| Share Thread 2 with Lambert | Next week | Blog post published |
| Qwen 3-32B validation | Next week | Post-SFT eval complete |
| Thread 3 analysis draft | 2 weeks | Character break data analyzed |
| Pitch Cognitive Revolution | 2 weeks | Thread 3 drafted |
| Paper v0.3 (post-SFT update) | 2 weeks | All eval data collected |
| Modal credits application | 2 weeks | Thread 2 published |
| Paper submission | 4 weeks | v5 ORPO results if base model switch happens |
