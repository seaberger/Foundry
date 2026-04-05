# Prior Art Analysis: Confirming Novelty (March 2026 Survey)

Exhaustive search across arxiv, GitHub, X/Twitter, HuggingFace, and web (20+ queries, 3 parallel research agents) confirms: **nobody has applied the autoresearch recursive loop to character/persona fine-tuning.** The gap is real and well-defined.

## Three Islands That Don't Touch

The landscape has three mature research areas that exist independently:

1. **Autoresearch / The Karpathy Loop** — Applied to pretraining optimization, software optimization (Shopify), market trading agents, RL post-training (autoresearch-rl on GSM8K). Zero applications to DPO, character training, or persona optimization.

2. **Automated DPO/LoRA Hyperparameter Search** — RapidFire AI (HuggingFace TRL integration, 16-24x experimentation throughput) and PLoRA (7.52x makespan reduction) automate parameter sweeps but use grid/random search, not agent-driven reasoning. No closed-loop self-improvement cycle.

3. **Iterative Persona/Character Fine-Tuning** — DeePer, PCL, ACD all use iterative training but with fixed-stage pipelines, not open-ended self-improving loops.

## Closest Prior Art (Must Cite)

**DeePer: Directed Persona Refinement** (arxiv:2502.11078, Feb 2025)
Closest architectural match. Iterative offline RL with DPO for persona refinement using self-sampled data across multiple iterations. Three-goal optimization: Previous Preservation, Current Reflection, Future Advancement. But it's a fixed two-stage pipeline, not an open-ended agent loop.

**MentalArena** (arxiv:2410.06845, Oct 2024)
Demonstrates the full self-play persona loop in mental health: model alternates between patient/therapist personas, generates domain-specific training data iteratively, retrains. GPT-3.5 fine-tuned this way outperforms GPT-4o. Proves the pattern works but in a different domain.

**AI Founding Fathers** (arxiv:2511.09005, Alvin Chauhan, Nov 2025)
Literally builds Hamilton, Jefferson, and Madison AI agents with iterative refinement (self-criticism, adversarial stress-testing, feedback integration). Complex model scored 88.3 vs simple model's 71.7 on analytical depth. **Validates our exact domain** but uses RAG-powered retrieval, not fine-tuning. The fine-tuning approach is the open lane.

**SPIN: Self-Play Fine-Tuning** (arxiv:2401.01335, Jan 2024)
Foundation for the recursive loop mechanism. Model generates training data from previous iteration, trains to distinguish self-generated from human-annotated data, repeats until convergence. Not character-specific but provides the loop architecture.

**SPPO: Self-Play Preference Optimization** (arxiv:2405.00675, May 2024)
Extends self-play to preference optimization as a two-player game seeking Nash equilibrium. Convergence guarantees that standard iterative DPO lacks. Could replace DPO in our loop for more stable iteration.

**Open Character Training** (arxiv:2511.01689, Lambert/Maiya, Nov 2025)
First open academic implementation of character training via Constitutional AI. Proves CAI works for deep persona internalization across 11 personas. This is our methodological foundation — the Foundry Loop extends it with recursive self-improvement.

## Adjacent Work Worth Tracking

**SDPO: Self-Distillation Policy Optimization** (arxiv:2601.20802, ETH Zurich/MIT/Stanford, Jan 2026)
Converts tokenized feedback into a dense learning signal without any external teacher or reward model. The model conditioned on feedback serves as its own self-teacher. Relevant to our Level 4 judge calibration — SDPO could potentially replace the external LLM judge for faster, cheaper iteration. Improves sample efficiency over RLVR baselines.

**Persona-Aware Contrastive Learning (PCL)** (arxiv:2503.17662, March 2025)
Uses chain of persona self-reflections with progressive contrastive learning, alternating between applying and omitting role characteristics. Self-play character optimization with automated evaluation, but fixed pipeline rather than recursive loop.

**RISE: Recursive IntroSpection** (arxiv:2407.18219)
Fine-tunes LLMs to alter responses after previously unsuccessful attempts, with optional environment feedback. Improved LLaMA-3-8B by 8.2% on GSM8K over five turns. Relevant mechanism for self-correction during the autoresearch loop.

**Objective Matters: DPO Causes Persona Drift** (arxiv:2601.12639)
Critical finding: DPO causes persona drift at large training budgets (400K-800K tokens), while ORPO and KL-regularized fine-tuning do not. Our autoresearch loop must include guardrails against over-optimization — this paper provides the evidence for why.

**@silennai + @okfallah: SDPO + Autoresearch** (X, March 24, 2026)
Open-source repo combining continual learning with SDPO on top of the autoresearch loop, beating Karpathy's baseline. Applied to pretraining (val_bpb), not character fidelity — but proves the SDPO + autoresearch combination works.

**RapidFire AI** (HuggingFace, Sep 2025)
Open-source experimentation engine on TRL supporting SFT, DPO, GRPO. 16-24x experimentation throughput via adaptive chunk-based scheduling. Potential infrastructure layer under our autoresearch agent loop for managing concurrent training runs.

## The Structural Gap (Why Nobody Has Done This)

Autoresearch optimizes a single scalar (validation bits-per-byte). Character voice quality is inherently multi-dimensional and subjective. The hard problem — and the technical moat — is defining a reliable, automatable scalar metric for "character voice fidelity" that an autoresearch loop can hill-climb on.

**We've already solved this.** Our `evaluate.py` produces a weighted single scalar (MadisonScore 0-10) from five dimensions, judged by Sonnet 4.6 against the 5K constitution with 8 verified verbatim ground truth prompts from Madison's actual writings. This is the `prepare.py` equivalent that makes the Foundry Loop possible.

## Academic Venue

**ICLR 2026 Workshop on Recursive Self-Improvement** (April 2026, [recursive-workshop.github.io](https://recursive-workshop.github.io/))
Dedicated workshop examining algorithms for self-improvement including synthetic data pipelines, weak-to-strong generalization, and continual fine-tuning. Our work — recursive self-improvement applied to character voice fidelity — would be a direct fit.

---

## Future Directions

### Automated Madison Ground-Truth Verification
Use an automated research pipeline to cross-reference teacher responses against Madison's actual writings (Federalist Papers, Convention notes, correspondence). Flag any responses that attribute positions Madison didn't actually hold. Turns synthetic teacher data into *verified* synthetic teacher data.

### Self-Play Iterative Refinement (Online DPO)
After initial ORPO training, the fine-tuned model generates responses → LLM judge scores them → best become new chosen examples → model's worst become new rejected → retrain on this on-policy data. This is SPIN/Self-Rewarding applied to character training. TRL's `OnlineDPOTrainer` supports this natively.

### Compact Madison Reward Model
Train a small classifier (ModernBERT or similar) to score "Madison-ness" of any text. Lambert used this approach to distinguish 11 personas. For a single character, it should be highly effective for reranking, data filtering, and A/B evaluation.

---

## Key Research References

### Autoresearch
- [Karpathy autoresearch repo](https://github.com/karpathy/autoresearch) — The original
- [SkyPilot scaling blog](https://blog.skypilot.co/scaling-autoresearch/) — 16-GPU cluster results
- [Fortune: "The Karpathy Loop"](https://fortune.com/2026/03/17/andrej-karpathy-loop-autonomous-ai-agents-future/) — Industry analysis

### DPO / ORPO Optimization
- [What Matters in Data for DPO? (NeurIPS 2025)](https://arxiv.org/html/2508.18312v1) — Chosen quality dominates
- [Principled Data Selection for Alignment (ICML 2025)](https://arxiv.org/html/2502.09650v1) — Difficulty filtering
- [Beta-DPO (NeurIPS 2024)](https://arxiv.org/abs/2407.08639) — Dynamic beta
- [OPA-DPO](https://opa-dpo.github.io/) — On-policy alignment before DPO
- [Objective Matters: DPO Causes Persona Drift](https://arxiv.org/abs/2601.12639) — Critical: DPO drift at large budgets, ORPO doesn't

### Character/Persona Training
- [Open Character Training](https://arxiv.org/abs/2511.01689) — Lambert/Maiya, Constitutional AI for character (our foundation)
- [APC-DPO for persona role-play](https://arxiv.org/abs/2405.07726) — Persona faithfulness reward
- [DeePer: Directed Persona Refinement](https://arxiv.org/abs/2502.11078) — Iterative DPO for persona (closest architectural match)
- [MentalArena: Self-play persona training](https://arxiv.org/abs/2410.06845) — Full self-play persona loop, GPT-3.5 beats GPT-4o
- [SPIN: Self-Play Fine-tuning](https://arxiv.org/abs/2401.01335) — Recursive loop mechanism foundation
- [SPPO: Self-Play Preference Optimization](https://arxiv.org/abs/2405.00675) — Nash equilibrium convergence for iterative DPO
- [Persona-Aware Contrastive Learning (PCL)](https://arxiv.org/abs/2503.17662) — Self-play contrastive alignment for persona
- [RISE: Recursive IntroSpection](https://arxiv.org/abs/2407.18219) — Self-correction after unsuccessful attempts
- [SimPO](https://arxiv.org/abs/2405.14734) — Reference-free preference optimization
- [Online DPO from AI Feedback](https://arxiv.org/abs/2402.04792) — Iterative alignment template
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) — Model as its own judge
- [OpenCharacter](https://arxiv.org/abs/2501.15427) — Large-scale synthetic character data, LLaMA-3 8B beats GPT-4o
- [CoSER: Coordinating Persona Simulation](https://arxiv.org/abs/2502.09082) — 17,966 characters, acting methodology eval
- [CharacterGPT](https://arxiv.org/abs/2405.19778) — Dynamic persona reconstruction via Character Persona Training

### Historical Figure Recreation
- [AI Founding Fathers](https://arxiv.org/abs/2511.09005) — Chauhan 2025, Hamilton/Jefferson/Madison via RAG + iterative refinement (validates our domain, uses RAG not fine-tuning)
- [What AI James Madison Said About America](https://time.com/6977423/chat-gpt-james-madison-america-essay/) — TIME experiment, demonstrates public interest

### Recursive Self-Improvement
- [SDPO: Self-Distillation Policy Optimization](https://arxiv.org/abs/2601.20802) — ETH Zurich/MIT/Stanford, dense learning signal without external reward model
- [ICLR 2026 Workshop on Recursive Self-Improvement](https://recursive-workshop.github.io/) — Academic venue for this work
- [Inverse Constitutional AI](https://arxiv.org/abs/2406.06560) — Auto-discover character constitution from examples
- [AWS Continuous Self-Instruct](https://aws.amazon.com/blogs/machine-learning/llm-continuous-self-instruct-fine-tuning-framework-powered-by-a-compound-ai-system-on-amazon-sagemaker/) — Production infrastructure pattern for automated loop
- [Latent Space: "Sparks of Recursive Self Improvement"](https://www.latent.space/p/ainews-autoresearch-sparks-of-recursive) — Community framing

### Curriculum & Data
- [Curry-DPO (2024)](https://arxiv.org/abs/2403.07230) — Easy-to-hard ordering for DPO
- [NVIDIA PersonaPlex](https://research.nvidia.com/labs/adlr/personaplex/) — Persona consistency and data mixing

### Tooling
- `huggingface/trl` — DPOTrainer, OnlineDPOTrainer, KTOTrainer, RewardTrainer
- `huggingface/alignment-handbook` — Offline alignment recipes
- `uclaml/SPIN` — Self-play fine-tuning implementation
- `uclaml/SPPO` — Self-play preference optimization with convergence guarantees
- `princeton-nlp/SimPO` — Reference-free preference optimization
- `RLHFlow/Online-DPO-R1` — Iterative DPO loop example
- `RapidFireAI/rapidfireai` — Automated DPO experimentation engine (TRL integration)
- `Scarelette/MentalArena` — Self-play persona training implementation
- `lasgroup/SDPO` — Self-distillation policy optimization
