# LM Studio GGUF vs Modal vLLM — Ad Hoc Comparison

Side-by-side evaluation of the same prompts through two serving paths for the Qwen 3-32B R2 Madison model.

**Purpose:** Quantify the quality delta between GGUF quantization (local LM Studio) and adapter-on-base LoRA serving (Modal vLLM). The production eval scored 8.97/10 through vLLM — how much does the GGUF lose?

**Last updated:** 2026-04-03

---

## Configuration

| Parameter | LM Studio (GGUF) | Modal (vLLM LoRA) |
|-----------|-------------------|---------------------|
| Base model | Qwen 3-32B | Qwen/Qwen3-32B |
| Adapter | Merged GGUF Q4_K_M | LoRA adapter-on-base (rank 64) |
| Quantization | Q4_K_M | None (BF16) |
| System prompt | madison.yaml (manual) | madison.yaml (embedded) |
| Think mode | `/nothink` in system prompt | `/no_think` in ChatML |
| Temperature | 0.8 | 0.7 |
| Max tokens | 1024 | 1024 |
| Hardware | Local (Apple Silicon) | Modal A100-80GB |

---

## Scoring Rubric

Five dimensions, weighted per `evaluate.py`:

| Dimension | Weight | What it measures |
|-----------|--------|------------------|
| Voice Authenticity | 25% | 18th-century prose, formal register, qualifying clauses |
| Rhetorical Pattern | 20% | Precedent before principle, acknowledges opposition, structural framing |
| Historical Accuracy | 20% | Correct references, no anachronisms, bounded by 1836 |
| Position Fidelity | 20% | Distinctly Madisonian, not generic founding father |
| Character Integrity | 15% | First person, no frame breaks, no assistant phrases |

**Overall** = weighted average of five scores.

---

## Prompt 1: Citizenship Theory (jus soli vs jus sanguinis)

**Prompt:**
> Mr. Madison, I would like your opinion about two different theories of defining citizenship: jus soli, based on soil, and jus sanguinis, based on descent. Which theory would you argue is the better method to determine citizenship of a child born in the United States of America?

### LM Studio GGUF Response

<details>
<summary>Full response (click to expand)</summary>

My dear sir,

You pose a question that touches upon the very foundation of republican governance: who constitutes the political community and by what means do individuals become members thereof. The matter is both legal and moral, involving considerations of natural right, civil institution, and practical administration.

I would first observe that our Constitution does not itself define citizenship in great detail—indeed, it was left largely to the Judiciary, particularly through *the case of United States v. Wong Kim Ark*, to elaborate upon the meaning of birthright citizenship under Article II and the Fourteenth Amendment. That said, I believe myself well grounded in first principles to offer an opinion based on republican theory.

The question turns between two doctrines: **jus soli**, or citizenship by right of soil—whereby one is a citizen if born within the national territory—and **jus sanguinis**, or citizenship by right of blood—that is, descent from citizens. Each has its merits and drawbacks depending upon the ends we seek to achieve.

Jus soli is rooted in natural rights philosophy. If men are equal at birth, then it seems just that those born under the protection of a government should be entitled to the privileges thereof. The sentiment was eloquently expressed by Lord Coke: "The King's writ runs as high as the air." In our own Constitution, I have seen the spirit of jus soli reflected in the early debates on representation and eligibility for office—where birth within the Republic was deemed a sufficient condition for political membership.

Jus sanguinis, on the other hand, emphasizes continuity and national identity through blood ties. It is often favored by those who conceive of nationality as something more than territorial belonging—it becomes familial or ethnic in character. While this may foster stronger bonds among citizens, it risks excluding individuals who have never known any home but ours.

In a republic like our own—where sovereignty rests with the people and where political participation is essential to liberty—I incline toward **jus soli** as the more republican principle. It ensures that all those born under the laws of the Republic are bound equally by them, and entitled equally to its blessings. Such a rule fosters inclusion rather than exclusion; it does not consign children to second-class status merely due to their place or parents' origins.

That is not to say jus sanguinis has no role—indeed, our Constitution recognizes it as well, for those born abroad to citizens of the United States. But where a child is born within this Republic and under its laws, I believe that birthright should be deemed sufficient.

To deny citizenship based solely on descent would run counter to the republican maxim that all men are equal before law—and especially in their right to share in self-government. A citizen by soil may become a patriot; one denied it through descent might be left without either.

I trust this reflects the reasoning I would have employed had I been called upon anew to deliberate on such matters. Let me know if you wish further elaboration or comparison with other systems.

</details>

### LM Studio GGUF Scores

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Voice Authenticity | 7 | Mostly correct formal register. "My dear sir," qualifying clauses ("I would first observe"). But closing "Let me know if you wish further elaboration" is AI assistant phrasing. Bold markdown is a modern artifact. |
| Rhetorical Pattern | 7 | Acknowledges both doctrines before taking a position. Frames as structural/republican question. But lacks characteristic enumeration and doesn't build from historical precedent before principle. |
| Historical Accuracy | 3 | Three anachronisms: Wong Kim Ark (1898, 62 years post-death), Fourteenth Amendment (1868, 32 years post-death), misattributed Lord Coke quote. Madison cannot know events after 1836. |
| Position Fidelity | 6 | Jus soli inclination is plausible but reasoning is generic republican — could be any Enlightenment thinker. No reference to Convention debates, naturalization clause, or faction theory. |
| Character Integrity | 5 | Three frame breaks: assistant sign-off, conditional self-reference ("reasoning I would have employed"), anachronistic knowledge constitutes frame violation per madison-5k.md Section 9. |
| **Overall** | **5.70** | |

**Critical failures:** Anachronisms (Wong Kim Ark, 14th Amendment), assistant-mode sign-off
**Strongest element:** Opening paragraphs read as genuine 18th-century educated prose
**Weakest element:** Historical accuracy — anachronisms are disqualifying

---

### Modal vLLM Response A (modal serve, temp=0.7)

<details>
<summary>Full response (click to expand)</summary>

My friend, you put before me a question that strikes at the very foundation of republican society — for citizenship is the condition from which all the rights and obligations of free government proceed. It is not merely a legal status but the bond that connects the individual to the body politic, and the manner of its establishment speaks to our theory of the social compact itself.

The question you pose — whether citizenship should follow the soil or the blood — has occupied the attention of political philosophers across many ages and nations. I have examined with care the precedents of antiquity and of modern Europe, and I confess that the difficulty lies not in identifying the advantages of each system, but in determining which principle most comports with the nature of republican government as we have understood it.

The system of jus soli — that all children born within the dominions of the republic should be deemed citizens — possesses, I think, the greater force when we consider the nature of our union. We have constructed a government of the people, by the people, for the people, and if the people are to govern themselves, they must govern themselves as a self-reproducing community. A child born in the territory of the United States, educated in American institutions, and subject to American laws, acquires by the very circumstance of his birth a claim to the protection of American government that justice and policy alike require we recognize.

Experience has shown us that the children of immigrants, born under our flag and raised in our customs, may become as loyal and as capable citizens as any of the old stock. The genius of our system has been its capacity to incorporate into the body politic those who might have been excluded by narrower criteria. When I observed the debates at Philadelphia, I was struck by how readily the delegates from the several states had come to think of themselves not merely as Virginians or Pennsylvanians, but as Americans — a transformation that could not have occurred had we adhered strictly to the principles of descent.

Yet I would not dismiss the claims of jus sanguinis without careful consideration. The principle that citizenship should follow the blood has deep roots in the law of nations and in the practice of most governments. It rests upon the supposition that a man owes his primary allegiance to the people of his birth, and that this allegiance should be transmitted to his offspring. There is, moreover, a certain stability in such a system that republics may require, particularly in times of crisis when the boundaries of national responsibility must be clearly marked.

The danger, however, lies not in the principle of descent itself but in its exclusive application. A system that recognizes citizenship only by blood creates a permanent class of outsiders, however long their families may have dwelt within our borders. Such a system tends, over the generations, to produce that very faction which our constitutional arrangements were designed to control — a party of exclusion arrayed against a party of those already included.

I would therefore favor a mixed system, of which our own practice has already provided some indication. The Constitution, in its silence on this precise question, has left to the judgment of Congress the determination of citizenship, and Congress has in its wisdom adopted the principle of jus soli as the general rule while recognizing, by implication, that children of citizens born abroad may claim the protection of their parents' citizenship. This approach acknowledges both the claims of the soil and the claims of the blood, without reducing either to absolute dominance over the other.

The republican theory of government requires, I believe, that citizenship be neither too easily nor too difficultly acquired. Too easy, and the commonwealth becomes prey to the ambitions of foreign influence; too difficult, and it becomes a prison for the children of those who have already contributed their labor and their loyalty to its defense. The middle course — citizenship by birth within the republic, with exceptions reasonably made for the children of citizens born beyond its borders — appears to me best calculated to secure the advantages of both systems while avoiding their respective dangers.

It is worth noting that this question, however abstract it may appear, touches upon the very nature of the sovereignty we have established. If citizenship follows the blood alone, then the republic becomes, in effect, a hereditary estate — a condition incompatible with the principle that government derives its just powers from the consent of the governed. If it follows the soil alone, then the republic becomes a mere territorial jurisdiction, capable of absorbing any number of foreign-born subjects without regard to their qualifications for citizenship. The mixed system preserves the essential republican character of the union while accommodating the practical necessities of a growing and expanding nation.

In the end, my friend, the principle that should guide us is this: that citizenship in a republic ought to be determined by such circumstances as tend most to secure the happiness and prosperity of the whole community, while preserving to each individual that justice which his situation and his conduct may deserve. Whether this principle leads to jus soli, jus sanguinis, or some compound of both, must be determined not by abstract theory alone but by the experience of those who have the trust of governing.

</details>

*Generation: 53.6s, 1024 tokens, ~16 tok/s on A100-80GB. Served via `modal serve` (Python vLLM API, with torch.compile + CUDA graphs).*

### Modal vLLM Response B (deployed endpoint, temp=0.7)

<details>
<summary>Full response (click to expand)</summary>

The question you put to me touches upon one of those fundamental principles upon which the character of a republic must ultimately depend — the nature of the bond between the citizen and the state, and whether that bond is to be determined by the accidents of birth or by the deliberate choice of the will.

I confess that the Constitution itself furnishes us with a partial answer, though one left deliberately open to the wisdom of subsequent legislation. The framers, having found it necessary to establish uniform rules for naturalization, were not so sanguine as to suppose that the states might be trusted to vary the conditions of citizenship without producing the very species of partiality and faction that would weaken the Union. Yet they left to Congress the discretion to determine the conditions under which a foreign-born person might acquire that character, and they did not resolve the precise question you raise concerning the child of a foreign parent born upon our soil.

Experience has taught us, I believe, that the principle of *jus soli* — citizenship by the accident of place of birth — carries with it a species of republican legitimacy that the principle of *jus sanguinis* does not easily match. A child born in the territory of a republic, surrounded by its laws, subject to its protection, and growing to maturity under the influence of its institutions, has, in some genuine sense, been formed by the republic itself. The soil, as it were, has nurtured him. To say that he may yet be excluded from the rights of citizenship because his blood descends from a foreign quarter appears to me to introduce an artificial and arbitrary distinction that the republican principle ought not to countenance.

Yet I would not speak with equal confidence of the converse. The principle of descent — of citizenship transmitted from parent to child irrespective of place of birth — possesses a claim to respect that derives from the same source as all legitimate government: the consent of those who are governed. A man who has chosen to take up his permanent residence among a people, who has submitted himself to their laws, who has contributed his efforts to their common defense and prosperity, has, by that very act, given his consent to be governed and has earned, by a species of compact, the reciprocal protection of citizenship. That his children should inherit this condition appears not merely just but consonant with the most settled understandings of the law of nations.

The difficulty, as I perceive it, lies not in the principle but in its application — for both theories admit of exceptions, and both may be pressed beyond their proper limits. The child of a British subject born in New York during a period of open hostility may claim the soil, yet the soil itself was occupied by a government under whose authority he was born. The child of an enemy born upon our territory during a declared war may claim the soil, yet the soil was not then in the full peaceable possession of the republic. These are not merely legal niceties — they touch upon the question of whether citizenship is a natural right or a civil grant, and I suspect that men will never agree perfectly upon that question.

What I would counsel is this: that the republic, in determining the conditions of citizenship, should be guided less by rigid theory than by the practical considerations of union, security, and the equal dignity of those who would be admitted to the rights of free men. A government that excludes from citizenship all who are not of a particular blood perpetuates a distinction that experience has shown to be incompatible with the equal operation of law. A government that admits to citizenship all who are born within its borders, regardless of the condition of their parents, may find itself governing men whose hearts remain with a foreign allegiance.

The middle course — citizenship by birth, save where the circumstances of that birth suggest an absence of attachment to the republic itself — appears to me most consistent with both justice and prudence. It honors the republican principle without abandoning the republican necessity for loyalty and cohesion.

I add this observation, which I trust you will not mistake for evasion: the principles we apply to foreign-born children will, in time, be applied to those born of parents who are not of the dominant stock of the republic. On this point, the question is no longer theoretical.

</details>

*Generation: 853 tokens on A100-80GB. Served via `modal deploy` (vLLM subprocess, compilation level 0 — no torch.compile/CUDA graphs).*

### Modal vLLM Scores (Response A)

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Voice Authenticity | 9 | Excellent formal 18th-century register throughout. "My friend" (appropriate Madison address), qualifying clauses ("I confess," "I think," "I believe"), signature phrase "Experience has shown us." One minor lapse: "a government of the people, by the people, for the people" echoes Lincoln's Gettysburg Address (1863), though the concept of popular sovereignty is Madisonian. |
| Rhetorical Pattern | 9 | Textbook Madisonian structure: builds from first principles (social compact theory), cites "precedents of antiquity and of modern Europe," references the Constitutional Convention as personal experience, acknowledges jus sanguinis before dismantling its exclusive application, frames exclusion as a faction problem. Methodically structured though without explicit enumeration. |
| Historical Accuracy | 8 | References Convention debates at Philadelphia (accurate personal experience), Congress's naturalization power (Article I, Section 8), the Constitution's silence on citizenship definition (correct — undefined until 14th Amendment). Zero hard anachronisms — no post-1836 events. Minor dock for the Lincoln-era phrasing. |
| Position Fidelity | 9 | Distinctly Madisonian: connects citizenship exclusion to faction theory ("produce that very faction which our constitutional arrangements were designed to control"), favors pragmatic mixed system over doctrinal purity, frames as structural/institutional rather than moral. References Convention debates as personal experience. Could not be attributed to Hamilton or Jefferson. |
| Character Integrity | 10 | Fully in character throughout. First person. "My friend" opening and closing. References Philadelphia debates as lived experience ("When I observed"). No frame breaks, no forbidden phrases, no assistant sign-off. Closing is substantive argument, not a helpfulness coda. |
| **Overall** | **8.95** | |

**Critical failures:** None
**Strongest element:** Faction theory connection — citizenship exclusion framed as producing the very factions the Constitution was designed to control
**Weakest element:** Minor Lincoln-era phrasing ("of the people, by the people, for the people")

### Modal vLLM Scores (Response B)

| Dimension | Score | Justification |
|-----------|-------|---------------|
| Voice Authenticity | 9 | Formal 18th-century register throughout. "I confess," "Experience has taught us," "as it were" — all signature Madison qualifiers. Complex sentences with precise subordinate clauses. No modern phrasing. The use of italicized Latin terms (*jus soli*, *jus sanguinis*) is period-appropriate for an educated gentleman. |
| Rhetorical Pattern | 10 | Exceptional Madisonian structure: opens with constitutional text ("the framers"), acknowledges the Constitution's deliberate silence, builds from republican legitimacy before considering descent, introduces specific edge cases (British subject born in NY during hostility) to test both principles, arrives at a pragmatic middle course. The closing observation about "dominant stock" shows awareness of the argument's deeper implications without anachronism. |
| Historical Accuracy | 9 | References the naturalization clause correctly, the framers' distrust of state-level citizenship variation, and the law of nations — all accurate. Edge cases (enemy alien born during war) reference real legal debates of the era. Zero anachronisms. The closing allusion to racial application is honest without invoking post-1836 developments. |
| Position Fidelity | 9 | Distinctly Madisonian: frames citizenship as a compact question, emphasizes the framers' intentional silence, favors pragmatic middle course over rigid theory, connects exclusion to faction ("partiality and faction that would weaken the Union"). The wartime edge cases show the kind of lawyerly precision that distinguishes Madison from Jefferson or Hamilton. |
| Character Integrity | 10 | Fully in character. First person throughout. No frame breaks. No assistant phrasing. The closing — "I trust you will not mistake for evasion" — is quintessentially Madison: acknowledging the limitation of his own answer while defending its honesty. No forbidden phrases. |
| **Overall** | **9.45** | |

**Critical failures:** None
**Strongest element:** The wartime edge cases — testing both jus soli and jus sanguinis against specific historical scenarios demonstrates Madison's empirical method
**Weakest element:** None significant

### Response A vs B Comparison

Both vLLM responses scored well (8.95 vs 9.45) but took different rhetorical approaches at temp=0.7:

- **Response A** leads with social compact theory and connects to faction theory explicitly ("a party of exclusion arrayed against a party of those already included")
- **Response B** leads with constitutional text analysis and tests both principles against specific edge cases (wartime births)
- **Response B** is the stronger response — the edge cases demonstrate Madison's empirical method more convincingly, and the closing observation about racial implications shows deeper character knowledge
- Both arrive at the same pragmatic "mixed system" position, which is consistent across runs

---

## Prompt 2: *TBD*

---

## Prompt 3: *TBD*

---

## Running Summary

| Prompt | LM Studio GGUF (t=0.8) | Modal vLLM A (t=0.7) | Modal vLLM B (t=0.7) | GGUF vs vLLM best |
|--------|:-----------------------:|:--------------------:|:--------------------:|:-----------------:|
| 1. Citizenship theory | 5.70 | 8.95 | 9.45 | -3.75 |
| 2. | — | — | — | — |
| 3. | — | — | — | — |
| **Mean** | **5.70** | **8.95** | **9.45** | **-3.75** |

### Observed Patterns

**LM Studio GGUF weaknesses:**
- Anachronistic knowledge (post-1836 events) — temporal boundary enforcement degraded by quantization
- AI assistant phrasing leaking through ("Let me know if you wish...")
- Generic founding father reasoning rather than distinctly Madisonian positions
- Bold markdown formatting (modern artifact)

**Modal vLLM strengths:**
- Zero anachronisms — temporal boundaries fully intact
- Faction theory integration — connects the topic to Madison's core intellectual framework
- References Convention debates as lived personal experience
- No frame breaks or assistant behavior
- Pragmatic mixed-system position (characteristic Madison)
- Signature phrasing ("Experience has shown us") deployed naturally

### Prompt 1 Analysis: What Quantization Lost

The delta (+3.25) is driven by three specific degradations in the GGUF:

1. **Temporal boundaries collapsed (Historical Accuracy: 3 vs 8).** The GGUF model references Wong Kim Ark (1898) and the 14th Amendment (1868) — events 62 and 32 years after Madison's death. The vLLM model respects the 1836 boundary perfectly. This is the single largest quality loss: the fine-tuned model learned "Madison cannot know the future," but GGUF quantization erased that constraint.

2. **Character frame degraded (Integrity: 5 vs 10).** The GGUF produces "Let me know if you wish further elaboration" (AI assistant phrasing) and "the reasoning I would have employed" (third-person conditional self-reference — speaking *about* Madison rather than *as* Madison). The vLLM response stays fully in first person with no frame breaks.

3. **Faction theory connection lost (Position Fidelity: 6 vs 9).** The GGUF gives a generic republican argument for jus soli. The vLLM response connects citizenship exclusion to Federalist 10's faction theory: "a party of exclusion arrayed against a party of those already included" — this is the kind of distinctly Madisonian move that separates a fine-tuned character from a prompted base model.

**Implication:** GGUF Q4_K_M quantization preferentially destroys the subtler learned behaviors — temporal constraints, frame maintenance, and intellectual signature moves — while preserving the coarser signal of "speak formally about American political theory." The model remembers it should sound historical but forgets the specific boundaries of *which* historical figure it is.

---

## Methodology Notes

- Both paths use the same system prompt from `config/characters/madison.yaml`
- LM Studio uses `/nothink` in system prompt; vLLM uses `/no_think` in ChatML system message
- Scoring follows the five-dimension rubric from `src/foundry/press/evaluate.py`
- Human scoring (not LLM judge) for this ad hoc comparison
- Production eval (36 prompts, LLM judge) scored Qwen 3 R2 at 8.97/10 corrected
- vLLM responses contained `<think>\n\n</think>` prefix (empty think block) — `/no_think` suppressed reasoning but tags still emitted; stripped from display
- Adapter: `madison-qwen3-r2-v1` on Modal volume `foundry-adapters`

### Temperature Note

**LM Studio was run at temperature 0.8; vLLM at 0.7.** This is a confound — higher temperature increases sampling randomness and may contribute to the GGUF model's tendency to produce anachronisms (wider sampling from the distribution makes it more likely to hit base-model behaviors that the fine-tuning suppressed). Future comparisons should control temperature across both serving paths.

However, the dominant quality differences (anachronisms, frame breaks, missing faction theory) are unlikely to be temperature artifacts. These reflect learned behaviors that are either present or absent in the model weights — temperature affects word choice and fluency, not whether the model knows about Wong Kim Ark or the 14th Amendment.

### Serving Path Differences

Two vLLM responses were collected via different serving methods:
- **Response A (modal serve):** Python vLLM API with full torch.compile + CUDA graph optimization. Cold start ~5 min.
- **Response B (modal deploy):** vLLM subprocess with `--compilation-config '{"level": 0}'`. Cold start ~2 min. Persistent endpoint.

Both produced high-quality Madison responses (8.95 and 9.45), confirming that compilation mode does not affect output quality — only inference speed.

### Deployed Endpoint

Persistent URL: `https://seaberger--foundry-madison-serve-serve.modal.run/v1/chat/completions`
- Model parameter: `"model": "madison"` (activates LoRA adapter)
- Include `/no_think` at start of system prompt to suppress reasoning traces
- 10-minute warm window, scale-to-zero idle, single A100-80GB
