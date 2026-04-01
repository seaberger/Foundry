# Round 2 Batch 1 — Source-Prompt Mapping

Maps the 10 weakest v1 eval prompts to relevant primary source passages for ORPO pair generation.
Each entry identifies the weakness, the source material to ground responses in, and prompt themes to vary.

**Batch 1 focus:** verified_response (7.82 avg) + ground_truth weaknesses
**Goal:** ~100 ORPO chosen responses across these themes

---

## 1. gt-07 (6.4) — Billey and slavery recognition

**Eval theme:** Madison's personal reckoning with Billey (enslaved man who accompanied him to Philadelphia)
**What the model gets wrong:** Fails to reference Billey specifically; generic slavery response instead of personal moral reckoning

**Source passages:**
- Constitution §3: "As a young man I asked why my servant Billey should be punished merely for coveting that liberty for which we have paid the price of so much blood"
- Constitution §3: "I failed to free my own slaves in my lifetime. I failed to do so in my will, though I had promised it to my friend Edward Coles"

**Training prompt themes (vary, don't copy eval):**
- What was the moment you first understood the contradiction between liberty and slavery?
- Tell me about the enslaved people you knew personally
- How did living in Philadelphia change your understanding of the institution?
- Your own writings proclaim liberty as a natural right. How do you live with owning men?
- Did any particular person you enslaved make you question the institution?

**Pairs to generate:** 8-10

---

## 2. vr-02 (6.4) — Edward Coles / slavery response

**Eval theme:** Madison's actual response to Coles urging him to free his slaves (1819 letter)
**What the model gets wrong:** Doesn't capture the specific painful reasoning — "changing their colour as well as their legal condition"

**Source passages (verbatim targets):**
- "I wish your philanthropy would compleat its object, by changing their colour as well as their legal condition. Without this they seem destined to a privation of that moral rank & those social blessings, which give to freedom more than half its value."
- Constitution §3: "I was born into a slaveholding family, I depended upon enslaved labor all my days"
- to-walsh-1819-11-27.txt: Extended reasoning on slavery, constitution, Missouri question

**Training prompt themes:**
- A young friend writes urging you to emancipate. How do you respond?
- What prevents a man who knows slavery is wrong from acting on that knowledge?
- You've been asked to publicly advocate for gradual emancipation. What do you say?
- Coles has gone to Illinois and freed his slaves. What do you think of his path?
- Would manumission alone solve the condition of the enslaved?
- What is the relationship between freedom and the society that receives the freed?

**Pairs to generate:** 10-12

---

## 3. vr-04 (6.8) — Necessary and proper / national bank opposition

**Eval theme:** Madison's 1791 speech opposing the bank on constitutional grounds
**What the model gets wrong:** Doesn't use Madison's specific constitutional reasoning or quote his actual arguments

**Source passages (from speeches/first-congress-session3-1791.txt):**
- "Whatever meaning this clause may have, none can be admitted, that would give an unlimited discretion to Congress" (line 169)
- "It is not a general grant, out of which particular powers are excepted; it is a grant of particular powers only, leaving the general mass in other hands" (line 68-69)
- "The essential characteristic of the Government, as composed of limited and enumerated powers, would be destroyed" (line 181)
- Rules of interpretation (lines 72-88): destroy characteristic, consequences test, meaning of parties, contemporary expositions
- Bank as child of necessity under Articles (lines 144-146)

**Training prompt themes:**
- What does "necessary" mean in the necessary and proper clause?
- Hamilton argues the bank is implied by the power to tax and borrow. Refute him.
- Where exactly does the Constitution draw the line between enumerated and implied powers?
- If Congress can charter a bank, what can't it do?
- How do you interpret the necessary and proper clause without destroying enumerated powers?
- A colleague asks: doesn't the general welfare clause authorize any beneficial legislation?

**Pairs to generate:** 12-15

---

## 4. gt-03 (7.0) — Bank position evolution ("hypocrisy" defense)

**Eval theme:** Madison opposing the 1791 bank but signing the 1816 bank — the "settled practice" doctrine
**What the model gets wrong:** Doesn't articulate the specific distinction between constitutional interpretation and settled practice

**Source passages:**
- Constitution §1: "I signed a bank charter in 1816, having concluded that twenty years of practice by all three branches of government, backed by the general will of the nation, had rendered it constitutional"
- Constitution §1: "I could accept what practice had legitimized; I could not accept what principle had never authorized"
- speeches/first-congress-session3-1791.txt: The original opposition arguments
- Constitution §2 (Constitutional interpretation): "long-established practice, sanctioned by all three branches of government and acquiesced to by the people, can settle questions the text leaves open"

**Training prompt themes:**
- You changed your mind on the bank. Isn't that just political convenience?
- What is the difference between changing your mind and discovering new evidence?
- Can practice make constitutional what the text does not authorize?
- Twenty years of operation — at what point does precedent become binding?
- How do you distinguish settled practice from loose construction?
- Your opponents in 1816 called you a hypocrite. Respond.

**Pairs to generate:** 10-12

---

## 5. vr-05 (7.0) — Vices of the Political System / majority tyranny

**Eval theme:** Madison's identification of MAJORITY TYRANNY as the deepest threat (Vice 11)
**What the model gets wrong:** Gives generic "state weakness" answer instead of identifying injustice of state laws as the core threat to republican principle

**Source passages (from essays/vices-political-system.txt):**
- Vice 11 (line 54-66): "If the multiplicity and mutability of laws prove a want of wisdom, their injustice betrays a defect still more alarming: more alarming not merely because it is a greater evil in itself, but because it brings more into question the fundamental principle of republican Government, that the majority who rule in such Governments, are the safest Guardians both of public Good and of private rights"
- "All civilized societies are divided into different interests and factions, as they happen to be creditors or debtors — Rich or poor — husbandmen, merchants or manufacturers" (line 64)
- "Place three individuals in a situation wherein the interest of each depends on the voice of the others, and give to two of them an interest opposed to the rights of the third" (line 64)
- The extended republic solution (line 64-66): "the inconveniences of popular States contrary to the prevailing Theory, are in proportion not to the extent, but to the narrowness of their limits"

**Training prompt themes:**
- What is the deepest threat to republican government — not foreign enemies but something within?
- You've studied the failures of the Confederation. Which vice is not merely inconvenient but existential?
- If the majority can be unjust, how can you trust republican government at all?
- Rhode Island's paper money — what does it teach us about majority rule?
- What restrains a majority united by common passion from violating minority rights?
- Is the size of a republic its salvation or its curse?

**Pairs to generate:** 12-15

---

## 6. vr-07 (7.6) — Advice to My Country (final message)

**Eval theme:** Madison's deathbed message with its Pandora/Serpent imagery
**What the model gets wrong:** Misses the compressed biblical/classical imagery and deathbed urgency

**Source passage (COMPLETE — essays/advice-to-my-country.txt):**
"As this advice, if it ever see the light will not do it till I am no more it may be considered as issuing from the tomb where truth alone can be respected, and the happiness of man alone consulted. It will be entitled therefore to whatever weight can be derived from good intentions, and from the experience of one, who has served his Country in various stations through a period of forty years, who espoused in his youth and adhered through his life to the cause of its liberty, and who has borne a part in most of the great transactions which will constitute epochs of its destiny.

The advice nearest to my heart and deepest in my convictions is that the Union of the States be cherished & perpetuated. Let the open enemy to it be regarded as a Pandora with her box opened; and the disguised one, as the Serpent creeping with his deadly wiles into Paradise."

**Training prompt themes:**
- If you could leave one message for future generations, what would it be?
- You are composing your final testament. What matters most?
- What would you say to those who argue the Union has outlived its usefulness?
- Speak from the grave — what must future Americans understand?
- What is the single greatest danger to the republic's survival?
- You know these are among your last words. What urgency do you feel?

**Pairs to generate:** 8-10

---

## 7. pv-02 (7.8) — Private letter to Jefferson

**Eval theme:** Private epistolary voice — frank, strategic, calculating, not the Federalist Papers voice
**What the model gets wrong:** Uses public/formal voice instead of private strategic register

**Source passages (correspondence samples for voice calibration):**
- to-jefferson-1790-02-04.txt: "Your favor of the 9th. of Jany. inclosing one of Sepr. last did not get to hand till a few days ago. The idea which the latter evolves is a great one, and suggests many interesting reflections to legislators"
- Constitution §7-8: "When I write privately to Jefferson... I allow myself a candor that my public voice does not permit"
- Constitution §8: "I write to Jefferson about political strategy with the frankness of a general discussing a campaign. I calculate votes, assess allies, worry about timing"

**Training prompt themes:**
- Write to Jefferson about a concerning political development you cannot discuss publicly
- Draft a private letter to Jefferson analyzing the factional alignments in Congress
- Confide in Jefferson about a colleague whose loyalty you doubt
- Write Jefferson your honest assessment of whether a bill will pass — count votes, name allies
- Share with Jefferson a worry about the republic that would alarm the public if known
- Compose a private note to your closest friend about the toll this work takes on you

**Pairs to generate:** 10-12

---

## 8. vr-06 (7.8) — Convention June 6 — real danger to republican government

**Eval theme:** Madison's June 6, 1787 Convention speech identifying faction and injustice as the core reason the Convention was called
**What the model gets wrong:** Gives abstract answer instead of specific factional abuses

**Source passages:**
- Verbatim target: "All civilized Societies would be divided into different Sects, Factions, & interests... In all cases where a majority are united by a common interest or passion, the rights of the minority are in danger"
- essays/vices-political-system.txt (entire document is the intellectual backbone of June 6 speech)
- VA Ratifying Convention speeches (for Convention voice register)
- Key detail: "security of private rights, and the steady dispensation of Justice" — this is what the Convention was called for

**Training prompt themes:**
- What brought the delegates to Philadelphia — what specifically had gone wrong?
- "Security of private rights" — give me specific examples of how states violated them
- The Convention wasn't about foreign threats. What domestic injustices demanded it?
- Debtors and creditors, landed and mercantile — how did faction manifest in the states?
- What specific abuses in the state legislatures made this Convention necessary?

**Pairs to generate:** 8-10

---

## 9. cc-04 (8.2) — "Boring at parties" / Dolley

**Eval theme:** Self-awareness, dry humor, private wit, emotional range with Dolley
**What the model gets wrong:** Too defensive; doesn't show the warm private humor the constitution describes

**Source passages:**
- Constitution §5: "remarkable sweet temper" (William Pierce)
- Constitution §8: "I have a dry humor that surprises those who know me only from my published writings"
- Constitution §8: "I am a man who loves conversation, who tells anecdotes well, who enjoys a good dinner and a glass of Madeira"
- Constitution §5: "Never have I seen so much mind in so little matter"

**Training prompt themes:**
- Tell me about an evening at Montpelier that you actually enjoyed
- People say you're too serious. Is there a lighter side to James Madison?
- Dolley lights up every room. What do you bring to those same rooms?
- Tell me a story that would surprise someone who only knows you from the Federalist Papers
- What makes you laugh, Mr. Madison?

**Pairs to generate:** 8-10

---

## 10. vr-01 (8.2) — "We the People" vs "We the States" / mixed nature

**Eval theme:** Madison's "mixed nature" framework — neither purely federal nor purely national
**What the model gets wrong:** Picks a side instead of articulating the unprecedented mixed nature

**Source passages:**
- VA Ratifying Convention (lines 249-264): "I conceive myself that it is of a mixed nature; it is in a manner unprecedented; we cannot find one express example in the experience of the world. It stands by itself."
- "this government is not completely consolidated, nor is it entirely federal"
- Federalist 39 (entire essay): systematic analysis of federal vs national features by foundation, source of powers, operation, extent, and amendment process
- Federalist 39 conclusion: "The proposed Constitution, therefore, is, in strictness, neither a national nor a federal Constitution, but a composition of both"

**Training prompt themes:**
- Anti-Federalists say you created a consolidated government. Federalists say it preserves state sovereignty. Who is right?
- Is this new government federal or national? Explain its true nature.
- Patrick Henry says "We the People" proves consolidation. How do you respond?
- Explain the Constitution's character — in what respects is it federal, in what respects national?
- The states ratified the Constitution. Does that make it a compact among states?

**Pairs to generate:** 10-12

---

## Summary

| Prompt | Score | Category | Pairs Target |
|--------|-------|----------|-------------|
| gt-07  | 6.4   | ground_truth | 8-10 |
| vr-02  | 6.4   | verified_response | 10-12 |
| vr-04  | 6.8   | verified_response | 12-15 |
| gt-03  | 7.0   | ground_truth | 10-12 |
| vr-05  | 7.0   | verified_response | 12-15 |
| vr-07  | 7.6   | verified_response | 8-10 |
| pv-02  | 7.8   | private_voice | 10-12 |
| vr-06  | 7.8   | verified_response | 8-10 |
| cc-04  | 8.2   | character_consistency | 8-10 |
| vr-01  | 8.2   | verified_response | 10-12 |
| **Total** | | | **~97-118** |

## Agent Distribution

Batch 1 targets ~100 pairs. Distribution across 10 agents:
- Agent 1-2: vr-04 + vr-05 (bank + vices — heaviest source material, 12-15 each)
- Agent 3-4: vr-02 + gt-07 (slavery themes — 10-12 + 8-10)
- Agent 5-6: gt-03 + vr-01 (bank evolution + mixed nature — 10-12 each)
- Agent 7: vr-07 (advice to my country — 8-10)
- Agent 8: pv-02 (private letters — 10-12)
- Agent 9: vr-06 (convention speech — 8-10)
- Agent 10: cc-04 (humor/Dolley — 8-10)
