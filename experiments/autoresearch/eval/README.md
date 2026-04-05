# Probe prompt file

Replace `probe-prompts.jsonl` with a real held-out probe subset.

Recommended composition:

- 4 to 5 prompts testing `ground_truth`
- 3 to 4 prompts testing `verified_response`
- 2 to 3 prompts guarding `private_voice`
- 2 guard prompts from strong categories such as `anachronism_trap` or `character_consistency`

Do not reuse the exact weakest training examples if they are now effectively in-distribution.
You want a probe set that is small, stable, and hard to game.
