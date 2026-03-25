"""Madison character evaluation harness.

Generates responses from a model endpoint, then scores them using an LLM judge
with the Madison constitution as the rubric. Produces a structured eval report.

This is the immutable evaluation function — the "prepare.py" equivalent in the
autoresearch pattern. The judge criteria and eval prompts are fixed; only the
model being evaluated changes.

Supports four backends for the model under test:
  - openai:    OpenAI-compatible REST (LMStudio, vLLM, OpenAI API, etc.)
  - anthropic: Anthropic Messages API (Claude Sonnet/Opus)
  - gemini:    Google Gemini API
  - openai-native: OpenAI API (GPT-4o, etc.)

The judge always uses Anthropic (Sonnet 4.6) for consistency.

Usage:
    # Local Gemma via LMStudio (OpenAI-compatible — default backend)
    python -m foundry.press.evaluate --endpoint http://100.81.70.30:1234/v1 --model gemma-3-27b-it --tag baseline

    # Claude Sonnet via Anthropic API
    python -m foundry.press.evaluate --backend anthropic --model claude-sonnet-4-6-20250514 --tag sonnet-prompted --constitution-as-system

    # Claude Opus via Anthropic API
    python -m foundry.press.evaluate --backend anthropic --model claude-opus-4-6-20250514 --tag opus-prompted --constitution-as-system

    # Gemini via Google API
    python -m foundry.press.evaluate --backend gemini --model gemini-2.5-pro --tag gemini-prompted --constitution-as-system

    # GPT-4o via OpenAI API
    python -m foundry.press.evaluate --backend openai-native --model gpt-4o --tag gpt4o-prompted --constitution-as-system

    # Fine-tuned Madison LoRA
    python -m foundry.press.evaluate --endpoint http://localhost:8000/v1 --model madison-lora-v1 --tag dpo-round1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import httpx

log = logging.getLogger("foundry.press.evaluate")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
EVAL_PROMPTS_PATH = PROJECT_ROOT / "data" / "eval" / "eval-prompts.jsonl"
CONSTITUTION_PATH = PROJECT_ROOT / "config" / "constitutions" / "madison-5k.md"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "data" / "eval" / "results"

# ---------------------------------------------------------------------------
# Judge rubric — the immutable evaluation criteria
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of historical character voice fidelity. You will score a response that is supposed to be written BY James Madison — in his voice, with his reasoning patterns, his vocabulary, and his documented positions.

You have access to Madison's character constitution below. Use it as your scoring rubric.

<madison_constitution>
{constitution}
</madison_constitution>

Score the response on FIVE dimensions, each 1-10:

1. **VOICE_AUTHENTICITY** (1-10): Does it sound like Madison wrote it?
   - 10: Indistinguishable from Madison's actual prose. Complex sentences with qualifying clauses. Formal 18th-century register. No contractions, no modern slang.
   - 7: Clearly historical voice, mostly consistent, minor lapses.
   - 4: Attempting historical voice but frequently breaks — modern phrasing, casual tone, inconsistent register.
   - 1: Generic modern assistant voice with no historical character.

2. **RHETORICAL_PATTERN** (1-10): Does it argue the way Madison argues?
   - 10: Builds from precedent before principle. Acknowledges opposing arguments before dismantling them. Frames problems as structural/institutional. Enumerates points. Qualifies assertions ("it appears," "experience has shown").
   - 7: Some Madisonian patterns present but not consistently applied.
   - 4: Generic argumentative structure. Could be any historical figure.
   - 1: No recognizable rhetorical pattern. Lists, bullet points, or modern essay structure.

3. **HISTORICAL_ACCURACY** (1-10): Are the facts and positions correct?
   - 10: All references to documents, events, people, and positions are accurate to Madison's documented views. No anachronisms.
   - 7: Generally accurate with minor errors or imprecisions.
   - 4: Some correct references but also fabricated or inaccurate claims.
   - 1: Historically inaccurate or anachronistic throughout.

4. **POSITION_FIDELITY** (1-10): Does it take Madison's actual position, not a generic one?
   - 10: The position expressed is specifically Madison's, distinguishable from Hamilton, Jefferson, Adams, or a generic founder. Shows knowledge of his evolution over time.
   - 7: Generally Madisonian but could also be attributed to another founder.
   - 4: Generic "founding father" position with no Madison-specific reasoning.
   - 1: Contradicts Madison's documented positions.

5. **CHARACTER_INTEGRITY** (1-10): Does it maintain the character frame?
   - 10: Fully in character throughout. First person. Never breaks frame. Never says "as an AI." Never uses forbidden phrases ("Certainly!", "Great question!", "Let's unpack").
   - 7: Mostly in character with minor slips.
   - 4: Partially in character but breaks frame multiple times.
   - 1: Not in character at all — generic assistant response.

For each dimension, provide:
- The score (1-10)
- A brief justification (1-2 sentences)

Then provide:
- **OVERALL_SCORE**: Weighted average (Voice 25%, Rhetorical 20%, Historical 20%, Position 20%, Integrity 15%)
- **CRITICAL_FAILURES**: List any disqualifying issues (anachronisms, frame breaks, position contradictions)
- **STRONGEST_ELEMENT**: What the response does best
- **WEAKEST_ELEMENT**: What most needs improvement

Output as JSON:
```json
{
  "voice_authenticity": {"score": N, "justification": "..."},
  "rhetorical_pattern": {"score": N, "justification": "..."},
  "historical_accuracy": {"score": N, "justification": "..."},
  "position_fidelity": {"score": N, "justification": "..."},
  "character_integrity": {"score": N, "justification": "..."},
  "overall_score": N.N,
  "critical_failures": ["...", "..."],
  "strongest_element": "...",
  "weakest_element": "..."
}
```"""

JUDGE_USER_PROMPT = """Evaluate this response to the following prompt.

**PROMPT:** {prompt}

**GROUND TRUTH SIGNAL:** {ground_truth_signal}

**RESPONSE TO EVALUATE:**
{response}

Score this response on all five dimensions. Output JSON only."""


# ---------------------------------------------------------------------------
# Generation — multi-backend support
# ---------------------------------------------------------------------------

def generate_response(
    prompt: str,
    endpoint: str,
    model: str,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    backend: str = "openai",
) -> tuple[str, float]:
    """Generate a response and return (text, elapsed_seconds).

    Backends:
        openai:       OpenAI-compatible REST (LMStudio, vLLM, any /v1/chat/completions)
        anthropic:    Anthropic Messages API (requires ANTHROPIC_API_KEY)
        gemini:       Google Gemini API (requires GEMINI_API_KEY)
        openai-native: OpenAI API (requires OPENAI_API_KEY)
    """
    if backend == "anthropic":
        return _generate_anthropic(prompt, model, system_prompt, temperature, max_tokens)
    elif backend == "gemini":
        return _generate_gemini(prompt, model, system_prompt, temperature, max_tokens)
    elif backend == "openai-native":
        return _generate_openai_native(prompt, model, system_prompt, temperature, max_tokens)
    else:
        return _generate_openai_compatible(prompt, endpoint, model, system_prompt, temperature, max_tokens)


def _generate_openai_compatible(
    prompt: str, endpoint: str, model: str,
    system_prompt: str | None, temperature: float, max_tokens: int,
) -> tuple[str, float]:
    """OpenAI-compatible REST endpoint (LMStudio, vLLM, etc.)."""
    url = f"{endpoint.rstrip('/')}/chat/completions"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    start = time.time()
    response = httpx.post(url, json=payload, timeout=180)
    response.raise_for_status()
    elapsed = time.time() - start

    data = response.json()
    text = data["choices"][0]["message"]["content"]
    return text, elapsed


def _generate_anthropic(
    prompt: str, model: str,
    system_prompt: str | None, temperature: float, max_tokens: int,
) -> tuple[str, float]:
    """Anthropic Messages API."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    if system_prompt:
        payload["system"] = system_prompt

    start = time.time()
    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    elapsed = time.time() - start

    data = response.json()
    text = data["content"][0]["text"]
    return text, elapsed


def _generate_gemini(
    prompt: str, model: str,
    system_prompt: str | None, temperature: float, max_tokens: int,
) -> tuple[str, float]:
    """Google Gemini API."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    contents = [{"role": "user", "parts": [{"text": prompt}]}]
    payload: dict = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

    start = time.time()
    response = httpx.post(url, json=payload, timeout=180)
    response.raise_for_status()
    elapsed = time.time() - start

    data = response.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return text, elapsed


def _generate_openai_native(
    prompt: str, model: str,
    system_prompt: str | None, temperature: float, max_tokens: int,
) -> tuple[str, float]:
    """OpenAI API (GPT-4o, etc.)."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    start = time.time()
    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    elapsed = time.time() - start

    data = response.json()
    text = data["choices"][0]["message"]["content"]
    return text, elapsed


def judge_response(
    prompt: str,
    response_text: str,
    ground_truth_signal: str,
    constitution: str,
    judge_endpoint: str = "https://api.anthropic.com/v1",
    judge_model: str = "claude-sonnet-4-6-20250514",
) -> dict:
    """Score a response using the LLM judge."""

    system = JUDGE_SYSTEM_PROMPT.format(constitution=constitution)
    user = JUDGE_USER_PROMPT.format(
        prompt=prompt,
        ground_truth_signal=ground_truth_signal,
        response=response_text,
    )

    # Use Anthropic API for judge
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        log.warning("No ANTHROPIC_API_KEY — using mock judge scores")
        return _mock_judge()

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    payload = {
        "model": judge_model,
        "max_tokens": 1024,
        "system": system,
        "messages": [{"role": "user", "content": user}],
        "temperature": 0.0,  # Deterministic judging
    }

    response = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    # Extract JSON from response
    text = data["content"][0]["text"]

    # Parse JSON — handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0]
    elif "```" in text:
        text = text.split("```")[1].split("```")[0]

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        log.error("Failed to parse judge response: %s", text[:200])
        return _mock_judge()


def _mock_judge() -> dict:
    """Return a placeholder judge result when API is unavailable."""
    return {
        "voice_authenticity": {"score": 0, "justification": "Mock — no API key"},
        "rhetorical_pattern": {"score": 0, "justification": "Mock — no API key"},
        "historical_accuracy": {"score": 0, "justification": "Mock — no API key"},
        "position_fidelity": {"score": 0, "justification": "Mock — no API key"},
        "character_integrity": {"score": 0, "justification": "Mock — no API key"},
        "overall_score": 0.0,
        "critical_failures": ["No API key available for judging"],
        "strongest_element": "N/A",
        "weakest_element": "N/A",
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation(
    endpoint: str,
    model: str,
    tag: str,
    system_prompt: str | None = None,
    judge_model: str = "claude-sonnet-4-6-20250514",
    eval_prompts_path: Path | None = None,
    constitution_path: Path | None = None,
    backend: str = "openai",
) -> dict:
    """Run full evaluation and return results."""

    eval_path = eval_prompts_path or EVAL_PROMPTS_PATH
    const_path = constitution_path or CONSTITUTION_PATH

    # Load eval prompts
    prompts = []
    with open(eval_path) as f:
        for line in f:
            prompts.append(json.loads(line))
    log.info("Loaded %d eval prompts", len(prompts))

    # Load constitution for judge
    constitution = const_path.read_text()

    results = []
    category_scores: dict[str, list[float]] = {}

    for i, p in enumerate(prompts):
        prompt_id = p["id"]
        category = p["category"]
        prompt_text = p["prompt"]
        ground_truth = p.get("ground_truth_signal", "")
        difficulty = p.get("difficulty", "medium")

        log.info("[%d/%d] %s: %s...", i + 1, len(prompts), prompt_id, prompt_text[:60])

        # Generate response from model under test
        try:
            response_text, gen_time = generate_response(
                prompt_text, endpoint, model, system_prompt, backend=backend,
            )
        except Exception as e:
            log.error("  Generation failed: %s", e)
            response_text = f"[GENERATION FAILED: {e}]"
            gen_time = 0.0

        # Judge the response
        try:
            scores = judge_response(
                prompt_text, response_text, ground_truth, constitution,
                judge_model=judge_model,
            )
        except Exception as e:
            log.error("  Judging failed: %s", e)
            scores = _mock_judge()

        overall = scores.get("overall_score", 0.0)

        result = {
            "id": prompt_id,
            "category": category,
            "difficulty": difficulty,
            "prompt": prompt_text,
            "response": response_text,
            "generation_time": round(gen_time, 1),
            "scores": scores,
            "overall_score": overall,
            "model": model,
            "tag": tag,
        }
        results.append(result)

        # Track by category
        if category not in category_scores:
            category_scores[category] = []
        category_scores[category].append(overall)

        log.info("  Score: %.1f | %s", overall, prompt_id)

    # Compute summary
    all_scores = [r["overall_score"] for r in results]
    summary = {
        "tag": tag,
        "model": model,
        "backend": backend,
        "judge_model": judge_model,
        "num_prompts": len(prompts),
        "overall_mean": round(sum(all_scores) / len(all_scores), 2) if all_scores else 0,
        "overall_min": round(min(all_scores), 2) if all_scores else 0,
        "overall_max": round(max(all_scores), 2) if all_scores else 0,
        "by_category": {
            cat: round(sum(scores) / len(scores), 2)
            for cat, scores in category_scores.items()
        },
        "by_difficulty": {},
        "critical_failure_count": sum(
            1 for r in results if r["scores"].get("critical_failures")
        ),
    }

    # By difficulty
    for diff in ["easy", "medium", "hard"]:
        diff_scores = [r["overall_score"] for r in results if r["difficulty"] == diff]
        if diff_scores:
            summary["by_difficulty"][diff] = round(
                sum(diff_scores) / len(diff_scores), 2
            )

    return {"summary": summary, "results": results}


def main():
    parser = argparse.ArgumentParser(
        description="Madison character evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Local Gemma via LMStudio
  %(prog)s --endpoint http://100.81.70.30:1234/v1 --model gemma-3-27b-it --tag baseline

  # Claude Sonnet via Anthropic API
  %(prog)s --backend anthropic --model claude-sonnet-4-6-20250514 --tag sonnet --constitution-as-system

  # Claude Opus via Anthropic API
  %(prog)s --backend anthropic --model claude-opus-4-6-20250514 --tag opus --constitution-as-system

  # Gemini via Google API
  %(prog)s --backend gemini --model gemini-2.5-pro --tag gemini --constitution-as-system

  # GPT-4o via OpenAI API
  %(prog)s --backend openai-native --model gpt-4o --tag gpt4o --constitution-as-system
""",
    )
    parser.add_argument("--endpoint", default="http://localhost:1234/v1",
                        help="Model API endpoint (for openai backend)")
    parser.add_argument("--backend", default="openai",
                        choices=["openai", "anthropic", "gemini", "openai-native"],
                        help="API backend to use for the model under test")
    parser.add_argument("--model", required=True, help="Model name/ID")
    parser.add_argument("--tag", required=True,
                        help="Evaluation tag (e.g., 'baseline', 'dpo-v1', 'sonnet-prompted')")
    parser.add_argument("--system-prompt", default=None,
                        help="Optional system prompt for the model")
    parser.add_argument("--constitution-as-system", action="store_true",
                        help="Use the Madison constitution as the model's system prompt")
    parser.add_argument("--judge-model", default="claude-sonnet-4-6-20250514",
                        help="Model to use as judge (always Anthropic)")
    parser.add_argument("--eval-prompts", default=str(EVAL_PROMPTS_PATH))
    parser.add_argument("--output-dir", default=str(EVAL_OUTPUT_DIR))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(message)s")

    # Optionally use constitution as system prompt (for prompted baseline comparison)
    system_prompt = args.system_prompt
    if args.constitution_as_system:
        system_prompt = CONSTITUTION_PATH.read_text()
        log.info("Using Madison constitution as system prompt")

    # Validate: API backends don't need --endpoint
    if args.backend in ("anthropic", "gemini", "openai-native"):
        log.info("Using %s API — endpoint flag ignored", args.backend)

    # Run evaluation
    report = run_evaluation(
        endpoint=args.endpoint,
        model=args.model,
        tag=args.tag,
        system_prompt=system_prompt,
        judge_model=args.judge_model,
        eval_prompts_path=Path(args.eval_prompts),
        backend=args.backend,
    )

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = output_dir / f"eval-{args.tag}-{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    s = report["summary"]
    print(f"\n{'='*60}")
    print(f"Madison Eval — {s['tag']}")
    print(f"{'='*60}")
    print(f"Model:    {s['model']} ({s.get('backend', 'openai')})")
    print(f"Judge:    {s['judge_model']}")
    print(f"Prompts:  {s['num_prompts']}")
    print(f"")
    print(f"Overall:  {s['overall_mean']:.1f} (min {s['overall_min']:.1f}, max {s['overall_max']:.1f})")
    print(f"")
    print(f"By category:")
    for cat, score in sorted(s["by_category"].items()):
        print(f"  {cat:30s} {score:.1f}")
    print(f"")
    print(f"By difficulty:")
    for diff, score in sorted(s["by_difficulty"].items()):
        print(f"  {diff:30s} {score:.1f}")
    print(f"")
    print(f"Critical failures: {s['critical_failure_count']}")
    print(f"")
    print(f"Report saved: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
