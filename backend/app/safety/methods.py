"""Three comparable abstention methods. Each returns (confidence, abstain).

S1 retrieval threshold  - cheapest; runs BEFORE generation, no LLM call.
S2 grounding check      - LLM splits the answer into claims and checks support.
S3 self-consistency     - sample several answers; low agreement = uncertain.

Thresholds come from config and must be tuned on the dev split only
(eval/experiments/tune_thresholds.py), then frozen for the test split.
"""
import itertools
import json
import re

from app.rag.prompts import GROUNDING
from app.rag.text import jaccard


def s1_retrieval(hits: list[dict], threshold: float) -> tuple[float, bool]:
    top = max((h["score"] for h in hits), default=0.0)
    return top, top < threshold


def parse_grounding(raw: str) -> float | None:
    """Return supported-claim fraction, or None if the judge output is unusable."""
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return None
    try:
        claims = json.loads(m.group(0)).get("claims", [])
    except json.JSONDecodeError:
        return None
    if not claims:
        return None
    return sum(bool(c.get("supported")) for c in claims) / len(claims)


async def s2_grounding(llm, answer: str, context: str, threshold: float) -> tuple[float, bool]:
    raw = await llm.generate(GROUNDING.format(context=context, answer=answer), temperature=0.0)
    frac = parse_grounding(raw)
    if frac is None:  # unusable judge output: be safe and abstain
        return 0.0, True
    return frac, frac < threshold


def agreement(answers: list[str]) -> float:
    """Mean pairwise character-trigram Jaccard. Script-agnostic, so it works for
    Kannada without a tokenizer. Known limitation: paraphrases score low; an
    NLI or embedding-based agreement is a planned ablation."""
    pairs = list(itertools.combinations(answers, 2))
    return sum(jaccard(a, b) for a, b in pairs) / len(pairs) if pairs else 1.0


async def s3_self_consistency(generate_fn, first_answer: str, n: int,
                              threshold: float) -> tuple[float, bool]:
    samples = [first_answer] + [await generate_fn(0.8) for _ in range(max(n - 1, 1))]
    score = agreement(samples)
    return score, score < threshold
