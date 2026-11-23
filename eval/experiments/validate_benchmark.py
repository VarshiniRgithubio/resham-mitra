"""Check a benchmark file against data/benchmark/SCHEMA.md. Exit 1 on errors."""
import sys
from collections import Counter

from _common import read_jsonl

REQUIRED = ["id", "question_kn", "topic", "label", "verified_by", "split"]
TOPICS = {"mulberry", "rearing", "disease_pest", "cocoon_market", "schemes", "other"}


def validate(rows: list[dict]) -> list[str]:
    errs, ids = [], Counter(r.get("id") for r in rows)
    for i, r in enumerate(rows, 1):
        tag = f"line {i} ({r.get('id')})"
        errs += [f"{tag}: missing {k}" for k in REQUIRED if not r.get(k)]
        if r.get("label") not in ("answer", "abstain"):
            errs.append(f"{tag}: label must be answer|abstain")
        if r.get("split") not in ("dev", "test"):
            errs.append(f"{tag}: split must be dev|test")
        if r.get("topic") and r["topic"] not in TOPICS:
            errs.append(f"{tag}: unknown topic {r['topic']}")
        if r.get("label") == "answer" and not (r.get("gold_answer_kn") and r.get("source_doc")):
            errs.append(f"{tag}: answer items need gold_answer_kn and source_doc")
        if ids[r.get("id")] > 1:
            errs.append(f"{tag}: duplicate id")
    return errs


if __name__ == "__main__":
    rows = read_jsonl(sys.argv[1])
    errs = validate(rows)
    print("\n".join(errs) or "OK")
    print(f"{len(rows)} items | labels {dict(Counter(r.get('label') for r in rows))} | "
          f"splits {dict(Counter(r.get('split') for r in rows))}")
    sys.exit(1 if errs else 0)
