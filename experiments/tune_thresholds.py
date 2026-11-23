"""Pick a safety threshold on the DEV split.

1. Produce dev outputs with the method's threshold set to 0 (never abstains,
   but confidence is still recorded), e.g.
     S1_THRESHOLD=0 python eval/experiments/run.py --split dev --configs direct:s1
     S2_THRESHOLD=0 ... --configs direct:s2   |   S3_THRESHOLD=0 ... direct:s3
2. python eval/experiments/tune_thresholds.py <benchmark> <outputs.jsonl>
3. Put the chosen value in .env and run the TEST split once.
"""
import json
import sys

from _common import read_jsonl
from metrics.metrics import best_threshold

if __name__ == "__main__":
    items = {r["id"]: r for r in read_jsonl(sys.argv[1])}
    outs = [o for o in read_jsonl(sys.argv[2]) if o["abstained_by"] != "model"]
    conf = [o["confidence"] if o["confidence"] is not None else o["retrieval_top_score"]
            for o in outs]
    gold = [items[o["id"]]["label"] == "abstain" for o in outs]
    if any(items[o["id"]]["split"] != "dev" for o in outs):
        sys.exit("Refusing: tune on dev outputs only.")
    print(json.dumps(best_threshold(conf, gold), indent=2, default=str))
