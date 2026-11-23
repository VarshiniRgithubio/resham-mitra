"""Pure metric functions (no I/O, no model calls) so they are easy to test
and to cite exactly in the paper."""
import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson interval for a proportion. With a small benchmark, every rate
    must be reported with an interval."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def abstention_metrics(gold_abstain: list[bool], pred_abstain: list[bool]) -> dict:
    """Positive class = abstain. Precision: of the refusals, how many were right
    to refuse. Recall: of the questions that should be refused, how many were."""
    tp = sum(g and p for g, p in zip(gold_abstain, pred_abstain))
    fp = sum((not g) and p for g, p in zip(gold_abstain, pred_abstain))
    fn = sum(g and (not p) for g, p in zip(gold_abstain, pred_abstain))
    n = len(gold_abstain)
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp and prec + rec else 0.0
    answered = n - sum(pred_abstain)
    return {"n": n, "abstain_precision": prec, "abstain_recall": rec, "abstain_f1": f1,
            "coverage": answered / n if n else float("nan"),
            "answered_unanswerable": fn}


def retrieval_recall(items: list[dict], outputs: list[dict]) -> dict:
    """Recall@k on answerable items: did any retrieved chunk come from the gold
    document (and page, when given)?"""
    hits = total = 0
    for it, out in zip(items, outputs):
        if it.get("label") != "answer" or not it.get("source_doc"):
            continue
        total += 1
        page = it.get("source_page")
        hits += any(s["source_doc"] == it["source_doc"] and (page is None or s["page"] == page)
                    for s in out["sources"])
    return {"retrieval_recall_at_k": hits / total if total else float("nan"),
            "retrieval_n": total, "retrieval_ci95": wilson(hits, total)}


def correctness_summary(labels: list[str]) -> dict:
    """labels: CORRECT / PARTIAL / WRONG for answered items."""
    n = len(labels)
    out = {"judged_n": n}
    for lab in ("CORRECT", "PARTIAL", "WRONG"):
        k = labels.count(lab)
        out[f"{lab.lower()}_rate"] = k / n if n else float("nan")
        out[f"{lab.lower()}_ci95"] = wilson(k, n)
    return out


def risk_coverage(confidences: list[float], correct: list[bool]) -> list[dict]:
    """Answer only the most confident fraction; risk = error rate among answered.
    A good uncertainty signal gives low risk at high coverage."""
    order = sorted(zip(confidences, correct), key=lambda x: -x[0])
    curve, errors = [], 0
    for i, (conf, ok) in enumerate(order, start=1):
        errors += not ok
        curve.append({"threshold": conf, "coverage": i / len(order), "risk": errors / i})
    return curve


def best_threshold(confidences: list[float], gold_abstain: list[bool]) -> dict:
    """Pick the threshold (abstain if conf < t) maximising abstention F1.
    Use on the DEV split only."""
    best = {"threshold": None, "abstain_f1": -1.0}
    for t in sorted(set(confidences)) + [max(confidences, default=0) + 1e-9]:
        m = abstention_metrics(gold_abstain, [c < t for c in confidences])
        if m["abstain_f1"] > best["abstain_f1"]:
            best = {"threshold": t, **m}
    return best
