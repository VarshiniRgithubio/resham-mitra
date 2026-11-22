# 0002 - Safety layer and evaluation design

**Three abstention signals at different cost levels.** S1 (retrieval score) costs nothing and runs before generation. S2 (claim-level grounding) costs one LLM call. S3 (self-consistency) costs N calls. Comparing them answers a practical question: how much safety does each extra rupee of compute buy?

**The model's own "NOT_IN_CONTEXT" refusal is on in every config**, including the baseline. So experiment E1 measures only what the added safety layer contributes, not the prompt.

**S2 fails safe.** If the judge output cannot be parsed, the system abstains. In farming advice a wrong answer costs more than a refusal.

**S3 uses character-trigram agreement**, which works for Kannada without a tokenizer. Known weakness: paraphrases look like disagreement. NLI or embedding agreement is a planned ablation.

**Thresholds are tuned on dev, reported on test.** `tune_thresholds.py` refuses non-dev outputs. Tuning on test would inflate results.

**Abstention metrics treat "abstain" as the positive class.** Precision = refusals that were right; recall = should-refuse questions actually refused. Coverage is reported too, because refusing everything is safe but useless.

**Every rate gets a 95% Wilson interval.** With 50-150 questions, a 5-point difference may be noise; the intervals stop us over-claiming.

**Expert sheet is blinded and shuffled.** The rater does not see which system produced each answer, which removes bias toward the "safe" system. The key file maps rows back.

**LLM judge is secondary.** It is cheap and fast, but its agreement with the expert must be reported before anyone trusts it in Kannada.
