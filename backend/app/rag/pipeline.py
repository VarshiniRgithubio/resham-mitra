"""The question-answering pipeline used by both the API and the eval harness.

Modes (experiment E2):
  direct    - retrieve and answer in Kannada directly.
  translate - Kannada question -> English, retrieve + answer in English,
              translate answer back to Kannada.
Safety (experiment E1): none | s1 | s2 | s3 | s1+s2
"""
import time

from app.ingestion.chunking import split_sentences

from app.rag.prompts import (ABSTAIN_EN, ABSTAIN_KN, ANSWER_SYSTEM, ANSWER_USER,
                             GREETING_EN, GREETING_KN, LANG_NAME, NOT_IN_CONTEXT,
                             TRANSLATE)
from app.rag.text import detect_language, is_greeting
from app.safety.methods import s1_retrieval, s2_grounding, s3_self_consistency

SAFETY_METHODS = {"none", "s1", "s2", "s3", "s1+s2"}
MODES = {"direct", "translate"}


EXTRACT_HEADER_KN = "📄 ಕೈಪಿಡಿಯಲ್ಲಿ ಹೀಗಿದೆ (ಮೂಲ ಇಂಗ್ಲಿಷ್ ಪಠ್ಯ):"
EXTRACT_HEADER_EN = "📄 From the booklet:"


def contact_suffix(contact: str | None) -> str:
    """': <number>.' when a real contact is configured, else just '.'.
    Keeps placeholders like <OFFICER_CONTACT_TBD> away from farmers."""
    c = (contact or "").strip()
    return "." if not c or c.startswith("<") else f": {c}"


def format_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{i + 1}] ({h['source_doc']} p.{h['page']}) {h['text']}"
                       for i, h in enumerate(hits))


class RagPipeline:
    def __init__(self, llm, embedder, store, settings):
        self.llm, self.embedder, self.store, self.s = llm, embedder, store, settings

    def _extract(self, question: str, hits: list[dict], max_sents: int = 3) -> str | None:
        """No-AI answer: rank sentences from the top passages by embedding
        similarity to the question (BGE-M3 works across Kannada / English)
        and return the best few verbatim, each with its source number.
        Only S1 applies in this mode: S2 / S3 need a generating model."""
        cands = [(i, s.strip(" •")) for i, h in enumerate(hits[:3])
                 for s in split_sentences(h["text"])]
        cands = [(i, s) for i, s in cands if 25 <= len(s) <= 400]
        if not cands:
            return None
        qv = self.embedder.embed([question])[0]
        vecs = self.embedder.embed([s for _, s in cands])
        scored = sorted(((sum(a * b for a, b in zip(qv, v)), i, s)
                         for (i, s), v in zip(cands, vecs)), reverse=True)
        best = scored[0][0]
        keep = [x for x in scored[:max_sents] if x[0] >= best - 0.08]
        return "\n".join(f"• {s} [{i + 1}]" for _, i, s in keep)

    async def _translate(self, text: str, target: str) -> str:
        return (await self.llm.generate(TRANSLATE.format(target=target, text=text),
                                        temperature=0.0)).strip()

    async def answer(self, question: str, mode: str | None = None,
                     safety: str | None = None) -> dict:
        mode, safety = mode or self.s.pipeline_mode, safety or self.s.safety_method
        if mode not in MODES or safety not in SAFETY_METHODS:
            raise ValueError(f"bad mode/safety: {mode}/{safety}")
        t0 = time.perf_counter()
        q_lang = detect_language(question)
        if is_greeting(question):
            # Small talk skips retrieval, the LLM and safety entirely, so it
            # never affects experiment numbers. Latin-script greetings (often
            # Kannada typed in English letters) get both languages.
            reply = GREETING_KN if q_lang == "kn" else f"{GREETING_KN}\n{GREETING_EN}"
            return {"question": question, "question_lang": q_lang, "mode": mode,
                    "safety_method": safety, "greeting": True, "answer": reply,
                    "abstained": False, "abstained_by": None, "confidence": None,
                    "retrieval_top_score": None, "sources": [], "raw_answer": None,
                    "latency_s": round(time.perf_counter() - t0, 3)}
        out_lang = q_lang
        work_q, work_lang = question, q_lang
        if mode == "translate" and q_lang == "kn":
            work_q, work_lang = await self._translate(question, "English"), "en"

        hits = self.store.search(self.embedder.embed([work_q])[0], self.s.retrieval_top_k)
        top_score = max((h["score"] for h in hits), default=0.0)
        context = format_context(hits)
        result = {"question": question, "question_lang": q_lang, "mode": mode,
                  "safety_method": safety, "working_question": work_q,
                  "retrieval_top_score": top_score, "confidence": None,
                  "abstained": False, "abstained_by": None, "raw_answer": None,
                  "sources": [{k: h[k] for k in ("chunk_id", "source_doc", "page", "score")}
                              | {"snippet": h["text"][:200]} for h in hits]}

        def abstain(by: str, conf: float | None):
            msg = ABSTAIN_KN if out_lang == "kn" else ABSTAIN_EN
            result.update(abstained=True, abstained_by=by, confidence=conf,
                          answer=msg.format(contact=contact_suffix(self.s.officer_contact)),
                          latency_s=round(time.perf_counter() - t0, 3))
            return result

        # S1 runs before generation: no point paying for an answer we will refuse.
        if safety in ("s1", "s1+s2"):
            conf, refuse = s1_retrieval(hits, self.s.s1_threshold)
            result["confidence"] = conf
            if refuse:
                return abstain("s1", conf)
        if not hits:
            return abstain("no_context", 0.0)

        if self.llm.name == "extractive":
            raw = self._extract(question, hits)
            result["raw_answer"] = raw
            if raw is None:
                return abstain("no_sentence", result["confidence"])
            header = EXTRACT_HEADER_KN if out_lang == "kn" else EXTRACT_HEADER_EN
            result.update(answer=header + "\n" + raw,
                          latency_s=round(time.perf_counter() - t0, 3))
            return result

        async def generate(temperature: float | None = None) -> str:
            prompt = ANSWER_USER.format(context=context, question=work_q,
                                        language=LANG_NAME[work_lang])
            return (await self.llm.generate(prompt, system=ANSWER_SYSTEM,
                                            temperature=temperature)).strip()

        raw = await generate()
        result["raw_answer"] = raw
        # The model's own refusal applies in every config, including the
        # no-safety baseline, so E1 measures only the added safety layer.
        if NOT_IN_CONTEXT in raw:
            return abstain("model", result["confidence"])

        if safety in ("s2", "s1+s2"):
            conf, refuse = await s2_grounding(self.llm, raw, context, self.s.s2_threshold)
            result["confidence"] = conf
            if refuse:
                return abstain("s2", conf)
        if safety == "s3":
            conf, refuse = await s3_self_consistency(generate, raw, self.s.s3_samples,
                                                     self.s.s3_threshold)
            result["confidence"] = conf
            if refuse:
                return abstain("s3", conf)

        final = raw
        if work_lang != out_lang:
            final = await self._translate(raw, "simple Kannada")
        result.update(answer=final, latency_s=round(time.perf_counter() - t0, 3))
        return result
