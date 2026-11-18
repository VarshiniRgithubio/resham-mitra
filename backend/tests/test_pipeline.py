import asyncio

from tests.conftest import StubLLM

from app.safety.methods import agreement, parse_grounding

run = asyncio.run


def test_baseline_answers_with_sources(make_pipeline):
    out = run(make_pipeline().answer("How do I clean the tray every morning?", safety="none"))
    assert not out["abstained"] and out["sources"][0]["source_doc"] == "test.pdf"


def test_s1_abstains_on_unrelated_question(make_pipeline):
    p = make_pipeline(s1_threshold=0.9)
    out = run(p.answer("What is the price of gold in Mumbai?", safety="s1"))
    assert out["abstained"] and out["abstained_by"] == "s1"
    assert p.llm.calls == 0  # S1 refuses before paying for generation


def test_model_refusal_respected_even_without_safety(make_pipeline):
    out = run(make_pipeline(StubLLM(answer="NOT_IN_CONTEXT")).answer("anything here", safety="none"))
    assert out["abstained_by"] == "model"


def test_s2_abstains_when_ungrounded(make_pipeline):
    out = run(make_pipeline(StubLLM(grounded=False)).answer("clean tray", safety="s2"))
    assert out["abstained_by"] == "s2" and out["confidence"] == 0.0


def test_s3_abstains_when_samples_disagree(make_pipeline):
    out = run(make_pipeline(StubLLM(varied=True), s3_threshold=0.5).answer("clean tray", safety="s3"))
    assert out["abstained_by"] == "s3"


def test_kannada_abstain_message_for_kannada_question(make_pipeline):
    out = run(make_pipeline(s1_threshold=2.0).answer("ತಟ್ಟೆ ತೊಳೆಯುವುದು ಹೇಗೆ?", safety="s1"))
    assert "ರೇಷ್ಮೆ" in out["answer"]


def test_translate_mode_translates_both_ways(make_pipeline):
    out = run(make_pipeline().answer("ತಟ್ಟೆ ತೊಳೆಯುವುದು ಹೇಗೆ?", mode="translate", safety="none"))
    assert out["working_question"] == "TRANSLATED" and out["answer"] == "TRANSLATED"


def test_grounding_parser_and_agreement():
    assert parse_grounding('x {"claims":[{"supported":true},{"supported":false}]}') == 0.5
    assert parse_grounding("garbage") is None
    assert agreement(["same text", "same text"]) == 1.0


def test_greetings_get_welcome_not_refusal(make_pipeline):
    from app.rag.text import is_greeting
    p = make_pipeline(s1_threshold=2.0)  # would refuse anything else
    for g in ["hlo", "Hi sir", "namaskara", "ನಮಸ್ಕಾರ", "Good morning!"]:
        out = run(p.answer(g, safety="s1"))
        assert out["greeting"] and not out["abstained"] and "ರೇಷ್ಮೆ ಮಿತ್ರ" in out["answer"]
    assert p.llm.calls == 0
    assert not is_greeting("hi my mulberry leaves are yellow")


def test_english_question_answered_in_english(make_pipeline):
    out = run(make_pipeline().answer("How do I clean the tray every morning?", safety="s1"))
    assert out["question_lang"] == "en" and not out["abstained"]


def test_placeholder_contact_hidden_from_farmers(make_pipeline):
    out = run(make_pipeline(s1_threshold=2.0).answer("anything at all", safety="s1"))
    assert "<" not in out["answer"] and out["answer"].endswith("officer.")
    out = run(make_pipeline(s1_threshold=2.0, officer_contact="080-1234567").answer("x y", safety="s1"))
    assert out["answer"].endswith(": 080-1234567")


def test_extractive_mode_quotes_sources_without_ai(make_pipeline):
    from app.llm.providers import ExtractiveLLM
    out = run(make_pipeline(ExtractiveLLM()).answer("How do I clean the tray every morning?", safety="s1"))
    assert not out["abstained"] and out["answer"].startswith("📄")
    assert "clean the tray every morning" in out["answer"] and "[1]" in out["answer"]


def test_pdf_bullet_glyphs_become_bullets():
    from app.ingestion.chunking import split_sentences
    from app.rag.text import normalize
    text = normalize("Usage  Disinfect rearing houses  Keep 50g in a glass")
    assert "" not in text
    assert split_sentences(text) == ["Usage", "• Disinfect rearing houses", "• Keep 50g in a glass"]
