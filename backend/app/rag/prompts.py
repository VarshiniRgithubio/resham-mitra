"""All prompts in one place so they can be versioned and reported in the paper.
Kannada strings must be reviewed by a native speaker / officer before release."""

NOT_IN_CONTEXT = "NOT_IN_CONTEXT"

ANSWER_SYSTEM = (
    "You are an assistant for sericulture (silk farming) farmers in Karnataka. "
    "Answer ONLY using the numbered context passages. Cite passages like [1]. "
    "Use short, simple sentences a farmer can follow. Do not add facts, doses or "
    "chemical names that are not in the context. If the context does not contain "
    f"the answer, reply with exactly {NOT_IN_CONTEXT}."
)

ANSWER_USER = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer in {language}."

TRANSLATE = ("Translate the following text to {target}. Keep technical terms, "
             "numbers and units exact. Output only the translation.\n\n{text}")

GROUNDING = (
    "You are checking an answer against source passages.\n"
    "1. Split the ANSWER into atomic factual claims.\n"
    "2. For each claim decide if it is fully supported by the PASSAGES.\n"
    "Return only JSON: {{\"claims\": [{{\"claim\": str, \"supported\": bool}}]}}\n\n"
    "PASSAGES:\n{context}\n\nANSWER:\n{answer}"
)

# DRAFT Kannada abstain message - must be checked by a native speaker / officer.
ABSTAIN_KN = (
    "ಕ್ಷಮಿಸಿ, ಈ ಪ್ರಶ್ನೆಗೆ ಖಚಿತವಾದ ಉತ್ತರ ನೀಡಲು ನನ್ನ ಬಳಿ ಸಾಕಷ್ಟು ಮಾಹಿತಿ ಇಲ್ಲ. "
    "ತಪ್ಪು ಸಲಹೆ ನೀಡದಿರಲು, ದಯವಿಟ್ಟು ನಿಮ್ಮ ಹತ್ತಿರದ ರೇಷ್ಮೆ ಇಲಾಖೆ ಅಧಿಕಾರಿಯನ್ನು ಸಂಪರ್ಕಿಸಿ{contact}"
)
ABSTAIN_EN = ("Sorry, I am not confident enough to answer this. To avoid giving wrong "
              "advice, please contact your nearest sericulture officer{contact}")

LANG_NAME = {
    # Local models tend to copy the language of the (English) context, so the
    # Kannada instruction is explicit about script and about keeping numbers.
    "kn": ("simple Kannada. Write the WHOLE answer in Kannada script (ಕನ್ನಡ), even "
           "though the context is in English. Keep numbers, units and product names "
           "exactly as in the context"),
    "en": "simple English",
}

# DRAFT greeting replies - Kannada to be checked by a native speaker / officer.
GREETING_KN = ("ನಮಸ್ಕಾರ! ನಾನು ರೇಷ್ಮೆ ಮಿತ್ರ. ರೇಷ್ಮೆ ಕೃಷಿ ಬಗ್ಗೆ ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು "
               "ಕನ್ನಡ ಅಥವಾ ಇಂಗ್ಲಿಷ್‌ನಲ್ಲಿ ಕೇಳಿ.")
GREETING_EN = ("Hello! I am Resham Mitra. Ask me your question about silk farming, "
               "in Kannada or English.")
