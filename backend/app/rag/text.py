"""Small text utilities shared by ingestion, retrieval and safety."""
import re
import unicodedata

KANNADA = re.compile(r"[ಀ-೿]")
LETTER = re.compile(r"[\wಀ-೿]")


def kannada_ratio(text: str) -> float:
    letters = LETTER.findall(text)
    return len(KANNADA.findall(text)) / len(letters) if letters else 0.0


def detect_language(text: str) -> str:
    """Script-based detection. Good enough for Kannada vs English; mixed text
    with >=30% Kannada letters counts as Kannada."""
    return "kn" if kannada_ratio(text) >= 0.3 else "en"


def normalize(text: str) -> str:
    """NFC-normalise (important for Kannada combining marks) and tidy spaces."""
    text = unicodedata.normalize("NFC", text).replace("­", "")
    # PDF symbol fonts (Wingdings etc.) extract bullets as private-use chars
    # that render as empty boxes; turn them into a real bullet.
    text = "".join(" • " if unicodedata.category(ch) == "Co" else ch for ch in text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def char_ngrams(text: str, n: int = 3) -> set[str]:
    out = set()
    for word in normalize(text).lower().split():
        padded = f" {word} "
        out.update(padded[i:i + n] for i in range(max(1, len(padded) - n + 1)))
    return out


def jaccard(a: str, b: str) -> float:
    sa, sb = char_ngrams(a), char_ngrams(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


GREETING_WORDS = {
    "hi", "hii", "hiii", "hello", "helo", "hlo", "hey", "namaste", "namaskara",
    "namaskaara", "namaskar", "good", "morning", "evening", "afternoon", "sir",
    "madam", "anna", "akka", "thanks", "thank", "you", "ok", "okay",
    "ನಮಸ್ಕಾರ", "ನಮಸ್ತೆ", "ನಮಸ್ಕಾರಗಳು", "ಹಾಯ್", "ಹಲೋ", "ಶುಭೋದಯ", "ಧನ್ಯವಾದ", "ಧನ್ಯವಾದಗಳು",
    "ಸರ್", "ಮೇಡಂ", "ಅಣ್ಣ", "ಅಕ್ಕ",
}


def is_greeting(text: str) -> bool:
    """True for short messages made only of greeting / politeness words.
    Any real content word (e.g. 'hi, my leaves are yellow') makes it a question."""
    words = re.findall(r"[\w\u0C80-\u0CFF]+", text.lower())
    return 0 < len(words) <= 4 and all(w in GREETING_WORDS for w in words)
