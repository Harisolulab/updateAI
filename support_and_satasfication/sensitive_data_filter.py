import spacy
import re
from typing import List, Dict

# Load spaCy English model with NER support
nlp = spacy.load("en_core_web_sm")

# Define sensitive keywords and patterns related to GDPR/compliance
SENSITIVE_KEYWORDS = [
    "data deletion request",
    "personal data request",
    "right to be forgotten",
    "data protection",
    "gdpr",
    "privacy policy",
    "remove my data",
    "erase my data",
    "subject access request",
]

# Compile regex for keyword matching (case insensitive)
keyword_pattern = re.compile("|".join(re.escape(k) for k in SENSITIVE_KEYWORDS), re.I)


def detect_sensitive_terms(text: str) -> Dict:
    """
    Detect sensitive terms or entities in the text.
    Returns dict with:
    - 'flagged_terms': list of detected keywords
    - 'entities': list of detected named entities from spaCy related to legal/privacy
    - 'redacted_text': text with sensitive terms redacted
    """

    # 1. Keyword matching
    flagged_terms = list(set(m.group(0) for m in keyword_pattern.finditer(text)))

    # 2. Named Entity Recognition for personal data or legal terms (if any)
    doc = nlp(text)
    # Custom filter: here we take PERSON, ORG, GPE as example sensitive entities
    personal_entities = [ent.text for ent in doc.ents if ent.label_ in ("PERSON", "ORG", "GPE")]

    # Combine all sensitive terms
    all_sensitive = set(flagged_terms + personal_entities)

    # Redact sensitive terms in text
    redacted_text = text
    for term in all_sensitive:
        redacted_text = re.sub(re.escape(term), "[REDACTED]", redacted_text, flags=re.I)

    return {
        "flagged_terms": flagged_terms,
        "entities": personal_entities,
        "redacted_text": redacted_text,
        "is_flagged": bool(all_sensitive),
    }


# Example usage
if __name__ == "__main__":
    sample_text = (
        "Hello, I want to file a data deletion request under GDPR. "
        "Please remove my personal data. Contact my lawyer at Acme Corp."
    )

    result = detect_sensitive_terms(sample_text)

    print("Sensitive Terms Detected:", result["flagged_terms"])
    print("Named Entities:", result["entities"])
    print("Redacted Text:", result["redacted_text"])
    print("Flagged (Compliance Alert):", result["is_flagged"])
