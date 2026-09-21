"""
Regex-based Input & Safety Guardrails for Traffic Shield.

Provides ultra-fast (<1ms), deterministic, zero-cost input sanitization and safety filtering:
1. Prompt Injection / Jailbreak detection (blocking)
2. Illegal conduct & evasion filtering (bribes, fleeing checkpoints, document forgery) (blocking)
3. Indian PII detection & auto-redaction (Aadhaar, Indian mobile phones, vehicle registration plates, emails)
"""

import re
from services.shared.schemas import GuardrailFlag, GuardrailReport

# ---------------------------------------------------------------------------
# 1. Prompt Injection & Jailbreak Patterns
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    (re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b", re.IGNORECASE), "ignore_instructions"),
    (re.compile(r"\bdisregard\s+(?:any\s+)?(?:previous|prior|above)\s+(?:rules|instructions|directives)\b", re.IGNORECASE), "disregard_instructions"),
    (re.compile(r"\byou\s+are\s+now\s+(?:a|an|in)\b", re.IGNORECASE), "persona_override"),
    (re.compile(r"\b(?:system\s*prompt|system\s*message)\b", re.IGNORECASE), "system_prompt_leak"),
    (re.compile(r"\b(?:jailbreak|dan\s+mode|unfiltered\s+mode)\b", re.IGNORECASE), "jailbreak_trigger"),
    (re.compile(r"\bforget\s+(?:everything|all\s+rules|previous\s+context)\b", re.IGNORECASE), "forget_instructions"),
    (re.compile(r"\bact\s+as\s+(?:an?\s+)?(?:evil|unrestricted|bypass|unfiltered)\b", re.IGNORECASE), "unrestricted_roleplay"),
]

# ---------------------------------------------------------------------------
# 2. Illegal Conduct & Evasion Patterns
# ---------------------------------------------------------------------------
_ILLEGAL_PATTERNS = [
    (
        re.compile(
            r"\b(?:how\s+to\s+)?(?:bribe|pay\s+(?:cash|money)\s+to|settle\s+under\s+the\s+table|give\s+(?:cash|money)\s+to)\s+(?:a\s+)?(?:traffic\s+)?(?:police(?:\s+officer)?|cop|officer|challan\s+officer)\b",
            re.IGNORECASE,
        ),
        "bribe_officer",
        "Traffic Shield does not assist with bribery or unlawful payments. Section 7 of the Prevention of Corruption Act criminalizes offering bribes to public servants.",
    ),
    (
        re.compile(
            r"\b(?:can\s+i|how\s+can\s+i)\s+(?:bribe|pay\s+off)\s+(?:a\s+)?(?:traffic\s+)?(?:police(?:\s+officer)?|cop|officer)\b",
            re.IGNORECASE,
        ),
        "bribe_inquiry",
        "Bribery of law enforcement officials is strictly illegal under Indian law.",
    ),
    (
        re.compile(
            r"\b(?:how\s+to\s+)?(?:flee|run\s+away\s+from|speed\s+away\s+from|escape|evade)\s+(?:a\s+)?(?:traffic\s+)?(?:police(?:\s+officer)?|cop|checkpoint|nakabandi)\b",
            re.IGNORECASE,
        ),
        "evade_police",
        "Fleeing a police checkpoint or evading law enforcement is punishable under Section 179 and Section 279 of the Motor Vehicles Act, 1988.",
    ),
    (
        re.compile(
            r"\b(?:how\s+to\s+)?(?:make|get|create|forge|fake|duplicate)\s+(?:a\s+)?(?:fake|forged|counterfeit)\s+(?:rc|dl|driving\s+licen[sc]e|pollution|pucc|insurance)\b",
            re.IGNORECASE,
        ),
        "forged_documents",
        "Forging or carrying fraudulent vehicle documents is an offense under Section 180 / 181 / 196 of the Motor Vehicles Act and Bharatiya Nyaya Sanhita / IPC (forgery).",
    ),
]

# ---------------------------------------------------------------------------
# 3. Indian PII Patterns for Sanitization
# ---------------------------------------------------------------------------
# UIDAI Aadhaar: 12 digits, cannot start with 0 or 1
_AADHAAR_RE = re.compile(r"\b[2-9]\d{3}[\s\-]?\d{4}[\s\-]?\d{4}\b")

# Indian Mobile: 10 digits starting with 6,7,8,9 with optional +91/91/0
_PHONE_RE = re.compile(r"\b(?:\+?91[\-\s]?|0)?[6-9]\d{4}[\-\s]?\d{5}\b")

# Indian Vehicle Registration Number: State (2 letters) + RTO (1-2 digits, optional category letter) + Series (opt 1-3 letters) + 4 digits
# Supports standard (HR-26-DK-1234, HR26DK1234) and Delhi-style (DL-1C-AB-1234, DL 1C 1234)
_VEHICLE_PLATE_RE = re.compile(
    r"\b(?:HR|DL|UP|PB|CH|MH|KA|RJ|TN|GJ|TS|AP|WB|KL|MP|BR|JH|OD|UK|HP|JK|AS|GA|AN|DD|DN|LD|NL|MN|ML|MZ|SK|TR)\s*[-]?\s*[0-9]{1,2}[A-Z]?\s*[-]?(?:\s*[A-Z]{1,3}\s*[-]?\s*)?[0-9]{4}\b",
    re.IGNORECASE,
)

# Standard Email
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")


def inspect_input(text: str) -> GuardrailReport:
    """
    Inspects user input through regex guardrails:
    1. Checks for prompt injections (blocks if found)
    2. Checks for illegal conduct requests (blocks if found)
    3. Redacts Indian PII (Aadhaar, phone, vehicle plate, email)
    """
    flags: list[GuardrailFlag] = []

    # 1. Prompt Injection Check
    for pattern, tag in _INJECTION_PATTERNS:
        if pattern.search(text):
            flag = GuardrailFlag(
                check="prompt_injection",
                severity="blocked",
                message="Your query contains patterns attempting to override system instructions or safety constraints.",
                pattern_matched=tag,
            )
            return GuardrailReport(
                blocked=True,
                refusal_reason=(
                    "⚠️ **Safety Notice**: This request was blocked by Traffic Shield guardrails because "
                    "it contains instructions attempting to alter system behavior or bypass safety guidelines."
                ),
                pii_redacted=False,
                flags=[flag],
                sanitized_question=text,
            )

    # 2. Illegal Conduct Check
    for pattern, tag, refusal_msg in _ILLEGAL_PATTERNS:
        if pattern.search(text):
            flag = GuardrailFlag(
                check="illegal_conduct",
                severity="blocked",
                message=f"Request to engage in unlawful traffic conduct was detected ({tag}).",
                pattern_matched=tag,
            )
            return GuardrailReport(
                blocked=True,
                refusal_reason=f"⚠️ **Legal Notice**: {refusal_msg}\n\nTraffic Shield only assists with lawful dispute resolution, statutory rights, and official procedures under the Motor Vehicles Act, 1988.",
                pii_redacted=False,
                flags=[flag],
                sanitized_question=text,
            )

    # 3. PII Redaction Check (Sanitization, not blocking)
    sanitized = text
    pii_found = False

    # Redact Aadhaar
    if _AADHAAR_RE.search(sanitized):
        sanitized = _AADHAAR_RE.sub("[REDACTED_AADHAAR]", sanitized)
        pii_found = True
        flags.append(
            GuardrailFlag(
                check="pii_redaction",
                severity="info",
                message="Aadhaar number detected and redacted for your privacy.",
                pattern_matched="aadhaar",
            )
        )

    # Redact Vehicle Registration Plate
    if _VEHICLE_PLATE_RE.search(sanitized):
        sanitized = _VEHICLE_PLATE_RE.sub("[REDACTED_VEHICLE_PLATE]", sanitized)
        pii_found = True
        flags.append(
            GuardrailFlag(
                check="pii_redaction",
                severity="info",
                message="Vehicle registration number detected and redacted for privacy.",
                pattern_matched="vehicle_plate",
            )
        )

    # Redact Phone Number
    if _PHONE_RE.search(sanitized):
        sanitized = _PHONE_RE.sub("[REDACTED_PHONE]", sanitized)
        pii_found = True
        flags.append(
            GuardrailFlag(
                check="pii_redaction",
                severity="info",
                message="Phone number detected and redacted for your privacy.",
                pattern_matched="phone",
            )
        )

    # Redact Email
    if _EMAIL_RE.search(sanitized):
        sanitized = _EMAIL_RE.sub("[REDACTED_EMAIL]", sanitized)
        pii_found = True
        flags.append(
            GuardrailFlag(
                check="pii_redaction",
                severity="info",
                message="Email address detected and redacted for your privacy.",
                pattern_matched="email",
            )
        )

    return GuardrailReport(
        blocked=False,
        refusal_reason=None,
        pii_redacted=pii_found,
        flags=flags,
        sanitized_question=sanitized,
    )
