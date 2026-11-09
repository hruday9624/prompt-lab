"""Local prompt linter.

Pure Python: no network, no API key, no model call. Each check is a plain
heuristic, so results are repeatable and the whole module can be tested
without any provider. Heuristics can be wrong on edge cases; every finding
says why it fired so the user can overrule it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SEVERITY_WEIGHT = {"high": 20, "medium": 10, "low": 5}

VAGUE_WORDS = [
    "good", "nice", "better", "best", "great", "some", "stuff", "things",
    "etc", "appropriate", "as needed", "properly", "interesting", "cool",
    "a bit", "kind of", "sort of",
]

FORMAT_HINTS = re.compile(
    r"\b(json|yaml|csv|table|bullet|bullets|list|markdown|paragraph|paragraphs|"
    r"sentence|sentences|words?|characters?|headings?|sections?|format|schema|"
    r"template|step[- ]by[- ]step|numbered|xml|html|code block)\b",
    re.IGNORECASE,
)
ROLE_HINTS = re.compile(
    r"\b(you are|act as|as an? |your role|persona|audience|readers?|"
    r"(for|to|aimed at|targeting) (my |our |the |a |an |new |first[- ]time )?([\w-]+ ){0,3}"
    r"(teams?|users?|customers?|readers?|students?|beginners?|managers?|developers?|"
    r"founders?|clients?|parents?|buyers?|leads?|executives?|engineers?|marketers?|"
    r"people|experts?|owners?|recruiters?|candidates?))\b",
    re.IGNORECASE,
)
# Mechanical tasks where an audience or purpose adds little.
MECHANICAL_TASK = re.compile(
    r"\b(extract|classify|parse|convert|translate|label|tag|reformat|deduplicate)\b",
    re.IGNORECASE,
)
GOAL_HINTS = re.compile(
    r"\b(so that|in order to|goal|purpose|objective|because|will be used|used for|"
    r"the aim|we want|i want|i need)\b",
    re.IGNORECASE,
)
FALLBACK_HINTS = re.compile(
    r"\b(if (you are |you're )?(not sure|unsure|unclear|unknown|missing)|"
    r"if (it|there) (is|are)n'?t|say (so|that you don't know)|"
    r"do not (guess|invent|make up)|don't (guess|invent|make up)|"
    r"cite|source|only use|based only on)\b",
    re.IGNORECASE,
)
FACT_REQUEST = re.compile(
    r"\b(statistics|stats|facts?|cite|citations?|sources?|research shows|"
    r"according to|data on|numbers? for|market size|quote)\b",
    re.IGNORECASE,
)
DELIMITERS = re.compile(r"(```|<[a-zA-Z_][\w-]*>|\"\"\"|^---+$|^#{1,6} )", re.MULTILINE)
SHORT_WORDS = re.compile(r"\b(brief|short|concise|succinct|one[- ]line|tl;?dr)\b", re.I)
LONG_WORDS = re.compile(
    r"\b(detailed|comprehensive|thorough|in[- ]depth|exhaustive|extensive)\b", re.I
)
NEGATIVE = re.compile(
    r"\b(don't|do not|never|avoid|without|no |not )", re.IGNORECASE
)
EXAMPLE_HINTS = re.compile(
    r"\b(for example|e\.g\.|example:|such as|here is an example|sample)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Finding:
    id: str
    severity: str  # "high" | "medium" | "low"
    message: str
    fix: str


@dataclass(frozen=True)
class LintResult:
    score: int
    findings: list[Finding]
    word_count: int

    @property
    def clean(self) -> bool:
        return not self.findings


def _words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9'’\-]+", text)


def lint_prompt(text: str) -> LintResult:
    """Return a 0-100 score and a list of findings for ``text``."""
    text = text or ""
    words = _words(text)
    n = len(words)
    findings: list[Finding] = []

    if n == 0:
        return LintResult(
            score=0,
            findings=[
                Finding("empty", "high", "The prompt is empty.", "Write the task first.")
            ],
            word_count=0,
        )

    if n < 8:
        findings.append(
            Finding(
                "too_short",
                "high",
                f"Only {n} words. A model has to guess the audience, scope and format.",
                "Add who the answer is for, what a good result looks like, and the format.",
            )
        )

    mechanical = bool(MECHANICAL_TASK.search(text))

    if n >= 8 and not mechanical and not ROLE_HINTS.search(text):
        findings.append(
            Finding(
                "no_audience",
                "medium",
                "No role or audience is stated.",
                "Say who the answer is for, or what expertise the model should apply.",
            )
        )

    if n >= 8 and not mechanical and not GOAL_HINTS.search(text):
        findings.append(
            Finding(
                "no_goal",
                "low",
                "No reason or end use is given.",
                "Add one sentence on what the output is for. Models make better trade-offs "
                "when they know the purpose.",
            )
        )

    if n >= 8 and not FORMAT_HINTS.search(text):
        findings.append(
            Finding(
                "no_format",
                "medium",
                "No output format or length is specified.",
                "State the shape you want: bullets, a table, JSON with named fields, "
                "a word limit.",
            )
        )

    lowered = text.lower()
    vague = [w for w in VAGUE_WORDS if re.search(rf"\b{re.escape(w)}\b", lowered)]
    if len(vague) >= 2:
        findings.append(
            Finding(
                "vague_words",
                "low",
                "Vague wording: " + ", ".join(f'"{w}"' for w in vague[:5]) + ".",
                "Replace each with something checkable, such as a number, a named "
                "criterion or an example.",
            )
        )

    if len(text) > 600 and not DELIMITERS.search(text):
        findings.append(
            Finding(
                "unfenced_input",
                "medium",
                "A long block of text with no delimiters. Pasted material can be "
                "mistaken for instructions.",
                "Wrap pasted content in tags such as <document>...</document> and "
                "tell the model to treat it as data.",
            )
        )

    if SHORT_WORDS.search(text) and LONG_WORDS.search(text):
        findings.append(
            Finding(
                "conflicting_length",
                "medium",
                "The prompt asks for both short and detailed output.",
                "Pick one, or say which part is short and which part is detailed.",
            )
        )

    negatives = len(NEGATIVE.findall(text))
    if negatives >= 3 and negatives * 25 > n:
        findings.append(
            Finding(
                "negative_heavy",
                "low",
                f"{negatives} prohibitions. A list of things to avoid leaves the model "
                "guessing what to do.",
                "Rewrite each rule as what the model should do instead.",
            )
        )

    caps = [w for w in words if len(w) > 2 and w.isupper() and not w.isdigit()]
    if len(caps) >= 3:
        findings.append(
            Finding(
                "shouting",
                "low",
                "Several all-caps words. Recent models follow plain instructions well "
                "and can over-apply emphasis.",
                "Use normal case and put the important rule first.",
            )
        )

    if FACT_REQUEST.search(text) and not FALLBACK_HINTS.search(text):
        findings.append(
            Finding(
                "no_fallback",
                "high",
                "The prompt asks for facts or figures but gives no rule for missing "
                "information.",
                "Add: use only the material provided, and say \"not found\" when the "
                "answer is not there.",
            )
        )

    if n >= 15 and FORMAT_HINTS.search(text) and not EXAMPLE_HINTS.search(text):
        if re.search(r"\b(json|yaml|csv|schema|template|table)\b", text, re.I):
            findings.append(
                Finding(
                    "no_example",
                    "low",
                    "A strict output shape is requested without an example.",
                    "Include one filled-in example of the exact output.",
                )
            )

    score = max(0, 100 - sum(SEVERITY_WEIGHT[f.severity] for f in findings))
    return LintResult(score=score, findings=findings, word_count=n)
