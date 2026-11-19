"""Standard benchmark evaluation text suite for Telugu speech synthesis quality.

Owner: P3 — Voice Generation Engineer
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TeluguBenchmarkSentence(BaseModel):
    """Specification of an individual benchmark sentence for Telugu TTS evaluation."""

    id: str = Field(..., description="Unique benchmark sentence identifier (e.g. TTS-01).")
    category: str = Field(..., description="Linguistic category or complexity class.")
    telugu_text: str = Field(..., description="Telugu script text to synthesize.")
    english_translation: str = Field(..., description="English semantic translation.")
    phonetic_focus: str = Field(..., description="Target phonetic features evaluated.")


TELUGU_TTS_BENCHMARKS: list[TeluguBenchmarkSentence] = [
    TeluguBenchmarkSentence(
        id="TTS-01",
        category="Greeting",
        telugu_text="నమస్కారం! వోక్స్ వర్స్ తెలుగు స్వర అనువాద వ్యవస్థకు స్వాగతం.",
        english_translation="Hello! Welcome to the VoxVerse Telugu voice translation system.",
        phonetic_focus="Anusvara (ం), conjuncts (స్వ, స్థ).",
    ),
    TeluguBenchmarkSentence(
        id="TTS-02",
        category="Short / Daily",
        telugu_text="మీరు ఈ రోజు ఎలా ఉన్నారు?",
        english_translation="How are you doing today?",
        phonetic_focus="Interrogative intonation, vowel length.",
    ),
    TeluguBenchmarkSentence(
        id="TTS-03",
        category="Conversational",
        telugu_text="నేను రేపు ఉదయం తొమ్మిది గంటలకు మిమ్మల్ని కలుస్తాను.",
        english_translation="I will meet you tomorrow morning at nine o'clock.",
        phonetic_focus="Gemination (మ్మి, ల్ని), temporal numbers.",
    ),
    TeluguBenchmarkSentence(
        id="TTS-04",
        category="Technical",
        telugu_text="కృత్రిమ మేధస్సు మరియు భాషా సాంకేతికత వేగంగా అభివృద్ధి చెందుతున్నాయి.",
        english_translation="Artificial intelligence and language technology are rapidly developing.",
        phonetic_focus="Sanskrit loanwords, complex conjuncts (త్ర, ధ్స, త్క, వృ).",
    ),
    TeluguBenchmarkSentence(
        id="TTS-05",
        category="Retroflex Plosives",
        telugu_text="చెట్టు కింద ఉన్న చిన్న పిల్లి పాలు తాగుతోంది.",
        english_translation="The little kitten under the tree is drinking milk.",
        phonetic_focus="Retroflex ట / ట్తు contrast with dental త.",
    ),
    TeluguBenchmarkSentence(
        id="TTS-06",
        category="Mahaprana (Aspirated)",
        telugu_text="భారతదేశం గొప్ప చారిత్రక మరియు సాంస్కృతిక వైభవం కలిగిన దేశం.",
        english_translation="India is a country with great historical and cultural glory.",
        phonetic_focus="Aspirated భ (bha), conjuncts (స్కృ).",
    ),
    TeluguBenchmarkSentence(
        id="TTS-07",
        category="Compound Sentence",
        telugu_text="వర్షం పడుతున్నందున మేము బయటకు వెళ్ళలేకపోయాము, కానీ ఇంట్లోనే చదువుకున్నాము.",
        english_translation="Because it was raining we could not go out, but studied at home.",
        phonetic_focus="Subordinate clause pause cadence.",
    ),
    TeluguBenchmarkSentence(
        id="TTS-08",
        category="Interrogative / Inversion",
        telugu_text="ఈ కొత్త ప్రాజెక్ట్ గురించి మీ అభిప్రాయం ఏమిటి?",
        english_translation="What is your opinion about this new project?",
        phonetic_focus="Pitch rise at utterance boundary (ఏమిటి).",
    ),
    TeluguBenchmarkSentence(
        id="TTS-09",
        category="Numerals & Currency",
        telugu_text="ఆ పుస్తకం ఖరీదు నాలుగు వందల యాభై రూపాయలు.",
        english_translation="The price of that book is four hundred and fifty rupees.",
        phonetic_focus="Colloquial numbers, aspirate ఖ.",
    ),
    TeluguBenchmarkSentence(
        id="TTS-10",
        category="Narrative / Long",
        telugu_text="సాంకేతిక పరిజ్ఞానం మానవ జీవితాన్ని ఎంతో సులభతరం చేసింది, అలాగే సమాచార మార్పిడిని వేగవంతం చేసింది.",
        english_translation="Technology has greatly simplified human life and accelerated information exchange.",
        phonetic_focus="16-word continuous rhythm, breath phrasing.",
    ),
]


def get_benchmark_suite() -> list[TeluguBenchmarkSentence]:
    """Return the fixed 10-sentence Telugu TTS benchmark suite."""
    return list(TELUGU_TTS_BENCHMARKS)
