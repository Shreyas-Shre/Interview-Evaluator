# import spacy
# from nltk.sentiment import SentimentIntensityAnalyzer
# import re

# # Load spaCy model for POS tagging
# nlp = spacy.load("en_core_web_sm")

# # Initialize sentiment analyzer
# sentiment_analyzer = SentimentIntensityAnalyzer()

# def evaluate_transcript(transcript: str):
#     if not transcript.strip():
#         return {"error": "Empty transcript"}

#     # --- Step 1: Basic counts ---
#     words = transcript.lower().split()
#     total_words = len(words)
#     if total_words == 0:
#         return {"error": "Transcript has no words"}



#     # Normalize transcript (lowercase, remove punctuation, normalize spaces)
#     norm = transcript.lower()
#     norm = re.sub(r"[^\w\s']", " ", norm)   # Replace punctuation with space
#     norm = re.sub(r"\s+", " ", norm).strip()

#     # Multi-word filler phrases (regex patterns)
#     phrase_patterns = [
#         r"\byou\s+know\b",
#         r"\bi\s+mean\b",
#         r"\bkind\s+of\b",
#         r"\bsort\s+of\b",
#         r"\bso\s+yeah\b"
#     ]
#     phrase_count = sum(len(re.findall(p, norm)) for p in phrase_patterns)

#     # Single word fillers (token level)
#     single_fillers = {"um", "uh", "like", "huh", "ah", "so", "well", "right", "actually", "basically", "literally"}
#     tokens = norm.split()
#     single_count = sum(tokens.count(w) for w in single_fillers)

#     filler_count = phrase_count + single_count
#     filler_ratio = filler_count / max(total_words, 1)


#     # Score (max 30)
#     if filler_ratio < 0.03:
#         filler_score = 30
#     elif filler_ratio < 0.07:
#         filler_score = 20
#     else:
#         filler_score = 10

#     # --- Step 3: POS tagging clarity ---
#     doc = nlp(transcript)
#     content_words = sum(1 for token in doc if token.pos_ in ["NOUN", "VERB"])
#     clarity_ratio = content_words / total_words

#     # Score (max 30)
#     if clarity_ratio > 0.55:
#         clarity_score = 30
#     elif clarity_ratio > 0.4:
#         clarity_score = 20
#     else:
#         clarity_score = 10

#     # --- Step 4: Sentiment analysis ---
#     sentiment_scores = sentiment_analyzer.polarity_scores(transcript)
#     compound = sentiment_scores["compound"]

#     if compound >= 0.6:
#         sentiment_score = 30
#         sentiment_label = "Positive"
#     elif compound >= 0.2:
#         sentiment_score = 24
#         sentiment_label = "Neutral/Moderate"
#     else:
#         sentiment_score = 15
#         sentiment_label = "Negative"

#     # --- Step 5: Final score ---
    
#     weighted_final = round(
#         (filler_score / 30) * 40 +
#         (clarity_score / 30) * 40 +
#         (sentiment_score / 30) * 20
#         )

#     # --- Step 6: Feedback generation ---
#     feedback_parts = []
#     if filler_score < 20:
#         feedback_parts.append("Try to reduce filler words like 'um' or 'like'.")
#     if clarity_score < 20:
#         feedback_parts.append("Use more clear and direct sentences.")
#     if sentiment_score < 30:
#         feedback_parts.append("Work on a more positive or confident tone.")
#     if not feedback_parts:
#         feedback_parts.append("Excellent clarity, confidence, and tone!")

#     feedback = " ".join(feedback_parts)

#     # --- Step 7: Result ---
#     return {
#         "total_words": total_words,
#         "filler_ratio": round(filler_ratio, 3),
#         "clarity_ratio": round(clarity_ratio, 3),
#         "sentiment": {
#             "compound": compound,
#             "label": sentiment_label
#         },
#         "scores": {
#             "filler": filler_score,
#             "clarity": clarity_score,
#             "sentiment": sentiment_score,
#             "final": weighted_final
#         },
#         "feedback": feedback
#     }


# # Example usage:
# if __name__ == "__main__":
#     sample_text = "Um I think I actually did a good job in my last project. It was challenging but rewarding."
#     result = evaluate_transcript(sample_text)
#     print(result)
# app/evaluator.py
import re
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer

# Load spaCy model for POS tagging
nlp = spacy.load("en_core_web_sm")

# Initialize sentiment analyzer (assumes vader_lexicon is available)
sentiment_analyzer = SentimentIntensityAnalyzer()


# ---------- Helpers ----------
PHRASE_PATTERNS = [
    r"\byou\s+know\b",
    r"\bi\s+mean\b",
    r"\bkind\s+of\b",
    r"\bsort\s+of\b",
    r"\bso\s+yeah\b",
    r"\bkinda\b",
    r"\bsorta\b",
]

SINGLE_FILLERS = {
    "um", "uh", "like", "huh", "ah", "so", "well", "right",
    "actually", "basically", "literally"
}


def _strip_leading_question(text: str) -> str:
    """Remove the interviewer's leading question if the first sentence ends with '?'."""
    qpos = text.find("?")
    if 0 <= qpos < 200:  # simple heuristic
        return text[qpos + 1 :].strip()
    return text


def _normalize(text: str) -> str:
    """Lowercase, remove punctuation (keep apostrophes), collapse spaces."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _count_fillers(normalized_text: str) -> int:
    """Count single and multi-word fillers robustly on normalized text."""
    # Multi-word phrases
    phrase_count = sum(len(re.findall(p, normalized_text)) for p in PHRASE_PATTERNS)
    # Single tokens
    tokens = normalized_text.split()
    single_count = sum(tokens.count(w) for w in SINGLE_FILLERS)
    return phrase_count + single_count


# ---------- Main API ----------
def evaluate_transcript(transcript: str):
    if not transcript or not transcript.strip():
        return {"error": "Empty transcript"}

    # Remove the interviewer's question if present
    transcript = _strip_leading_question(transcript)

    # Normalized string for counting
    norm = _normalize(transcript)

    # --- Step 1: Basic counts ---
    total_words = len(re.findall(r"\b\w+\b", norm))
    if total_words == 0:
        return {"error": "Transcript has no words"}

    # --- Step 2: Filler word analysis (robust) ---
    filler_count = _count_fillers(norm)
    filler_ratio = filler_count / max(total_words, 1)

    # Score (max 30)
    if filler_ratio < 0.03:
        filler_score = 30
    elif filler_ratio < 0.07:
        filler_score = 20
    else:
        filler_score = 10

    # --- Step 3: POS tagging clarity ---
    doc = nlp(transcript)
    content_words = sum(1 for token in doc if token.pos_ in ["NOUN", "VERB"])
    clarity_ratio = content_words / max(total_words, 1)

    # Score (max 30)
    if clarity_ratio > 0.55:
        clarity_score = 30
    elif clarity_ratio > 0.40:
        clarity_score = 20
    else:
        clarity_score = 10

    # --- Step 4: Sentiment analysis (less generous mapping) ---
    sentiment_scores = sentiment_analyzer.polarity_scores(transcript)
    compound = sentiment_scores["compound"]

    if compound >= 0.60:
        sentiment_score = 30
        sentiment_label = "Positive"
    elif compound >= 0.20:
        sentiment_score = 24
        sentiment_label = "Neutral/Moderate"
    else:
        sentiment_score = 15
        sentiment_label = "Negative"

    # --- Step 5: Weighted final score (Filler 40, Clarity 40, Sentiment 20) ---
    weighted_final = round(
        (filler_score / 30) * 40 +
        (clarity_score / 30) * 40 +
        (sentiment_score / 30) * 20
    )

    # --- Step 6: Feedback generation ---
    feedback_parts = []
    if filler_score < 20:
        feedback_parts.append("Try to reduce filler words and hesitation phrases.")
    if clarity_score < 20:
        feedback_parts.append("Use more clear, specific, and action-oriented sentences.")
    if sentiment_score < 24:
        feedback_parts.append("Aim for a more confident and positive tone.")
    if not feedback_parts:
        feedback_parts.append("Excellent clarity, confidence, and tone!")

    feedback = " ".join(feedback_parts)

    # --- Step 7: Result ---
    return {
        "total_words": total_words,
        "filler_ratio": round(filler_ratio, 3),
        "clarity_ratio": round(clarity_ratio, 3),
        "sentiment": {
            "compound": round(compound, 4),
            "label": sentiment_label
        },
        "scores": {
            "filler": filler_score,
            "clarity": clarity_score,
            "sentiment": sentiment_score,
            "final": weighted_final
        },
        "feedback": feedback
    }


# Example usage:
if __name__ == "__main__":
    sample_text = (
        "Can you explain your project? Um, I mean, I worked on a machine learning project. "
        "It was basically a pipeline for data processing. I optimized training and evaluated results."
    )
    result = evaluate_transcript(sample_text)
    print(result)

