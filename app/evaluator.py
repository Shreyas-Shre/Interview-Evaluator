import spacy
from nltk.sentiment import SentimentIntensityAnalyzer

# Load spaCy model for POS tagging
nlp = spacy.load("en_core_web_sm")

# Initialize sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

def evaluate_transcript(transcript: str):
    if not transcript.strip():
        return {"error": "Empty transcript"}

    # --- Step 1: Basic counts ---
    words = transcript.lower().split()
    total_words = len(words)
    if total_words == 0:
        return {"error": "Transcript has no words"}

    # --- Step 2: Filler word analysis ---
    filler_words = {"um", "uh", "like", "you know", "actually", "basically", "literally"}
    filler_count = sum(words.count(w) for w in filler_words)
    filler_ratio = filler_count / total_words

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
    clarity_ratio = content_words / total_words

    # Score (max 30)
    if clarity_ratio > 0.55:
        clarity_score = 30
    elif clarity_ratio > 0.4:
        clarity_score = 20
    else:
        clarity_score = 10

    # --- Step 4: Sentiment analysis ---
    sentiment_scores = sentiment_analyzer.polarity_scores(transcript)
    compound = sentiment_scores["compound"]

    if compound > 0.5:
        sentiment_score = 40
        sentiment_label = "Positive"
    elif compound > 0.2:
        sentiment_score = 30
        sentiment_label = "Neutral/Moderate"
    else:
        sentiment_score = 15
        sentiment_label = "Negative"

    # --- Step 5: Final score ---
    total_score = filler_score + clarity_score + sentiment_score

    # --- Step 6: Feedback generation ---
    feedback_parts = []
    if filler_score < 20:
        feedback_parts.append("Try to reduce filler words like 'um' or 'like'.")
    if clarity_score < 20:
        feedback_parts.append("Use more clear and direct sentences.")
    if sentiment_score < 30:
        feedback_parts.append("Work on a more positive or confident tone.")
    if not feedback_parts:
        feedback_parts.append("Excellent clarity, confidence, and tone!")

    feedback = " ".join(feedback_parts)

    # --- Step 7: Result ---
    return {
        "total_words": total_words,
        "filler_ratio": round(filler_ratio, 3),
        "clarity_ratio": round(clarity_ratio, 3),
        "sentiment": {
            "compound": compound,
            "label": sentiment_label
        },
        "scores": {
            "filler": filler_score,
            "clarity": clarity_score,
            "sentiment": sentiment_score,
            "final": total_score
        },
        "feedback": feedback
    }


# Example usage:
if __name__ == "__main__":
    sample_text = "Um I think I actually did a good job in my last project. It was challenging but rewarding."
    result = evaluate_transcript(sample_text)
    print(result)
