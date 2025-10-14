import whisper
import re
import spacy

# Load spacy model once
nlp = spacy.load("en_core_web_sm")

# Common filler words
FILLER_WORDS = {"um", "uh", "like", "you know", "actually", "basically", "so"}

def transcribe_audio(file_path):
    print("Loading whisper model...")
    model = whisper.load_model("medium")

    print("Transcribing audio...")
    result = model.transcribe(file_path)

    print("Transcription complete.")
    return result['text']

def clean_transcript(text: str):
    """
    Cleans transcript, detects filler words, and optionally performs POS tagging.
    """
    # 1. Basic cleanup
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 2. Tokenize and process
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if not token.is_space]

    # 3. Filler word detection
    filler_counts = {}
    for token in tokens:
        if token in FILLER_WORDS:
            filler_counts[token] = filler_counts.get(token, 0) + 1

    # 4. POS tagging summary (optional but useful)
    pos_counts = {}
    for token in doc:
        if token.pos_ in ["NOUN", "VERB", "PRON", "ADJ", "ADV"]:
            pos_counts[token.pos_] = pos_counts.get(token.pos_, 0) + 1

    return {
        "clean_text": " ".join(tokens),
        "tokens": tokens,
        "filler_count": sum(filler_counts.values()),
        "filler_words": filler_counts,
        "pos_counts": pos_counts  # <-- optional NLP clarity metrics
    }

if __name__ == "__main__":
    audio_file = "./test_data/john.mp3"
    
    # Step 1: Transcribe
    transcript = transcribe_audio(audio_file)
    print("\nRaw Transcript:\n", transcript)

    # Step 2: Clean & analyze
    processed = clean_transcript(transcript)
    print("\nProcessed Transcript:", processed["clean_text"])
    print("Filler Count:", processed["filler_count"])
    print("Filler Words:", processed["filler_words"])
    print("POS Tagging:",processed["pos_counts"])
