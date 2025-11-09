from app.transcriber import transcribe_audio
from app.evaluator import evaluate_transcript

def main():
    audio_path = input("Enter path to audio file: ").strip()
    print("\n🎧 Transcribing audio...\n")
    transcript = transcribe_audio(audio_path)

    print("📝 Transcript:\n", transcript)
    print("\n🔍 Evaluating performance...\n")
    result = evaluate_transcript(transcript)

    if "error" in result:
        print("Error:", result["error"])
        return

    print("✅ Evaluation Results:")
    print(f"Total Words: {result['total_words']}")
    print(f"Filler Ratio: {result['filler_ratio']}")
    print(f"Clarity Ratio: {result['clarity_ratio']}")
    print(f"Sentiment: {result['sentiment']['label']} ({result['sentiment']['compound']})")
    print(f"Final Score: {result['scores']['final']}/100")
    print("\nFeedback:")
    print(result["feedback"])

if __name__ == "__main__":
    
    from app.feedback import generate_supportive_feedback

    transcript = """I recently worked on a project where I built a simple movie recommendation system.
    I used Python and a cosine similarity approach to recommend movies based on user preferences.
    I enjoyed experimenting with different similarity measures and tuning the system to get better results.
    The project helped me understand how recommendation engines work in real life."""

    evaluation = {
        "total_words": 54,
        "filler_ratio": 0.02,
        "clarity_ratio": 0.48,
        "sentiment": {
            "compound": 0.72,
            "label": "Positive"
        },
        "scores": {
            "filler": 30,
            "clarity": 20,
            "sentiment": 30,
            "final": 72
        },
        "feedback": "Clear explanation. Could include more specific details."
    }

    print(generate_supportive_feedback(transcript, evaluation))

