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
    main()
