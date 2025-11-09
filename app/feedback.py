from openai import OpenAI
client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key="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",
    )

def generate_supportive_feedback(transcript: str, evaluation: dict):
    """
    Generates friendly and supportive interview improvement feedback using an LLM.
    Does NOT affect scoring — only explanation.
    """

    prompt = f"""
    You are a professional interview communication coach. 
    Your goal is to give thoughtful, clear, and structured feedback — 
    not short, not over-simplified, and not casual.

    Write feedback using **this structure** and **clear formatting**:

    1) **Overall Impression**  
    2–3 sentences describing how the candidate sounded overall (clarity, calmness, confidence).

    2) **Strengths**  
    Bullet list of 2–3 positive points.  
    Keep each bullet to one sentence.

    3) **Areas for Improvement**  
    Bullet list of 3–4 improvements.  
    Each bullet must be clear, actionable, and specific.  
    No repeated or vague advice.

    4) **Closing Encouragement**  
    One sentence that is supportive but professional.

    ---

    Candidate's answer:
    {transcript}

    Evaluation metrics to inform your feedback (do not repeat them directly):
    - Filler Ratio: {evaluation['filler_ratio']}
    - Clarity Ratio: {evaluation['clarity_ratio']}
    - Tone: {evaluation['sentiment']['label']}
    - Final Score: {evaluation['scores']['final']}
    """


    response = client.chat.completions.create(
        model="openai/gpt-oss-20b:free",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
if __name__ == "__main__":
  
    transcript = (
        "I recently worked on a project where I built a simple movie recommendation system. "
        "I used Python and a cosine similarity approach to recommend movies based on user preferences. "
        "I enjoyed experimenting with different similarity measures and tuning the system to get better results. "
        "The project helped me understand how recommendation engines work in real life."
    )

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
