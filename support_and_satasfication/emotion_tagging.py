from transformers import pipeline

# Initialize multi-label emotion classification pipeline with a suitable fine-tuned model
# Example model: "j-hartmann/emotion-english-distilroberta-base"
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True  # Return all class probabilities for multi-label
)


def tag_emotions_in_chat(message: str):
    """
    Apply multi-label emotion classification on a support chat message.
    Returns a dict of emotion labels and their scores.
    """
    results = emotion_classifier(message)
    # results is a list of dicts with label and score
    emotions = {item['label']: item['score'] for item in results[0]}
    return emotions

