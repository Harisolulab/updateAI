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


# Logging or tagging example
if __name__ == "__main__":
    chat_message = "I'm really upset about the delay and this is unacceptable!"
    emotion_scores = tag_emotions_in_chat(chat_message)

    print("Chat message:", chat_message)
    print("Emotion scores:")
    for emotion, score in emotion_scores.items():
        print(f" - {emotion}: {score:.3f}")

    # You can apply logic here: if anger or sadness > threshold, escalate/log for training
    if emotion_scores.get("anger", 0) > 0.5 or emotion_scores.get("sadness", 0) > 0.5:
        print("Flagged for escalation or training.")
