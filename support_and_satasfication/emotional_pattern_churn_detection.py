import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def calculate_time_gaps(timestamps):
    """
    Calculate time differences in seconds between consecutive messages.
    Args:
        timestamps (List[datetime.datetime]): Sorted ascending.
    Returns:
        List[float]: Time gaps in seconds.
    """
    gaps = []
    for i in range(1, len(timestamps)):
        gaps.append((timestamps[i] - timestamps[i-1]).total_seconds())
    return gaps

def tone_decay(sentiment_scores):
    """
    Calculate tone decay as sum of negative changes in sentiment.
    Args:
        sentiment_scores (List[float]): Sentiment scores over time (+1 to -1).
    Returns:
        float: Positive value indicating magnitude of decay.
    """
    diffs = np.diff(sentiment_scores)
    decay = -np.sum(diffs[diffs < 0])  # sum of negative dips
    return decay

def detect_repeated_requests(messages, similarity_threshold=0.85):
    """
    Detect repetitive or very similar messages using cosine similarity of TF-IDF vectors.
    Args:
        messages (List[str]): Messages in conversation.
        similarity_threshold (float): Threshold for repetition detection.
    Returns:
        float: Normalized repetition frequency [0,1].
    """
    if len(messages) < 2:
        return 0.0
    vectorizer = TfidfVectorizer().fit_transform(messages)
    vectors = vectorizer.toarray()
    sim_matrix = cosine_similarity(vectors)
    count_repeats = 0
    total_pairs = 0
    n = len(messages)
    for i in range(n):
        for j in range(i+1, n):
            total_pairs += 1
            if sim_matrix[i][j] >= similarity_threshold:
                count_repeats += 1
    return count_repeats / total_pairs if total_pairs else 0.0

def detect_unanswered_questions(messages):
    """
    Estimate unanswered questions by counting question messages 
    without suitable following response.
    (Simplistic heuristic based on presence of '?' and no reply after).
    Returns a ratio [0,1] of unanswered questions.
    """
    question_indices = [i for i, msg in enumerate(messages) if '?' in msg]
    unanswered = 0
    for idx in question_indices:
        # If last message or next message from same user (no answer)
        if idx == len(messages) - 1:
            unanswered += 1
        else:
            # Heuristic: next message shorter or also question means no real answer
            next_msg = messages[idx + 1]
            if len(next_msg) < 10 or '?' in next_msg:
                unanswered += 1
    return unanswered / len(messages) if messages else 0.0

def detect_drop_off(timestamps, max_idle=600):
    """
    Detect drop-off as any long idle gap exceeding max_idle seconds (default 10 minutes).
    Returns 1.0 if drop-off detected, else 0.0
    """
    gaps = calculate_time_gaps(timestamps)
    for gap in gaps:
        if gap > max_idle:
            return 1.0
    return 0.0

def calculate_churn_risk(timestamps, sentiment_scores, messages):
    """
    Calculate composite churn risk score [0,1] weighted from:
    - Pause durations (with threshold)
    - Tone decay (sentiment deterioration)
    - Repeated requests frequency
    - Unanswered questions ratio
    - Drop-off indicator
    """
    gaps = calculate_time_gaps(timestamps)
    max_gap = max(gaps) if gaps else 0.0
    gap_score = min(max_gap / 300.0, 1.0)  # Normalize by 5 minutes
    
    decay_score = tone_decay(sentiment_scores)
    decay_score = min(decay_score / 2.0, 1.0)  # Normalize expected max decay
    
    repetition_score = detect_repeated_requests(messages)
    
    unanswered_score = detect_unanswered_questions(messages)
    
    dropoff_score = detect_drop_off(timestamps)
    
    # Weighted sum (adjust weights as needed)
    churn_score = (
        0.3 * gap_score +
        0.3 * decay_score +
        0.15 * repetition_score +
        0.15 * unanswered_score +
        0.1 * dropoff_score
    )
    return min(max(churn_score, 0.0), 1.0)

