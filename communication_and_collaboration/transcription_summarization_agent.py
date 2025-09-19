import os
import uuid
import logging
from typing import Dict, Any, List, Optional

# Optional heavy deps (lazy import)
try:
    import whisper  # type: ignore
except Exception:
    whisper = None  # type: ignore

try:
    from pyannote.audio import Pipeline  # type: ignore
except Exception:
    Pipeline = None  # type: ignore

# OpenAI GPT-4o wrapper
from langchain_openai import ChatOpenAI

logger = logging.getLogger("TranscriptionSummarizationAgent")

# Lazy init for heavy models
ASR_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
_ASR_MODEL = None
_DIARIZATION_PIPELINE = None

def get_asr_model():
    global _ASR_MODEL
    if _ASR_MODEL is None:
        if whisper is None:
            raise ImportError("whisper is not installed")
        _ASR_MODEL = whisper.load_model(ASR_MODEL_NAME)
    return _ASR_MODEL

def get_diarization_pipeline() -> Optional[Any]:
    # Use HF token if available
    global _DIARIZATION_PIPELINE
    if _DIARIZATION_PIPELINE is None:
        if Pipeline is None:
            logger.warning("pyannote.audio not installed; skipping diarization")
            return None
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        try:
            if hf_token:
                os.environ["HF_TOKEN"] = hf_token
            _DIARIZATION_PIPELINE = Pipeline.from_pretrained("pyannote/speaker-diarization")
        except Exception as e:
            logger.warning(f"Could not initialize diarization pipeline: {e}")
            _DIARIZATION_PIPELINE = None
    return _DIARIZATION_PIPELINE

# Initialize OpenAI GPT-4o for summarization
def get_llm():
    return ChatOpenAI(
        temperature=0.2,
        model="gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

def transcribe_audio_with_diarization(audio_path: str) -> List[Dict[str, Any]]:
    """Transcribes audio and performs speaker diarization when available.
    Returns list of speaker segments with text.
    Each segment dict: {speaker: str, start: float, end: float, text: str}
    """
    # Try diarization
    diarizer = get_diarization_pipeline()
    diarization = None
    if diarizer is not None:
        logger.info(f"Starting diarization on audio {audio_path}")
        try:
            diarization = diarizer(audio_path)
        except Exception as e:
            logger.warning(f"Diarization failed, continuing without it: {e}")
            diarization = None

    # Whisper transcription with detailed segments
    logger.info("Starting ASR transcription")
    asr = get_asr_model()
    result = asr.transcribe(audio_path, word_timestamps=True)

    # Assign words to speakers if diarization available; otherwise single speaker
    segments: List[Dict[str, Any]] = []
    words_segments = result.get("segments", [])

    if diarization is None:
        # Fallback: single speaker transcript
        full_text = " ".join(w["word"] for seg in words_segments for w in seg.get("words", []))
        if full_text:
            segments.append({
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": result.get("duration", 0.0),
                "text": full_text,
            })
        logger.info("Diarization unavailable; produced 1 combined segment")
        return segments

    # With diarization: map words into speaker turns by time overlap
    for turn in diarization.itertracks(yield_label=True):
        segment, speaker = turn[0], turn[1]
        speaker_words = [
            w
            for seg in words_segments
            for w in seg.get("words", [])
            if (w.get("start", 0) >= segment.start and w.get("end", 0) <= segment.end)
        ]
        text = " ".join(w.get("word", "") for w in speaker_words)
        if text:
            segments.append({
                "speaker": speaker,
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
            })
    logger.info(f"Diarization produced {len(segments)} speaker segments")
    return segments

def generate_meeting_summary(speaker_segments: List[Dict[str, Any]]) -> str:
    """
    Uses GPT-4o to generate a structured meeting summary from transcribed speaker segments.
    Includes key points, action items, decisions.
    """

    llm = get_llm()

    # Build prompt from speaker segments
    transcript_text = ""
    for seg in speaker_segments:
        transcript_text += f"Speaker {seg['speaker']}: {seg['text']}\n"

    prompt = (
        "You are an assistant that summarizes meetings.\n"
        "Given the following multi-speaker transcript, create a concise summary with:\n"
        "- Key points\n"
        "- Action items\n"
        "- Decisions made\n"
        "Be accurate and brief.\n\n"
        f"Transcript:\n{transcript_text}\n"
        "Summary:"
    )

    response = llm.invoke(prompt)

    summary = getattr(response, "content", str(response))
    return summary

def transcribe_and_summarize(audio_path: str, require_accuracy: float = 0.85) -> Dict[str, Any]:
    """
    Full pipeline: audio -> diarization + transcription -> summary.
    Returns structured dict with segments and summary.
    """
    speaker_segments = transcribe_audio_with_diarization(audio_path)
    summary = generate_meeting_summary(speaker_segments)

    return {
        "speaker_segments": speaker_segments,
        "summary": summary,
        "accuracy_estimate": require_accuracy  # stub; estimate can come from ASR/confidence scores
    }
