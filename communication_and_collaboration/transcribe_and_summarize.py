import asyncio
import os
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv
from pydub import AudioSegment
import whisper
from pyannote.audio import Pipeline
from langchain_community.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Load environment variables from .env file
load_dotenv()

# Load API keys securely
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")  # Hugging Face token for pyannote

# Hugging Face token for pyannote

def check_api_keys():
    """Ensure required API keys are configured."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in environment variables or .env file")
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN (Hugging Face token) not set in environment variables or .env file")

async def transcribe_audio(audio_path: str, whisper_model) -> str:
    """
    Convert audio to 16kHz mono WAV and transcribe with Whisper large model.
    """
    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    wav_path = f"{audio_path}_16k.wav"
    audio.export(wav_path, format="wav")
    result = whisper_model.transcribe(wav_path)
    return result["text"]

async def diarize_audio(audio_path: str, diarization_pipeline) -> List[Dict]:
    """
    Perform speaker diarization and return segments with speaker labels.
    """
    diarization = diarization_pipeline(audio_path)
    return [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]

def segment_transcription_by_speaker(
    transcription: str, segments: List[Dict]
) -> List[Dict]:
    """
    Simple heuristic to segment text roughly aligned per diarized speaker.
    """
    if not segments:
        return []
    total_chars = len(transcription)
    per_segment = max(1, total_chars // len(segments))
    speaker_segments, pos = [], 0
    for seg in segments:
        segment_text = transcription[pos:pos + per_segment]
        speaker_segments.append({
            "speaker": seg["speaker"],
            "text": segment_text.strip()
        })
        pos += per_segment
    if pos < total_chars:
        speaker_segments[-1]["text"] += " " + transcription[pos:]
    return speaker_segments

async def summarize_conversation(speaker_segments: List[Dict], chat_model) -> str:
    """
    Generate bullet-point summary using OpenAI chat model.
    """
    if not speaker_segments:
        return "No conversation data to summarize."
    conversation = "\n".join(
        f"Speaker {seg['speaker']}: {seg['text']}" for seg in speaker_segments
    )
    prompt = (
        "You are a meeting note assistant.\n"
        "Summarize the following conversation, highlighting key points, decisions, and action items.\n"
        "Provide the response as bullet points."
    )
    response = chat_model.invoke(
        [SystemMessage(content=prompt), HumanMessage(content=conversation)]
    )
    return response.content.strip()

async def transcribe_diarize_and_summarize(audio_path: str) -> str:
    """
    Complete pipeline: transcribe audio, diarize speakers, then summarize.
    """
    check_api_keys()
    whisper_model = whisper.load_model("large")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization", use_auth_token=HF_TOKEN
    )
    chat_model = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=OPENAI_API_KEY,
    )
    transcription = await transcribe_audio(audio_path, whisper_model)
    diarization_segments = await diarize_audio(audio_path, diarization_pipeline)
    speaker_segments = segment_transcription_by_speaker(transcription, diarization_segments)
    summary = await summarize_conversation(speaker_segments, chat_model)
    return summary


