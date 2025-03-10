import sys
import os
import time

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.asr.whisper_asr import WhisperASR
from src.audio.audio_io import AudioIO

def test_asr_pipeline():
    """Test the ASR pipeline with live audio recording"""
    print("Testing ASR pipeline with live recording")
    
    # Initialize components
    audio_io = AudioIO(sample_rate=16000)
    asr = WhisperASR(model_name="openai/whisper-small")
    
    # Record audio
    print("Please speak a sentence when recording starts...")
    time.sleep(1)  # Give user time to prepare
    audio_data = audio_io.record_audio(duration=5)
    
    audio_io.save_audio(audio_data, "data/samples/test_asr_recording.wav")
    
    print("Transcribing...")
    start_time = time.time()
    transcription = asr.transcribe_audio(audio_data)
    end_time = time.time()
    
    print(f"Transcription: {transcription}")
    print(f"Transcription time: {end_time - start_time:.2f} seconds")
    
    return transcription

if __name__ == "__main__":
    test_asr_pipeline()