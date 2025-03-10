import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import numpy as np

def download_whisper_model(model_name="openai/whisper-small"):
    """Download and test Whisper model"""
    print(f"Downloading {model_name}...")
    
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Create simple sine wave audio for testing
    sample_rate = 16000
    duration = 2  # seconds
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Test transcription (this will download the model if not already downloaded)
    input_features = processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_features
    
    print(f"Testing model with sample audio...")
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    print(f"Test transcription: {transcription}")
    print(f"Model downloaded and tested successfully!")
    
    return model, processor

if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Create models directory if it doesn't exist
    os.makedirs("data/models", exist_ok=True)
    
    # Download Whisper model
    download_whisper_model()