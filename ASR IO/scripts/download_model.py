import os
import sys
import torch
import time
import argparse

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import model classes
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoProcessor,
    AutoModel
)
import numpy as np

def download_wav2vec_model(model_name="facebook/wav2vec2-large-960h-lv60-self", device=None):
    """
    Download and test Wav2Vec 2.0 ASR model.
    
    Args:
        model_name: Name of the Wav2Vec model
        device: Computation device ('cuda' or 'cpu')
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Downloading {model_name} on {device}...")
    start_time = time.time()
    
    # Download model and processor
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    
    # Create simple sine wave audio for testing
    sample_rate = 16000
    duration = 2  # seconds
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Test transcription
    print(f"Testing {model_name} with sample audio...")
    input_values = processor(audio, sampling_rate=sample_rate, return_tensors="pt").input_values.to(device)
    
    with torch.no_grad():
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = processor.batch_decode(predicted_ids)[0]
    
    end_time = time.time()
    print(f"ASR Model test transcription: {transcription}")
    print(f"ASR Model downloaded and tested in {end_time - start_time:.2f} seconds")
    
    return model, processor

def download_nllb_model(model_name="facebook/nllb-200-distilled-600M", device=None):
    """
    Download and test NLLB-200 translation model.
    
    Args:
        model_name: Name of the translation model
        device: Computation device ('cuda' or 'cpu')
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Downloading {model_name} on {device}...")
    start_time = time.time()
    
    # Download model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    # Test translation
    print(f"Testing {model_name} with sample text...")
    text = "Hello, how are you today?"
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Generate translation with specified target language
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id["spa_Latn"]
        )
        
        # Decode translation
        translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    
    end_time = time.time()
    print(f"MT Model test translation (eng->spa): '{text}' -> '{translation}'")
    print(f"MT Model downloaded and tested in {end_time - start_time:.2f} seconds")
    
    return model, tokenizer

def download_vits_model(model_name="facebook/mms-tts-eng", device=None):
    """
    Download and test VITS/MMS-TTS model.
    
    Args:
        model_name: Name of the TTS model
        device: Computation device ('cuda' or 'cpu')
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    print(f"Downloading {model_name} on {device}...")
    start_time = time.time()
    
    try:
        # Download model and processor
        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        
        # Test TTS
        print(f"Testing {model_name} with sample text...")
        text = "This is a test of the text to speech system."
        
        # Prepare inputs
        inputs = processor(text=text, return_tensors="pt").to(device)
        
        # Generate speech
        with torch.no_grad():
            output = model(**inputs, language="eng")
            speech = output.waveform.cpu().numpy()
        
        end_time = time.time()
        print(f"TTS Model test successful. Generated {speech.shape[1]} samples.")
        print(f"TTS Model downloaded and tested in {end_time - start_time:.2f} seconds")
        
        return model, processor
        
    except Exception as e:
        print(f"Error downloading TTS model: {e}")
        return None, None

def main():
    """
    Download and test all models required for the translation system.
    """
    parser = argparse.ArgumentParser(description="Download models for real-time translation")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    parser.add_argument("--asr-model", default="facebook/wav2vec2-large-960h-lv60-self", help="ASR model name")
    parser.add_argument("--mt-model", default="facebook/nllb-200-distilled-600M", help="MT model name")
    parser.add_argument("--tts-model", default="facebook/mms-tts-eng", help="TTS model name")
    
    args = parser.parse_args()
    
    # Set device
    if args.no_gpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available() and not args.no_gpu:
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Create models directory if it doesn't exist
    os.makedirs("data/models", exist_ok=True)
    
    # Total time tracking
    total_start_time = time.time()
    
    # Download models
    download_wav2vec_model(args.asr_model, device)
    download_nllb_model(args.mt_model, device)
    download_vits_model(args.tts_model, device)
    
    total_end_time = time.time()
    print(f"All models downloaded in {total_end_time - total_start_time:.2f} seconds")

if __name__ == "__main__":
    main()