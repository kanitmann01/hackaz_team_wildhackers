import os
import sys
import numpy as np
import wave
import torch

# Add parent directory to path to allow for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the EnhancedStreamingTTS
from enhanced_tts import EnhancedStreamingTTS

def test_tts_with_language(language_code, text=None):
    """
    Test TTS generation with a specific language.
    
    Args:
        language_code: ISO or NLLB language code
        text: Text to synthesize (or use default)
    """
    # Set default texts for different languages
    default_texts = {
        "en": "This is a test of the speech synthesis system.",
        "eng_Latn": "This is a test of the speech synthesis system.",
        "es": "Esta es una prueba del sistema de síntesis de voz.",
        "spa_Latn": "Esta es una prueba del sistema de síntesis de voz.",
        "fr": "Ceci est un test du système de synthèse vocale.",
        "fra_Latn": "Ceci est un test du système de synthèse vocale.",
        "de": "Dies ist ein Test des Sprachsynthesesystems.",
        "deu_Latn": "Dies ist ein Test des Sprachsynthesesystems.",
        "it": "Questo è un test del sistema di sintesi vocale.",
        "ita_Latn": "Questo è un test del sistema di sintesi vocale.",
        "pt": "Este é um teste do sistema de síntese de fala.",
        "por_Latn": "Este é um teste do sistema de síntese de fala.",
        "ru": "Это тест системы синтеза речи.",
        "rus_Cyrl": "Это тест системы синтеза речи.",
        "zh": "这是语音合成系统的测试。",
        "zho_Hans": "这是语音合成系统的测试。",
        "ja": "これは音声合成システムのテストです。",
        "jpn_Jpan": "これは音声合成システムのテストです。",
        "ko": "이것은 음성 합성 시스템의 테스트입니다.",
        "kor_Hang": "이것은 음성 합성 시스템의 테스트입니다.",
        "ar": "هذا اختبار لنظام توليف الكلام.",
        "ara_Arab": "هذا اختبار لنظام توليف الكلام.",
        "hi": "यह वाक् संश्लेषण प्रणाली का परीक्षण है।",
        "hin_Deva": "यह वाक् संश्लेषण प्रणाली का परीक्षण है।"
    }
    
    # Use provided text or default
    text = text or default_texts.get(language_code, "This is a test of the speech synthesis system.")
    
    print(f"\n{'='*50}")
    print(f"TESTING TTS WITH LANGUAGE: {language_code}")
    print(f"TEXT: {text}")
    print(f"{'='*50}")
    
    # Initialize TTS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tts = EnhancedStreamingTTS(device=device)
    print("TTS initialized")
    
    # Set language
    success = tts.set_language(language_code)
    print(f"Set language to {language_code}: {'Success' if success else 'Failed'}")
    print(f"Actual language code in use: {tts.language_code}")
    print(f"TTS model: {tts.model_name}")
    
    # Generate speech
    print("Generating speech...")
    audio = tts.synthesize_speech(text)
    
    # Print audio info
    if audio is None:
        print("ERROR: Generated audio is None")
        return False
    
    print(f"Generated {len(audio)} audio samples")
    print(f"Audio stats: min={np.min(audio):.4f}, max={np.max(audio):.4f}, mean={np.mean(audio):.4f}")
    
    # Save audio to file
    os.makedirs("data/samples", exist_ok=True)
    filename = f"data/samples/tts_test_{language_code.replace('_', '-')}.wav"
    
    # Check if we have actual audio
    if len(audio) == 0:
        print("ERROR: Generated audio is empty")
        return False
    
    # Save to WAV file
    audio_int16 = (audio * 32767).astype(np.int16)
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(16000)  # Standard rate for TTS
        wf.writeframes(audio_int16.tobytes())
    
    print(f"Audio saved to {filename}")
    
    # Try playing the audio
    try:
        import sounddevice as sd
        sd.play(audio, 16000)
        sd.wait()
        print("Audio played successfully")
    except Exception as e:
        print(f"Error playing audio: {e}")
    
    print(f"{'='*50}")
    
    return len(audio) > 0

def test_mt_tts_integration():
    """Test the integration of MT and TTS components."""
    print("\n\n" + "="*50)
    print("TESTING MT+TTS INTEGRATION")
    print("="*50)
    
    try:
        # Import MT component
        from src.mt.streaming_mt import StreamingTranslator
        
        # Create MT model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        mt = StreamingTranslator(
            source_lang="eng_Latn", 
            target_lang="spa_Latn", 
            device=device
        )
        print("MT model initialized")
        
        # Create TTS model
        tts = EnhancedStreamingTTS(device=device)
        tts.set_language("spa_Latn")  # Set to Spanish
        print("TTS model initialized")
        
        # Test text
        text = "This is a test of the translation and speech synthesis system."
        print(f"Source text: {text}")
        
        # Translate
        print("Translating...")
        translation = mt.translate_chunk(text, force=True)
        translated_text = translation['full_text']
        print(f"Translation: {translated_text}")
        
        # Synthesize speech
        print("Synthesizing speech...")
        audio = tts.synthesize_speech(translated_text)
        
        # Save audio
        os.makedirs("data/samples", exist_ok=True)
        filename = "data/samples/mt_tts_integration.wav"
        
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())
        
        print(f"Audio saved to {filename}")
        
        # Try playing the audio
        try:
            import sounddevice as sd
            sd.play(audio, 16000)
            sd.wait()
            print("Audio played successfully")
        except Exception as e:
            print(f"Error playing audio: {e}")
        
        print(f"{'='*50}")
        return True
    
    except Exception as e:
        import traceback
        print(f"Error in MT+TTS integration test: {e}")
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Test basic TTS
    test_tts_with_language("en")
    
    # Test with Spanish
    test_tts_with_language("es")
    
    # Test with NLLB code
    test_tts_with_language("fra_Latn")
    
    # Test MT+TTS integration
    test_mt_tts_integration()