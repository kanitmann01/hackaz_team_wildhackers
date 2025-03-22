import os
import sys
import time
import numpy as np
import torch
import argparse
import threading

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import project components
from src.asr.streaming_asr import StreamingASR
from src.mt.streaming_mt import StreamingTranslator
from src.tts.streaming_tts import StreamingTTS
from src.audio.streaming_audio import StreamingAudioCapture, StreamingAudioPlayback
from src.pipeline.realtime_pipeline import RealTimeTranslationPipeline

def test_asr_component(device=None):
    """Test the ASR component with synthetic audio."""
    print("\n--- Testing ASR Component ---")
    
    try:
        # Initialize ASR
        asr = StreamingASR(model_name="openai/whisper-small", device=device)
        
        # Create test audio (sine wave)
        sample_rate = 16000
        duration = 2  # seconds
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Transcribe
        print("Transcribing test audio...")
        start_time = time.time()
        result = asr.transcribe_chunk(audio, sample_rate)
        end_time = time.time()
        
        print(f"Transcription result: {result['full_text']}")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        
        print("ASR component test successful!")
        return True
        
    except Exception as e:
        print(f"ASR test failed: {e}")
        return False

def test_mt_component(device=None):
    """Test the MT component with sample text."""
    print("\n--- Testing MT Component ---")
    
    try:
        # Initialize MT
        translator = StreamingTranslator(
            source_lang="en", 
            target_lang="es", 
            device=device
        )
        
        # Test translation
        test_text = "Hello, this is a test of the translation system."
        print(f"Translating: '{test_text}'")
        
        start_time = time.time()
        result = translator.translate_chunk(test_text)
        end_time = time.time()
        
        print(f"Translation result: '{result['full_text']}'")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        
        print("MT component test successful!")
        return True
        
    except Exception as e:
        print(f"MT test failed: {e}")
        return False

def test_tts_component(device=None):
    """Test the TTS component with sample text."""
    print("\n--- Testing TTS Component ---")
    
    try:
        # Initialize TTS
        tts = StreamingTTS(device=device)
        
        # Test audio generation
        test_text = "This is a test of the speech synthesis system."
        print(f"Synthesizing: '{test_text}'")
        
        start_time = time.time()
        audio = tts.synthesize_speech(test_text, play_callback=None)
        end_time = time.time()
        
        if audio is not None:
            print(f"Generated audio length: {len(audio)} samples")
            print(f"Processing time: {end_time - start_time:.2f} seconds")
            
            # Create output directory if it doesn't exist
            os.makedirs("data/samples", exist_ok=True)
            
            # Save audio to file
            output_file = "data/samples/tts_test_output.wav"
            import wave
            with wave.open(output_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(16000)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())
            
            print(f"Audio saved to {output_file}")
            
            print("TTS component test successful!")
            return True
        else:
            print("No audio generated")
            return False
        
    except Exception as e:
        print(f"TTS test failed: {e}")
        return False

def test_audio_components():
    """Test audio capture and playback components."""
    print("\n--- Testing Audio Components ---")
    
    try:
        # Test audio playback
        audio_playback = StreamingAudioPlayback()
        
        # Generate test audio (sine wave)
        sample_rate = 16000
        duration = 2  # seconds
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        print("Starting audio playback...")
        audio_playback.start()
        
        print("Playing test audio...")
        audio_playback.play(audio)
        
        time.sleep(2)  # Wait for audio to finish
        
        audio_playback.stop()
        print("Audio playback test completed")
        
        # Test audio capture (brief recording)
        print("Testing audio capture (3 seconds)...")
        
        captured_audio = []
        capture_event = threading.Event()
        
        def audio_callback(audio_chunk, sample_rate):
            captured_audio.append(audio_chunk)
            if len(captured_audio) >= 6:  # ~3 seconds with 0.5s chunks
                capture_event.set()
        
        audio_capture = StreamingAudioCapture(
            callback=audio_callback,
            sample_rate=16000,
            chunk_duration=0.5
        )
        
        print("Starting audio capture...")
        audio_capture.start()
        
        print("Recording for 3 seconds...")
        capture_event.wait(timeout=5)
        
        audio_capture.stop()
        
        if captured_audio:
            # Combine captured chunks
            combined_audio = np.concatenate(captured_audio)
            
            # Save to file
            output_file = "data/samples/capture_test_output.wav"
            with wave.open(output_file, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(16000)
                wf.writeframes((combined_audio * 32767).astype(np.int16).tobytes())
            
            print(f"Captured audio saved to {output_file}")
            print("Audio capture test completed")
            
        print("Audio components test successful!")
        return True
        
    except Exception as e:
        print(f"Audio components test failed: {e}")
        return False

def test_full_pipeline(device=None):
    """Test the full translation pipeline with synthetic audio."""
    print("\n--- Testing Full Pipeline ---")
    
    try:
        # Initialize components
        asr = StreamingASR(model_name="openai/whisper-small", device=device)
        translator = StreamingTranslator(
            source_lang="en", 
            target_lang="es", 
            device=device
        )
        tts = StreamingTTS(device=device)
        
        # Create audio components
        audio_playback = StreamingAudioPlayback()
        
        # Pipeline output text
        output_text = {"source": "", "translated": ""}
        
        # Update callback
        def update_callback(source_text, translated_text):
            output_text["source"] = source_text
            output_text["translated"] = translated_text
            print(f"\nSource: {source_text}")
            print(f"Translation: {translated_text}")
        
        # Create pipeline
        audio_capture = StreamingAudioCapture(callback=lambda x, y: None)
        
        pipeline = RealTimeTranslationPipeline(
            asr_model=asr,
            mt_model=translator,
            tts_model=tts,
            audio_capture=audio_capture,
            audio_playback=audio_playback
        )
        
        # Set callback
        pipeline.set_update_callback(update_callback)
        
        # Start pipeline components
        print("Starting pipeline components...")
        audio_playback.start()
        
        # Process a sample audio file or generate test audio
        try:
            # Try to load a sample WAV file if available
            sample_file = "data/samples/sample_en.wav"
            if os.path.exists(sample_file):
                print(f"Loading sample audio from {sample_file}")
                import wave
                with wave.open(sample_file, 'rb') as wf:
                    sample_rate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            else:
                # Generate synthetic speech-like audio
                print("Generating synthetic audio")
                sample_rate = 16000
                duration = 5  # seconds
                t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
                
                # Generate a more complex signal (mixture of frequencies)
                audio_data = 0.3 * np.sin(2 * np.pi * 200 * t)  # Base frequency
                audio_data += 0.2 * np.sin(2 * np.pi * 400 * t)  # Overtone
                audio_data += 0.1 * np.sin(2 * np.pi * 600 * t)  # Another overtone
                
                # Add some amplitude modulation to simulate speech
                am = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
                audio_data = audio_data * am
        except Exception as e:
            print(f"Error loading/generating audio: {e}")
            # Fallback to simple sine wave
            sample_rate = 16000
            duration = 3  # seconds
            t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
            audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)
        
        # Process the audio directly
        print("Processing audio through pipeline...")
        
        # Process in chunks to simulate streaming
        chunk_size = int(sample_rate * 0.5)  # 500ms chunks
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            if len(chunk) == chunk_size:  # Skip partial chunks
                # Process the chunk through the pipeline
                start_time = time.time()
                
                # ASR
                asr_result = asr.transcribe_chunk(chunk, sample_rate)
                
                # If new text, translate
                if asr_result['text']:
                    # MT
                    mt_result = translator.translate_chunk(asr_result['text'])
                    
                    # If new translation, synthesize
                    if mt_result['text']:
                        # TTS
                        audio = tts.synthesize_speech(mt_result['text'])
                        
                        # Play audio
                        if audio is not None:
                            audio_playback.play(audio)
                    
                    # Update output text
                    output_text["source"] = asr_result['full_text']
                    output_text["translated"] = mt_result['full_text']
                    
                    # Print update
                    print(f"\nSource: {output_text['source']}")
                    print(f"Translation: {output_text['translated']}")
                
                end_time = time.time()
                print(f"Chunk {i//chunk_size + 1} processed in {end_time - start_time:.2f} seconds")
                
                # Add small delay to allow audio to play
                time.sleep(0.2)
        
        # Wait for any remaining audio to play
        time.sleep(2)
        
        # Stop components
        print("Stopping pipeline components...")
        audio_playback.stop()
        
        print("Pipeline test completed")
        return True
        
    except Exception as e:
        print(f"Pipeline test failed: {e}")
        return False

def run_tests():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Run tests for real-time translation system")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")
    parser.add_argument("--component", choices=["asr", "mt", "tts", "audio", "pipeline", "all"], 
                       default="all", help="Component to test")
    
    args = parser.parse_args()
    
    # Set device
    if args.no_gpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Running tests on device: {device}")
    
    # Create output directories
    os.makedirs("data/samples", exist_ok=True)
    
    # Track test results
    results = {}
    
    # Run selected tests
    if args.component == "all" or args.component == "asr":
        results["asr"] = test_asr_component(device)
    
    if args.component == "all" or args.component == "mt":
        results["mt"] = test_mt_component(device)
    
    if args.component == "all" or args.component == "tts":
        results["tts"] = test_tts_component(device)
    
    if args.component == "all" or args.component == "audio":
        results["audio"] = test_audio_components()
    
    if args.component == "all" or args.component == "pipeline":
        results["pipeline"] = test_full_pipeline(device)
    
    # Print summary
    print("\n--- Test Results Summary ---")
    all_passed = True
    for component, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{component.upper()}: {status}")
        all_passed = all_passed and passed
    
    print(f"\nOverall: {'PASSED' if all_passed else 'FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    run_tests()