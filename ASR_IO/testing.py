import os
import sys
import numpy as np
import torch
import wave
import time
import sounddevice as sd
import traceback

# Add parent directory to path to allow for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Create a very simple audio test function that doesn't depend on Whisper
def test_audio_recording():
    """Test basic audio recording functionality."""
    print("\n==== TESTING BASIC AUDIO RECORDING ====")
    try:
        # List available devices
        print("Available audio devices:")
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            print(f"  [{i}] {device['name']} (inputs: {device['max_input_channels']}, outputs: {device['max_output_channels']})")
        
        # Get default device info
        default_device = sd.query_devices(kind='input')
        print(f"\nDefault input device: {default_device['name']}")
        
        # Record a short audio clip
        duration = 3  # seconds
        sample_rate = 16000
        print(f"\nRecording {duration} seconds of audio...")
        print("Please speak into your microphone...")
        
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        
        # Wait until recording is finished
        sd.wait()
        
        # Flatten array
        audio_data = audio_data.flatten()
        
        # Check audio data
        print(f"Recorded {len(audio_data)} samples")
        print(f"Audio shape: {audio_data.shape}")
        print(f"Audio dtype: {audio_data.dtype}")
        print(f"Audio min: {np.min(audio_data):.6f}, max: {np.max(audio_data):.6f}, mean: {np.mean(np.abs(audio_data)):.6f}")
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Save audio to WAV file
        audio_file = os.path.join(output_dir, "test_recording.wav")
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(audio_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        print(f"Audio saved to {audio_file}")
        
        # Play it back to confirm
        print("Playing back the recorded audio...")
        sd.play(audio_data, sample_rate)
        sd.wait()
        
        # Check for silence or very low audio
        audio_level = np.mean(np.abs(audio_data))
        if audio_level < 0.01:
            print("WARNING: Audio level is very low. Check your microphone.")
            return False, audio_file
        
        return True, audio_file
    
    except Exception as e:
        print(f"ERROR in audio recording: {e}")
        traceback.print_exc()
        return False, None

# Test function that generates a sine wave instead of using a microphone
def generate_test_audio():
    """Generate a test audio signal (sine wave)."""
    print("\n==== GENERATING TEST AUDIO ====")
    
    # Parameters
    sample_rate = 16000
    duration = 2  # seconds
    
    # Generate time array
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    
    # Generate sine wave (440 Hz = A4 note)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save audio to WAV file
    audio_file = os.path.join(output_dir, "test_sine.wav")
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with wave.open(audio_file, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes for int16
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    
    print(f"Test sine wave generated and saved to {audio_file}")
    
    # Play it back
    print("Playing back the sine wave...")
    sd.play(audio, sample_rate)
    sd.wait()
    
    return audio, sample_rate, audio_file

# Function to load an audio file
def load_audio_file(file_path):
    """Load audio from a WAV file."""
    print(f"\n==== LOADING AUDIO FILE: {file_path} ====")
    try:
        with wave.open(file_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            
            print(f"Audio file info:")
            print(f"  Sample rate: {sample_rate} Hz")
            print(f"  Channels: {channels}")
            print(f"  Sample width: {sample_width} bytes")
            print(f"  Number of frames: {n_frames}")
            print(f"  Duration: {n_frames / sample_rate:.2f} seconds")
            
            # Read all frames
            frames = wf.readframes(n_frames)
            
            # Convert to numpy array
            if sample_width == 2:  # 16-bit
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            elif sample_width == 4:  # 32-bit
                audio_data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483647.0
            else:  # Assume 8-bit
                audio_data = np.frombuffer(frames, dtype=np.uint8).astype(np.float32) / 255.0 - 0.5
            
            # Convert stereo to mono if needed
            if channels == 2:
                audio_data = audio_data.reshape(-1, 2).mean(axis=1)
            
            print(f"Loaded audio: {len(audio_data)} samples")
            print(f"Audio shape: {audio_data.shape}")
            print(f"Audio min: {np.min(audio_data):.6f}, max: {np.max(audio_data):.6f}, mean: {np.mean(np.abs(audio_data)):.6f}")
            
            return audio_data, sample_rate
    
    except Exception as e:
        print(f"ERROR loading audio file: {e}")
        traceback.print_exc()
        return None, None

# Now test with WhisperASR in verbose mode
def test_whisper_verbose(language="en", audio_file=None):
    """
    Test Whisper ASR with detailed debug output.
    
    Args:
        language: Language code
        audio_file: Path to audio file (optional)
    """
    print(f"\n==== TESTING WHISPER ASR WITH LANGUAGE: {language} ====")
    
    # Import Whisper separately to catch import errors
    try:
        print("Importing Whisper ASR class...")
        from src.asr.whisper_asr import WhisperASR
        print("Import successful!")
    except ImportError as e:
        print(f"ERROR importing WhisperASR: {e}")
        traceback.print_exc()
        return False
    
    try:
        # Check if whisper module is accessible directly
        try:
            import whisper
            print(f"Whisper module directly accessible: version info = {whisper.__version__ if hasattr(whisper, '__version__') else 'unknown'}")
        except ImportError:
            print("Whisper module not directly accessible - may be installed by WhisperASR class")
        
        # Check device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        if device.type == "cuda":
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        
        # Print PyTorch version
        print(f"PyTorch version: {torch.__version__}")
        
        # Initialize Whisper with specific parameters
        print(f"Initializing WhisperASR with language: {language}...")
        print(f"Model name: tiny (using smallest for faster testing)")
        
        # Create model with detailed reporting
        asr = WhisperASR(model_name="tiny", device=device, language=language)
        print("WhisperASR initialized successfully!")
        
        # Check details of ASR object
        print("\nWhisperASR object details:")
        print(f"  Language setting: {asr.language}")
        print(f"  Model loaded: {'Yes' if hasattr(asr, 'model') and asr.model is not None else 'No'}")
        print(f"  Model type: {type(asr.model).__name__ if hasattr(asr, 'model') and asr.model is not None else 'None'}")
        
        # Source audio
        if audio_file and os.path.exists(audio_file):
            # Load audio file
            audio_data, sample_rate = load_audio_file(audio_file)
            
            if audio_data is None:
                print("Failed to load audio file")
                return False
        else:
            # Generate test audio
            print("No audio file provided, generating test sine wave...")
            audio_data, sample_rate, _ = generate_test_audio()
        
        # Make sure audio is formatted correctly
        if len(audio_data.shape) > 1:
            print(f"WARNING: Audio has multiple channels: {audio_data.shape}")
            print("Converting to mono by taking mean across channels")
            audio_data = audio_data.mean(axis=1)
        
        # Test with explicit parameters
        print("\nTranscribing audio...")
        
        # Reset ASR context
        asr.reset_context()
        
        # Time the transcription
        start_time = time.time()
        
        # Transcribe
        result = asr.transcribe_chunk(audio_data, sample_rate)
        
        # Calculate time
        transcription_time = time.time() - start_time
        
        # Process results
        if result:
            print(f"Transcription successful! Time: {transcription_time:.2f} seconds")
            print(f"Full text: {result['full_text']}")
            print(f"Text: {result['text']}")
            
            # Check if text is empty
            if not result['full_text'] and not result['text']:
                print("WARNING: Transcription returned empty text")
                
                # Try running the built-in test
                print("\nTrying built-in test function...")
                test_result = asr.test_transcription()
                print(f"Built-in test result: {'Success' if test_result else 'Failed'}")
        else:
            print("Transcription returned None")
            return False
        
        return True
    
    except Exception as e:
        print(f"ERROR in WhisperASR test: {e}")
        traceback.print_exc()
        return False

def main():
    """Run a series of diagnostic tests."""
    print("\n==== WHISPER ASR DIAGNOSTICS ====")
    print("This script will run diagnostics to identify issues with audio transcription")
    
    # Test basic audio recording
    audio_success, audio_file = test_audio_recording()
    
    # Test WhisperASR
    if audio_success:
        print("\nAudio recording successful. Testing with Whisper ASR...")
        whisper_success = test_whisper_verbose("en", audio_file)
    else:
        print("\nSkipping real audio test due to recording failure.")
        print("Testing Whisper with generated sine wave...")
        whisper_success = test_whisper_verbose("en")
    
    # Print summary
    print("\n==== DIAGNOSTIC SUMMARY ====")
    print(f"Audio recording: {'SUCCESS' if audio_success else 'FAILED'}")
    print(f"Whisper ASR: {'SUCCESS' if whisper_success else 'FAILED'}")
    
    if not audio_success:
        print("\nAudio recording issues detected:")
        print("1. Check if your microphone is properly connected")
        print("2. Check if your microphone is selected as the default recording device")
        print("3. Check if your microphone is muted or volume is too low")
        print("4. Try restarting your computer to reset audio drivers")
    
    if not whisper_success:
        print("\nWhisper ASR issues detected:")
        print("1. Make sure all dependencies are properly installed:")
        print("   - Run: pip install openai-whisper")
        print("2. Check for CUDA/GPU compatibility issues if using GPU")
        print("3. Check that the 'tiny' model is being downloaded correctly")
        
    print("\nFor further investigation, review the detailed output above.")

if __name__ == "__main__":
    main()