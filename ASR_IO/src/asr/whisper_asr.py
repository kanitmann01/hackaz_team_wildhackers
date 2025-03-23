import torch
import numpy as np
import time
import os
import sys
import wave

# Disable progress bars to avoid console errors
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["TQDM_DISABLE"] = "1"

# Try to load Whisper - if not installed, print instructions
try:
    import whisper
except ImportError:
    print("Whisper not found. Installing with:")
    print("pip install openai-whisper")
    # Try to install it automatically
    os.system("pip install openai-whisper")
    try:
        import whisper
    except ImportError:
        print("Failed to import whisper after installation. Please install manually.")

class WhisperASR:
    """
    Real-time streaming Automatic Speech Recognition using OpenAI Whisper.
    """
    
    def __init__(self, model_name="tiny", device=None, language=None):
        """
        Initialize the Whisper ASR system.
        
        Args:
            model_name: Whisper model size (tiny, base, small, medium, large)
            device: Computation device ('cuda' or 'cpu')
            language: Source language code (optional)
        """
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Store language for transcription
        self.language = language
        
        # Load model - use a smaller model initially for faster loading
        print(f"Loading Whisper model '{model_name}' on {self.device}...")
        self.model_loaded = False
        try:
            # Disable stdout temporarily to avoid progress bar issues
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            # Load the model
            self.model = whisper.load_model(model_name, device=self.device)
            
            # Restore stdout
            sys.stdout.close()
            sys.stdout = original_stdout
            
            print("Whisper model loaded successfully!")
            self.model_loaded = True
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            print("Falling back to dummy ASR implementation")
            self.model = None
        
        # Context tracking
        self.context_text = ""
        self.current_text = ""
        
        # Performance tracking
        self.total_processing_time = 0
        self.chunk_count = 0
        
        # VAD (Voice Activity Detection) parameters - reduced threshold for better sensitivity
        self.vad_threshold = 0.003  # Lower threshold for detecting speech
        self.min_active_ratio = 0.05  # Lower ratio for detecting speech
    
    def set_language(self, language_code):
        """Set the source language for ASR."""
        self.language = language_code
        print(f"ASR language set to: {language_code}")
    
    def reset_context(self):
        """Reset the transcription context."""
        self.context_text = ""
        self.current_text = ""
        self.total_processing_time = 0
        self.chunk_count = 0
    
    def test_transcription(self, test_phrase="testing one two three"):
        """Test the transcription functionality with sample audio."""
        if not self.model_loaded:
            print("Cannot test transcription: Model not loaded")
            return False
        
        print(f"Testing ASR with test audio...")
        
        # Generate a test waveform (sine wave with varying frequency to simulate speech)
        sample_rate = 16000
        duration = 3  # seconds
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        
        # Generate a more complex signal to simulate speech
        audio = 0.3 * np.sin(2 * np.pi * 200 * t)  # Base frequency
        audio += 0.2 * np.sin(2 * np.pi * 400 * t)  # Overtone
        audio += 0.1 * np.sin(2 * np.pi * 600 * t)  # Another overtone
        
        # Add amplitude modulation to make it more speech-like
        am = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
        audio = audio * am
        
        # Try loading a sample audio file if available
        sample_found = False
        try:
            sample_file = "data/samples/test_en.wav"
            if os.path.exists(sample_file):
                print(f"Found test audio file: {sample_file}")
                with wave.open(sample_file, 'rb') as wf:
                    sample_rate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                    sample_found = True
        except Exception as e:
            print(f"Error loading test audio file: {e}")
        
        # Process the test audio
        start_time = time.time()
        result = self.transcribe_chunk(audio, sample_rate)
        end_time = time.time()
        
        print(f"Test result: '{result['full_text']}'")
        print(f"Processing took {end_time - start_time:.2f}s")
        
        # Create test sample directory if not exists
        if not sample_found:
            try:
                os.makedirs("data/samples", exist_ok=True)
                # Save the test audio for future tests
                audio_int16 = (audio * 32767).astype(np.int16)
                with wave.open("data/samples/test_en.wav", 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)  # 2 bytes for int16
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_int16.tobytes())
                print("Saved test audio to data/samples/test_en.wav")
            except Exception as e:
                print(f"Error saving test audio: {e}")
        
        # Return success if any text was detected
        return len(result['full_text']) > 0
    
    def transcribe_chunk(self, audio_chunk, sample_rate=16000):
        """
        Transcribe an audio chunk.
        
        Args:
            audio_chunk: NumPy array of audio samples
            sample_rate: Sample rate of the audio
            
        Returns:
            dict: Result with text and timing information
        """
        start_time = time.time()
        
        # If model failed to load, return dummy result
        if self.model is None:
            dummy_text = "Whisper model failed to load. Please check installation."
            return {
                'text': dummy_text,
                'full_text': dummy_text,
                'stable_text': dummy_text,
                'processing_time': 0,
                'avg_processing_time': 0
            }
        
        # Enhanced audio level diagnostics
        audio_max = np.max(np.abs(audio_chunk))
        audio_mean = np.mean(np.abs(audio_chunk))
        audio_energy = audio_mean
        
        print(f"Audio diagnostics: max={audio_max:.4f}, mean={audio_mean:.4f}, energy={audio_energy:.6f}")
        
        # Check if audio is too quiet (basic voice activity detection) - using lower threshold
        if audio_energy < self.vad_threshold:
            print(f"Audio energy too low: {audio_energy:.6f}, skipping processing")
            return {
                'text': '',
                'full_text': self.context_text,
                'stable_text': self.context_text,
                'processing_time': 0,
                'avg_processing_time': self.total_processing_time / max(1, self.chunk_count)
            }
        
        # Resample if needed (Whisper requires 16kHz)
        if sample_rate != 16000:
            resampling_factor = 16000 / sample_rate
            resampled_length = int(len(audio_chunk) * resampling_factor)
            indices = np.linspace(0, len(audio_chunk) - 1, resampled_length)
            audio_chunk = np.interp(indices, np.arange(len(audio_chunk)), audio_chunk)
        
        # Ensure audio is normalized
        if np.max(np.abs(audio_chunk)) > 0:
            audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
        
        # Run Whisper transcription
        options = {
            "fp16": False  # Avoid fp16 issues on some systems
        }
        
        if self.language:
            options["language"] = self.language
        
        try:
            print(f"Processing audio chunk: length={len(audio_chunk)}, max={np.max(audio_chunk):.3f}")
            result = self.model.transcribe(
                audio_chunk, 
                **options
            )
            
            transcription = result["text"].strip()
            print(f"Raw transcription: {transcription}")
            
            # Determine new text
            if not self.current_text:
                new_text = transcription
            elif transcription.startswith(self.current_text):
                new_text = transcription[len(self.current_text):].strip()
            else:
                new_text = transcription
            
            # Update state
            self.current_text = transcription
            if new_text:
                self.context_text += " " + new_text if self.context_text else new_text
                self.context_text = self.context_text.strip()
                print(f"Updated context: {self.context_text}")
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.chunk_count += 1
            avg_processing_time = self.total_processing_time / self.chunk_count if self.chunk_count > 0 else 0
            
            print(f"Transcription completed in {processing_time:.2f}s")
            
            return {
                'text': new_text.strip(),
                'full_text': self.context_text.strip(),
                'stable_text': self.context_text.strip(),
                'processing_time': processing_time,
                'avg_processing_time': avg_processing_time
            }
            
        except Exception as e:
            print(f"Error in transcription: {e}")
            import traceback
            print(traceback.format_exc())
            return {
                'text': '',
                'full_text': self.context_text.strip(),
                'stable_text': self.context_text.strip(),
                'processing_time': time.time() - start_time,
                'avg_processing_time': self.total_processing_time / max(1, self.chunk_count)
            }
    
    def get_stats(self):
        """Get performance statistics."""
        return {
            'total_chunks': self.chunk_count,
            'total_processing_time': self.total_processing_time,
            'avg_processing_time': self.total_processing_time / self.chunk_count if self.chunk_count > 0 else 0,
            'model_loaded': self.model_loaded
        }