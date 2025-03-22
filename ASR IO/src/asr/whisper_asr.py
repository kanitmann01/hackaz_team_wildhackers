import torch
import numpy as np
import time
import os
import sys

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
        
        # Check if audio is too quiet (basic voice activity detection)
        audio_energy = np.mean(np.abs(audio_chunk))
        if audio_energy < 0.005:  # Threshold can be adjusted
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
            'avg_processing_time': self.total_processing_time / self.chunk_count if self.chunk_count > 0 else 0
        }