import torch
import numpy as np
import time
import re
from transformers import AutoProcessor, AutoModel

class StreamingTTS:
    """
    Real-time streaming Text-to-Speech synthesis using VITS.
    
    This class implements incremental speech synthesis, generating
    audio for text fragments as they arrive from the translation system
    and allowing for immediate playback.
    """
    
    def __init__(self, device=None, model_name="facebook/mms-tts-eng"):
        """
        Initialize the streaming TTS component with VITS model.
        
        Args:
            device: Computation device ('cuda' or 'cpu')
            model_name: TTS model to use (default: MMS-TTS English model)
        """
        # Set device (use CUDA if available unless specified otherwise)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load VITS model
        print(f"Loading VITS TTS model '{model_name}' on {self.device}...")
        
        try:
            # Load MMS-TTS model
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            
            # Configure for better streaming performance
            # Half-precision for faster processing on GPU
            if self.device.type == 'cuda':
                self.model = self.model.half()
                
            print("VITS TTS model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading VITS TTS model: {e}")
            print("Falling back to dummy TTS (will generate silence)")
            self.model = None
            self.processor = None
        
        # Current language code
        self.language_code = "eng"
        
        # Punctuation regex for chunk splitting
        self.punct_regex = re.compile(r'([.!?;:])')
        
        # Performance tracking
        self.total_processing_time = 0
        self.chunk_count = 0
        self.sample_rate = 16000
    
    def set_language(self, language_code):
        """
        Set the target language for TTS.
        
        Args:
            language_code: Language code in ISO or MMS-TTS format
        """
        # MMS language code mapping (ISO to internal codes)
        iso_to_mms = {
            "en": "eng",
            "es": "spa",
            "fr": "fra",
            "de": "deu",
            "it": "ita",
            "pt": "por",
            "ru": "rus",
            "zh": "cmn",
            "ja": "jpn",
            "ko": "kor",
            "ar": "ara",
            "hi": "hin"
        }
        
        # Convert language code if needed
        language = iso_to_mms.get(language_code, language_code)
        
        # Update the model if language changed
        if language != self.language_code:
            self.language_code = language
            try:
                model_name = f"facebook/mms-tts-{language}"
                print(f"Loading VITS TTS model for {language}...")
                self.processor = AutoProcessor.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name).to(self.device)
                
                # Configure for better streaming performance
                if self.device.type == 'cuda':
                    self.model = self.model.half()
                    
                print(f"VITS TTS model for {language} loaded successfully!")
            except Exception as e:
                print(f"Error loading VITS TTS model for {language}: {e}")
                print(f"Keeping current model ({self.language_code})")
    
    def _split_into_sentences(self, text):
        """
        Split text into sentence-like chunks for better synthesis.
        
        Args:
            text: Input text
            
        Returns:
            list: List of text chunks
        """
        # Add space after punctuation if not present
        text = self.punct_regex.sub(r'\1 ', text)
        
        # Split by punctuation
        chunks = []
        current_chunk = ""
        
        for word in text.split():
            current_chunk += " " + word if current_chunk else word
            
            # If word ends with punctuation, end the chunk
            if any(word.endswith(p) for p in ['.', '!', '?', ';', ':']):
                chunks.append(current_chunk.strip())
                current_chunk = ""
        
        # Add remaining text as a chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def synthesize_speech(self, text, play_callback=None):
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            play_callback: Function to call with synthesized audio
            
        Returns:
            numpy.ndarray: Audio data
        """
        start_time = time.time()
        
        # Skip empty text
        if not text or not text.strip():
            return None
        
        # Use dummy TTS if model failed to load
        if self.model is None or self.processor is None:
            # Generate silence with proper duration (approx. 5 samples per character)
            duration = len(text) * 5
            audio = np.zeros(duration, dtype=np.float32)
            
            if play_callback:
                play_callback(audio)
                
            return audio
        
        # Split text into sentence-like chunks for better synthesis
        chunks = self._split_into_sentences(text)
        
        all_audio = []
        
        for chunk in chunks:
            try:
                # Process text with VITS
                inputs = self.processor(
                    text=chunk,
                    return_tensors="pt"
                ).to(self.device)
                
                # Generate speech
                with torch.no_grad():
                    output = self.model(
                        **inputs,
                        language=self.language_code
                    )
                
                # Extract audio data
                speech = output.waveform.cpu().numpy().squeeze()
                
                # Normalize audio
                if np.abs(speech).max() > 0:
                    speech = speech / np.abs(speech).max() * 0.9
                
                # Add to combined output
                all_audio.append(speech)
                
                # If callback provided, play this chunk immediately
                if play_callback:
                    play_callback(speech)
                    
            except Exception as e:
                print(f"Error synthesizing speech for chunk '{chunk}': {e}")
        
        # Combine all audio chunks
        combined_audio = np.concatenate(all_audio) if all_audio else np.array([], dtype=np.float32)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        self.chunk_count += 1
        
        return combined_audio
    
    def get_stats(self):
        """
        Get performance statistics.
        
        Returns:
            dict: Performance statistics
        """
        return {
            'total_chunks': self.chunk_count,
            'total_processing_time': self.total_processing_time,
            'avg_processing_time': self.total_processing_time / self.chunk_count if self.chunk_count > 0 else 0
        }
