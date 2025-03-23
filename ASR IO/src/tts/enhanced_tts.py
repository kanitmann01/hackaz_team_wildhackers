import torch
import numpy as np
import time
import re
from transformers import AutoProcessor, AutoModel
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TTS')

class EnhancedStreamingTTS:
    """
    Improved real-time streaming Text-to-Speech synthesis using MMS-TTS.
    
    This class addresses compatibility issues with the translation pipeline
    and provides better error handling and diagnostics.
    """
    
    def __init__(self, device=None, model_name="facebook/mms-tts-eng"):
        """
        Initialize the streaming TTS component with MMS-TTS model.
        
        Args:
            device: Computation device ('cuda' or 'cpu')
            model_name: TTS model to use (default: MMS-TTS English model)
        """
        # Set device (use CUDA if available unless specified otherwise)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Store initial model name for reference
        self.initial_model_name = model_name
        
        # Current language code and model
        self.language_code = "eng"
        self.model_name = model_name
        self.processor = None
        self.model = None
        
        # Load the model
        self._load_model(model_name)
        
        # Punctuation regex for chunk splitting
        self.punct_regex = re.compile(r'([.!?;:])')
        
        # Performance tracking
        self.total_processing_time = 0
        self.chunk_count = 0
        self.sample_rate = 16000
        
        # Error tracking
        self.last_error = None
        self.error_count = 0
        
        # Debugging - save a sample for diagnostics
        os.makedirs("data/samples", exist_ok=True)
    
    def _load_model(self, model_name):
        """Load the TTS model with proper error handling."""
        logger.info(f"Loading TTS model: {model_name} on {self.device}")
        
        try:
            # Load MMS-TTS model
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            
            # Configure for better streaming performance
            if self.device.type == 'cuda':
                self.model = self.model.half()
                
            logger.info("TTS model loaded successfully!")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error loading TTS model: {e}")
            logger.warning("TTS will not work correctly")
            self.model = None
            self.processor = None
            return False
    
    def set_language(self, language_code):
        """
        Set the target language for TTS with improved language code handling.
        
        Args:
            language_code: Language code in ISO or MMS-TTS format
        
        Returns:
            bool: True if language changed successfully
        """
        # MMS language code mapping (ISO to internal codes)
        iso_to_mms = {
            "en": "eng", "eng_Latn": "eng",
            "es": "spa", "spa_Latn": "spa",
            "fr": "fra", "fra_Latn": "fra",
            "de": "deu", "deu_Latn": "deu",
            "it": "ita", "ita_Latn": "ita",
            "pt": "por", "por_Latn": "por",
            "ru": "rus", "rus_Cyrl": "rus",
            "zh": "cmn", "zho_Hans": "cmn",
            "ja": "jpn", "jpn_Jpan": "jpn",
            "ko": "kor", "kor_Hang": "kor",
            "ar": "ara", "ara_Arab": "ara",
            "hi": "hin", "hin_Deva": "hin"
        }
        
        # Convert language code if needed
        mms_code = iso_to_mms.get(language_code.lower(), "eng")
        target_model_name = f"facebook/mms-tts-{mms_code}"
        
        logger.info(f"Language request: {language_code} → MMS code: {mms_code}")
        logger.info(f"Target model: {target_model_name}")
        
        # Update the model if language changed
        if mms_code != self.language_code:
            self.language_code = mms_code
            
            # Don't reload if model matches desired language
            if self.model_name == target_model_name:
                logger.info("Already using the correct language model")
                return True
                
            # Save old model name
            old_model = self.model_name
            self.model_name = target_model_name
            
            # Try to load new model
            success = self._load_model(target_model_name)
            
            # Fallback to English if loading fails
            if not success:
                logger.warning(f"Failed to load model for {mms_code}, falling back to English")
                self.language_code = "eng"
                self.model_name = "facebook/mms-tts-eng"
                return self._load_model("facebook/mms-tts-eng")
            
            return success
        
        return True
    
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
        Synthesize speech from text with improved error handling.
        
        Args:
            text: Text to synthesize
            play_callback: Function to call with synthesized audio
            
        Returns:
            numpy.ndarray: Audio data
        """
        start_time = time.time()
        
        # Skip empty text
        if not text or not text.strip():
            return np.array([], dtype=np.float32)
        
        # Log what we're synthesizing
        logger.info(f"Synthesizing with {self.language_code}: {text[:50]}...")
        
        # Use dummy TTS if model failed to load
        if self.model is None or self.processor is None:
            # Generate silence with proper duration (approx. 5 samples per character)
            duration = len(text) * 5
            audio = np.zeros(duration, dtype=np.float32)
            logger.warning(f"Using dummy TTS (silence) because model failed to load")
            
            if play_callback:
                play_callback(audio)
                
            return audio
        
        # Split text into sentence-like chunks for better synthesis
        chunks = self._split_into_sentences(text)
        logger.info(f"Split into {len(chunks)} chunks for synthesis")
        
        all_audio = []
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            try:
                # Process text with MMS-TTS (model-specific approach)
                inputs = self.processor(
                    text=chunk,
                    return_tensors="pt"
                ).to(self.device)
                
                # Track what we're trying
                logger.info(f"Processing chunk {i+1}/{len(chunks)}: {chunk[:30]}...")
                
                # Try several approaches to generation
                speech = None
                error_msg = None
                
                # Approach 1: Standard method (works for newer models)
                try:
                    with torch.no_grad():
                        output = self.model(**inputs)
                    
                    if hasattr(output, 'waveform'):
                        speech = output.waveform.cpu().numpy().squeeze()
                        logger.info("Approach 1 successful")
                except Exception as e1:
                    error_msg = f"Approach 1 failed: {e1}"
                    logger.debug(error_msg)
                
                # Approach 2: With language parameter (for older models)
                if speech is None:
                    try:
                        with torch.no_grad():
                            output = self.model(**inputs, language=self.language_code)
                        
                        if hasattr(output, 'waveform'):
                            speech = output.waveform.cpu().numpy().squeeze()
                            logger.info("Approach 2 successful")
                    except Exception as e2:
                        error_msg = f"{error_msg}\nApproach 2 failed: {e2}"
                        logger.debug(f"Approach 2 failed: {e2}")
                
                # Approach 3: Using generate method
                if speech is None:
                    try:
                        with torch.no_grad():
                            output = self.model.generate(**inputs)
                        
                        if isinstance(output, torch.Tensor):
                            speech = output.cpu().numpy().squeeze()
                            logger.info("Approach 3 successful")
                    except Exception as e3:
                        error_msg = f"{error_msg}\nApproach 3 failed: {e3}"
                        logger.debug(f"Approach 3 failed: {e3}")
                
                # If all approaches failed, log and skip
                if speech is None:
                    self.error_count += 1
                    if self.error_count <= 3:  # Only log the first few errors
                        logger.error(f"All synthesis approaches failed: {error_msg}")
                    continue
                
                # Save first chunk for diagnostics
                if i == 0 and len(speech) > 0:
                    try:
                        # Save raw audio for inspection
                        import wave
                        filename = f"data/samples/tts_latest_{self.language_code}.wav"
                        audio_int16 = (speech * 32767).astype(np.int16)
                        
                        with wave.open(filename, 'wb') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(16000)
                            wf.writeframes(audio_int16.tobytes())
                        
                        logger.info(f"Saved diagnostic audio to {filename}")
                    except Exception as save_error:
                        logger.error(f"Error saving diagnostic audio: {save_error}")
                
                # Normalize audio
                if len(speech) > 0 and np.abs(speech).max() > 0:
                    speech = speech / np.abs(speech).max() * 0.9
                
                # Check audio quality
                if len(speech) > 0 and np.abs(speech).max() < 0.01:
                    logger.warning(f"Audio is too quiet: max={np.abs(speech).max():.6f}")
                    # Amplify quiet audio
                    speech = speech * 100  # Amplify by 100x
                
                # Add to combined output
                all_audio.append(speech)
                
                # If callback provided, play this chunk immediately
                if play_callback and len(speech) > 0:
                    logger.info(f"Playing audio chunk: {len(speech)} samples")
                    play_callback(speech)
                    
            except Exception as e:
                logger.error(f"Error synthesizing speech for chunk '{chunk}': {e}")
                self.last_error = str(e)
        
        # Combine all audio chunks
        if all_audio:
            combined_audio = np.concatenate(all_audio)
            logger.info(f"Generated {len(combined_audio)} samples of audio")
        else:
            logger.warning("No audio was generated!")
            combined_audio = np.array([], dtype=np.float32)
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        self.chunk_count += 1
        
        return combined_audio
    
    def get_stats(self):
        """Get performance statistics."""
        return {
            'total_chunks': self.chunk_count,
            'total_processing_time': self.total_processing_time,
            'avg_processing_time': self.total_processing_time / self.chunk_count if self.chunk_count > 0 else 0,
            'errors': self.error_count,
            'last_error': self.last_error,
            'language': self.language_code,
            'model': self.model_name
        }