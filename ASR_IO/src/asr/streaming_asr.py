import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
import time

class StreamingASR:
    """
    Real-time streaming Automatic Speech Recognition using Whisper.
    
    This class implements the Whisper model for real-time speech
    recognition, processing audio in small chunks while maintaining context
    for better transcription continuity.
    """
    
    def __init__(self, model_name="openai/whisper-tiny", device=None, 
                 use_stable_chunks=True, language="en"):
        """
        Initialize the streaming ASR system.
        
        Args:
            model_name: Whisper model name
            device: Computation device ('cuda' or 'cpu')
            use_stable_chunks: Whether to use the stable chunk approach
            language: Source language code (default: "en" for English)
        """
        # Set device (use CUDA if available unless specified otherwise)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Map language code to Whisper language
        self.language = language
        
        # Load model and processor
        print(f"Loading Whisper model '{model_name}' on {self.device}...")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        # Configure for better streaming performance
        if self.device.type == 'cuda':
            self.model = self.model.half()  # Half-precision for faster processing
        print("Whisper model loaded successfully!")
        
        # Context tracking
        self.use_stable_chunks = use_stable_chunks
        self.context_text = ""
        self.current_text = ""
        self.last_chunk_text = ""
        self.chunk_history = []  # Store recent chunk transcriptions
        self.stable_window_size = 3  # Number of chunks to consider for stability
        
        # Streaming buffer for audio context
        self.audio_context = np.array([], dtype=np.float32)
        self.context_duration = 3.0  # seconds of audio context to keep (increased from 1.0)
        self.context_samples = int(16000 * self.context_duration)
        
        # Performance tracking
        self.total_processing_time = 0
        self.chunk_count = 0
    
    def set_language(self, language_code):
        """
        Set the source language for ASR.
        
        Args:
            language_code: ISO language code (e.g., "en", "es", "fr")
        """
        self.language = language_code
    
    def reset_context(self):
        """Reset the transcription context history."""
        self.context_text = ""
        self.current_text = ""
        self.last_chunk_text = ""
        self.chunk_history = []
        self.audio_context = np.array([], dtype=np.float32)
        self.total_processing_time = 0
        self.chunk_count = 0
    
    def _preprocess_audio(self, audio_chunk):
        """
        Preprocess audio for Whisper model.
        
        Args:
            audio_chunk: NumPy array of audio samples
            
        Returns:
            torch.Tensor: Processed input features
        """
        # Ensure float32 dtype
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        # Normalize audio to [-1, 1]
        if np.abs(audio_chunk).max() > 0:
            audio_chunk = audio_chunk / np.abs(audio_chunk).max()
        
        # Update audio context buffer
        self.audio_context = np.append(self.audio_context, audio_chunk)
        
        # Keep only the most recent context
        if len(self.audio_context) > self.context_samples:
            self.audio_context = self.audio_context[-self.context_samples:]
            
        # Use the full context for processing
        context_chunk = self.audio_context.copy()
        
        # Process audio with the model's processor
        inputs = self.processor(
            context_chunk, 
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        return inputs.input_features.to(self.device), inputs.attention_mask.to(self.device) if hasattr(inputs, 'attention_mask') else None
    
    def _decode_prediction(self, prediction):
        """
        Decode Whisper prediction to text.
        
        Args:
            prediction: Model output prediction
            
        Returns:
            str: Decoded text
        """
        # Decode prediction to text
        transcription = self.processor.batch_decode(prediction, skip_special_tokens=True)[0]
        
        return transcription
    
    def _determine_new_text(self, transcription):
        """
        Determine the new text from a transcription by comparing with context.
        
        Args:
            transcription: Full transcription from Whisper
            
        Returns:
            str: New text (increment from previous context)
        """
        # If no previous context, return full transcription
        if not self.current_text:
            new_text = transcription
        # If transcription starts with current text, extract the new part
        elif transcription.startswith(self.current_text):
            new_text = transcription[len(self.current_text):].strip()
        # If we can't cleanly extract, return the full transcription
        else:
            new_text = transcription
        
        # Store in history for stability analysis
        self.chunk_history.append(new_text)
        if len(self.chunk_history) > self.stable_window_size:
            self.chunk_history.pop(0)
        
        return new_text
    
    def transcribe_chunk(self, audio_chunk, sample_rate=16000):
        """
        Transcribe an audio chunk in the context of previous chunks.
        
        Args:
            audio_chunk: NumPy array of audio samples
            sample_rate: Sample rate of the audio
            
        Returns:
            dict: Containing 'text' (new text), 'full_text' (all text so far),
                  'stable_text' (confirmed text) and 'processing_time'
        """
        start_time = time.time()
        
        # Check audio level - skip processing if below threshold (voice activity detection)
        audio_level = np.mean(np.abs(audio_chunk))
        if audio_level < 0.01:  # Adjust threshold based on testing
            return {
                'text': '',
                'full_text': self.context_text.strip(),
                'stable_text': self.context_text.strip(),
                'processing_time': 0,
                'avg_processing_time': self.total_processing_time / max(1, self.chunk_count)
            }
        
        # Resample if needed (Whisper requires 16kHz)
        if sample_rate != 16000:
            # Simple resampling by linear interpolation
            resampling_factor = 16000 / sample_rate
            resampled_length = int(len(audio_chunk) * resampling_factor)
            indices = np.linspace(0, len(audio_chunk) - 1, resampled_length)
            audio_chunk = np.interp(indices, np.arange(len(audio_chunk)), audio_chunk)
        
        # Process audio
        input_features, attention_mask = self._preprocess_audio(audio_chunk)
        
        # Generate transcription
        with torch.no_grad():
            # Set decoder input based on current language
            forced_decoder_ids = None
            if self.language != "en":
                # Map language code to Whisper language ID
                lang_id = self.processor.tokenizer.get_language_id(self.language)
                forced_decoder_ids = [(1, lang_id)]
            
            # Generate transcription
            predicted_ids = self.model.generate(
                input_features,
                attention_mask=attention_mask,
                forced_decoder_ids=forced_decoder_ids,
                max_length=448  # Maximum sequence length for Whisper
            )
            
        # Decode transcription
        transcription = self._decode_prediction(predicted_ids)
        
        print(f"Raw transcription: {transcription}")
        print(f"Audio input level: {np.max(np.abs(audio_chunk))}")
        
        # Identify new text
        new_text = self._determine_new_text(transcription)
        
        # Update state for next chunk
        self.current_text = transcription
        self.last_chunk_text = new_text
        
        # Update context with new text (if any)
        if new_text:
            self.context_text += " " + new_text if self.context_text else new_text
            self.context_text = self.context_text.strip()
        
        # Determine stable text (text that is unlikely to change)
        stable_text = self.context_text
        
        # Update performance metrics
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        self.chunk_count += 1
        avg_processing_time = self.total_processing_time / self.chunk_count if self.chunk_count > 0 else 0
        
        return {
            'text': new_text.strip(),
            'full_text': self.context_text.strip(),
            'stable_text': stable_text.strip(),
            'processing_time': processing_time,
            'avg_processing_time': avg_processing_time
        }
    
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