from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

class WhisperASR:
    def __init__(self, model_name="openai/whisper-small"):
        """
        Initialize the Whisper ASR model.
        
        Args:
            model_name (str): The name of the Whisper model to use.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper model on {self.device}...")
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name).to(self.device)
        print("Whisper model loaded successfully!")
    
    def transcribe_audio(self, audio_array, sampling_rate=16000):
        """
        Transcribe audio to text.
        
        Args:
            audio_array (numpy.ndarray): The audio waveform to transcribe.
            sampling_rate (int): The sampling rate of the audio (default: 16000).
            
        Returns:
            str: The transcribed text.
        """
        # Process audio input
        input_features = self.processor(
            audio_array, 
            sampling_rate=sampling_rate, 
            language="en",
            return_tensors="pt"
        ).input_features.to(self.device)
        
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription
