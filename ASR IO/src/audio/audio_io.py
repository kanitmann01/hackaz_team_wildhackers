import sounddevice as sd
import numpy as np
import wave
import os

class AudioIO:
    def __init__(self, sample_rate=16000, channels=1):
        """
        Initialize the AudioIO class for audio recording and playback.
        
        Args:
            sample_rate (int): The sampling rate to use for audio (default: 16000)
            channels (int): Number of audio channels (default: 1, mono)
        """
        self.sample_rate = sample_rate
        self.channels = channels
    
    def record_audio(self, duration=5):
        """
        Record audio for a specified duration.
        
        Args:
            duration (float): Recording duration in seconds
            
        Returns:
            numpy.ndarray: Recorded audio as numpy array
        """
        print(f"Recording for {duration} seconds...")
        audio_data = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        print("Recording complete")
        return audio_data.flatten()
    
    def play_audio(self, audio_data):
        """
        Play audio from numpy array.
        
        Args:
            audio_data (numpy.ndarray): Audio data to play
        """
        print("Playing audio...")
        sd.play(audio_data, self.sample_rate)
        sd.wait()  # Wait until audio is done playing
    
    def save_audio(self, audio_data, filename="recorded_audio.wav"):
        """
        Save audio data to WAV file.
        
        Args:
            audio_data (numpy.ndarray): Audio data to save
            filename (str): Output filename
        """
        # Ensure data is in the right format
        audio_data = (audio_data * 32767).astype(np.int16)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Write WAV file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data.tobytes())
        
        print(f"Audio saved to {filename}")
    
    def load_audio(self, filename):
        """
        Load audio from WAV file.
        
        Args:
            filename (str): Input WAV filename
            
        Returns:
            tuple: (audio_data, sample_rate)
        """
        with wave.open(filename, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
            sample_rate = wf.getframerate()
        
        return audio_data, sample_rate