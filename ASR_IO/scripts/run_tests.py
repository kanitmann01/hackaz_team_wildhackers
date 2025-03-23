import numpy as np

def test_transcription(self, test_phrase="testing one two three"):
    """Test the transcription functionality with a synthesized audio."""
    print(f"Testing ASR with phrase: '{test_phrase}'")
    
    # Generate a test waveform (simple sine wave)
    sample_rate = 16000
    duration = 3  # seconds
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Process the test audio
    result = self.transcribe_chunk(audio)
    
    print(f"Test result: '{result['full_text']}'")
    return result['full_text'] != ""