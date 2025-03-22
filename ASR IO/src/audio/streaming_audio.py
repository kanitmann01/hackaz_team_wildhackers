import sounddevice as sd
import numpy as np
import threading
import queue
import time
import wave
import os

class StreamingAudioCapture:
    """
    Captures audio in real-time streaming chunks with overlap.
    
    This class provides continuous audio recording capability by capturing
    small chunks of audio (typically 300-500ms) with overlap between chunks
    to ensure no speech is lost at chunk boundaries.
    """
    
    def __init__(self, callback, sample_rate=16000, channels=1, 
                 chunk_duration=0.5, overlap=0.1, device=None):
        """
        Initialize streaming audio capture.
        
        Args:
            callback: Function to be called with each audio chunk
            sample_rate: Audio sample rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1)
            chunk_duration: Duration of each audio chunk in seconds (default: 0.5)
            overlap: Overlap between chunks as a fraction of chunk_duration (default: 0.1)
            device: Audio device to use (default: system default)
        """
        self.callback = callback
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = int(chunk_duration * sample_rate)
        self.overlap_size = int(overlap * self.chunk_size)
        self.device = device
        
        # Audio buffer to maintain overlap between chunks
        self.buffer = np.zeros(self.overlap_size, dtype=np.float32)
        
        # Processing queue and thread control
        self.audio_queue = queue.Queue()
        self.running = False
        self.stream = None
        self.process_thread = None
        
        # Audio level monitoring
        self.current_audio_level = 0
        self.peak_audio_level = 0
    
    def audio_callback(self, indata, frames, time_info, status):
        """
        Callback function called by sounddevice for each audio block.
        
        Args:
            indata: Recorded audio data
            frames: Number of frames
            time_info: Time information
            status: Status flags
        """
        if status:
            print(f"Audio capture status: {status}")
        
        # Convert to mono if needed and ensure correct data type
        if indata.ndim > 1 and indata.shape[1] > 1:
            current_data = indata[:, 0].astype(np.float32)
        else:
            current_data = indata.flatten().astype(np.float32)
        
        # Update audio level metrics
        self.current_audio_level = np.mean(np.abs(current_data))
        self.peak_audio_level = max(self.peak_audio_level, np.max(np.abs(current_data)))
        
        # Concatenate with previous overlap to form a complete chunk
        full_chunk = np.concatenate((self.buffer, current_data))
        
        # Queue the chunk for processing
        self.audio_queue.put(full_chunk.copy())
        
        # Update buffer with overlap for next chunk
        self.buffer = current_data[-self.overlap_size:].copy() if len(current_data) >= self.overlap_size else current_data.copy()
    
    def process_queue(self):
        """
        Process audio chunks from the queue and call the user callback.
        """
        while self.running:
            try:
                # Get chunk from queue with timeout to allow for clean shutdown
                chunk = self.audio_queue.get(timeout=0.5)
                
                # Normalize audio level for consistent processing
                if np.max(np.abs(chunk)) > 0.0:
                    normalized_chunk = chunk / np.max(np.abs(chunk)) * 0.9
                else:
                    normalized_chunk = chunk
                
                # Call user callback with the audio chunk
                self.callback(normalized_chunk, self.sample_rate)
                
                # Mark task as done
                self.audio_queue.task_done()
                
            except queue.Empty:
                # Queue timeout, continue loop
                continue
            except Exception as e:
                print(f"Error processing audio chunk: {e}")
    
    def start(self):
        """
        Start continuous audio streaming and processing.
        """
        if self.running:
            print("Audio streaming already running")
            return
        
        self.running = True
        
        # Start processing thread
        self.process_thread = threading.Thread(
            target=self.process_queue,
            daemon=True
        )
        self.process_thread.start()
        
        # Start audio stream
        self.stream = sd.InputStream(
            callback=self.audio_callback,
            channels=self.channels,
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            device=self.device
        )
        self.stream.start()
        
        print(f"Audio streaming started (sample rate: {self.sample_rate}Hz, "
              f"chunk size: {self.chunk_size} samples, "
              f"device: {self.device or 'default'})")
    
    def stop(self):
        """
        Stop streaming and clean up resources.
        """
        if not self.running:
            return
        
        self.running = False
        
        # Stop and close audio stream
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        
        # Wait for processing thread to finish
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=1)
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
        
        print("Audio streaming stopped")
    
    def get_audio_level(self):
        """
        Get the current audio level, useful for UI feedback.
        
        Returns:
            tuple: (current_level, peak_level) as float values from 0.0 to 1.0
        """
        # Reset peak after reading to track new peaks
        peak = self.peak_audio_level
        self.peak_audio_level = self.current_audio_level
        
        return self.current_audio_level, peak
    
    def save_audio(self, audio_data, filename="recorded_audio.wav"):
        """
        Save audio data to WAV file.
        
        Args:
            audio_data (numpy.ndarray): Audio data to save
            filename (str): Output filename
        """
        # Ensure directories exist
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
        
        # Convert float32 to int16 for WAV file
        audio_data_int16 = (audio_data * 32767).astype(np.int16)
        
        # Write WAV file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio_data_int16.tobytes())
        
        print(f"Audio saved to {filename}")


class StreamingAudioPlayback:
    """
    Handles streaming audio playback for real-time TTS output.
    
    This class manages a queue of audio chunks to be played sequentially,
    allowing for continuous speech output from the TTS system.
    """
    
    def __init__(self, sample_rate=16000, channels=1, device=None):
        """
        Initialize streaming audio playback.
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1)
            device: Audio device to use (default: system default)
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        
        # Queue for audio chunks
        self.audio_queue = queue.Queue()
        
        # Thread control
        self.running = False
        self.playback_thread = None
    
    def start(self):
        """
        Start the audio playback thread.
        """
        if self.running:
            return
        
        self.running = True
        self.playback_thread = threading.Thread(
            target=self._playback_worker,
            daemon=True
        )
        self.playback_thread.start()
        
        print(f"Audio playback started (sample rate: {self.sample_rate}Hz, "
              f"device: {self.device or 'default'})")
    
    def stop(self):
        """
        Stop playback and clean up resources.
        """
        if not self.running:
            return
        
        self.running = False
        
        # Wait for playback thread to finish
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1)
        
        # Clear queue
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break
        
        print("Audio playback stopped")
    
    def play(self, audio_data):
        """
        Queue audio data for playback.
        
        Args:
            audio_data: NumPy array of audio samples
        """
        # Check if audio data is not empty
        if audio_data is not None and len(audio_data) > 0:
            self.audio_queue.put(audio_data)
    
    def _playback_worker(self):
        """
        Worker thread that continuously plays audio chunks from the queue.
        """
        while self.running:
            try:
                # Get audio from queue with timeout
                audio_data = self.audio_queue.get(timeout=0.5)
                
                # Play audio
                sd.play(audio_data, self.sample_rate)
                sd.wait()  # Wait until audio is done playing
                
                # Mark task as done
                self.audio_queue.task_done()
                
            except queue.Empty:
                # Queue timeout, continue loop
                continue
            except Exception as e:
                print(f"Error playing audio: {e}")