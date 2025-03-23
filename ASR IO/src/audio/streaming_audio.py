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
    small chunks of audio with overlap between chunks
    to ensure no speech is lost at chunk boundaries.
    """
    
    def __init__(self, callback, sample_rate=16000, channels=1, 
                 chunk_duration=1.0, overlap=0.25, device=None):
        """
        Initialize streaming audio capture.
        
        Args:
            callback: Function to be called with each audio chunk
            sample_rate: Audio sample rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1)
            chunk_duration: Duration of each audio chunk in seconds (default: 1.0)
            overlap: Overlap between chunks as a fraction of chunk_duration (default: 0.25)
            device: Audio device to use (default: system default)
        """
        self.callback = callback
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = int(chunk_duration * sample_rate)
        self.overlap_size = int(overlap * self.chunk_size)
        self.device = device
        
        # List available devices
        print("\nAvailable audio devices:")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            print(f"{i}: {dev['name']} (inputs: {dev['max_input_channels']}, outputs: {dev['max_output_channels']})")
        
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
        
        # Voice Activity Detection parameters
        self.vad_threshold = 0.003  # Lowered threshold for better sensitivity
        self.min_active_ratio = 0.05  # Lowered minimum active frames ratio
        
        # Diagnostic info
        self.total_chunks_captured = 0
        self.chunks_with_activity = 0
        self.last_error = None
        
        print(f"Initialized audio capture: {chunk_duration}s chunks with {overlap * 100:.0f}% overlap")
    
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
            self.last_error = str(status)
        
        # Convert to mono if needed and ensure correct data type
        if indata.ndim > 1 and indata.shape[1] > 1:
            current_data = indata[:, 0].astype(np.float32)
        else:
            current_data = indata.flatten().astype(np.float32)
        
        # Check for NaN or Inf values
        if np.isnan(current_data).any() or np.isinf(current_data).any():
            print("WARNING: NaN or Inf values detected in audio input")
            current_data = np.nan_to_num(current_data)
        
        # Update audio level metrics
        audio_min = np.min(current_data)
        audio_max = np.max(current_data)
        self.current_audio_level = np.mean(np.abs(current_data))
        self.peak_audio_level = max(self.peak_audio_level, np.max(np.abs(current_data)))
        
        # Basic Voice Activity Detection
        active_frames = np.sum(np.abs(current_data) > self.vad_threshold)
        active_ratio = active_frames / len(current_data)
        
        # More detailed audio diagnostics
        self.total_chunks_captured += 1
        if active_ratio > self.min_active_ratio:
            self.chunks_with_activity += 1
        
        # Every 10 chunks, print detailed diagnostics
        if self.total_chunks_captured % 10 == 0:
            print(f"Audio diagnostics: min={audio_min:.4f}, max={audio_max:.4f}, " 
                  f"mean={self.current_audio_level:.4f}, peak={self.peak_audio_level:.4f}, "
                  f"active_ratio={active_ratio:.2f}, device={self.device or 'default'}")
            
            # Check for very quiet audio over time
            activity_percentage = (self.chunks_with_activity / self.total_chunks_captured) * 100
            if self.total_chunks_captured >= 50 and activity_percentage < 5:
                print(f"WARNING: Very little audio activity detected ({activity_percentage:.1f}%). "
                      f"Check your microphone or adjust the VAD threshold.")
        
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
                # Get chunk from queue with reduced timeout for faster response
                chunk = self.audio_queue.get(timeout=0.2)
                
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
                self.last_error = str(e)
                print(f"Error processing audio chunk: {e}")
    
    def start(self):
        """
        Start continuous audio streaming and processing.
        """
        if self.running:
            print("Audio streaming already running")
            return
        
        self.running = True
        
        # Reset diagnostics
        self.total_chunks_captured = 0
        self.chunks_with_activity = 0
        self.last_error = None
        
        # Start processing thread
        self.process_thread = threading.Thread(
            target=self.process_queue,
            daemon=True
        )
        self.process_thread.start()
        
        # Start audio stream
        try:
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
                
        except Exception as e:
            self.last_error = str(e)
            print(f"ERROR starting audio stream: {e}")
            self.running = False
            
            # Try to find a working audio device
            try:
                print("Attempting to find a working audio device...")
                devices = sd.query_devices()
                for i, dev in enumerate(devices):
                    if dev['max_input_channels'] > 0:
                        print(f"Trying device {i}: {dev['name']}")
                        try:
                            self.device = i
                            self.stream = sd.InputStream(
                                callback=self.audio_callback,
                                channels=self.channels,
                                samplerate=self.sample_rate,
                                blocksize=self.chunk_size,
                                device=i
                            )
                            self.stream.start()
                            self.running = True
                            print(f"Successfully using audio device {i}: {dev['name']}")
                            break
                        except Exception as e2:
                            print(f"  Failed to use device {i}: {e2}")
            except Exception as e3:
                print(f"Error during device enumeration: {e3}")
            
            if not self.running:
                print("Could not find a working audio input device.")
    
    def stop(self):
        """
        Stop streaming and clean up resources.
        """
        if not self.running:
            return
        
        self.running = False
        
        # Stop and close audio stream
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
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
    
    def get_diagnostics(self):
        """
        Get detailed diagnostic information.
        
        Returns:
            dict: Diagnostic information
        """
        return {
            "total_chunks": self.total_chunks_captured,
            "chunks_with_activity": self.chunks_with_activity,
            "activity_percentage": (self.chunks_with_activity / max(1, self.total_chunks_captured)) * 100,
            "current_level": self.current_audio_level, 
            "peak_level": self.peak_audio_level,
            "last_error": self.last_error,
            "device": self.device
        }
    
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
        
        # List available output devices
        print("\nAvailable audio output devices:")
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if dev['max_output_channels'] > 0:
                print(f"{i}: {dev['name']} ({dev['max_output_channels']} output channels)")
        
        # Queue for audio chunks
        self.audio_queue = queue.Queue()
        
        # Thread control
        self.running = False
        self.playback_thread = None
        
        # Diagnostics
        self.chunks_played = 0
        self.last_error = None
        self.currently_playing = False
    
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
        
        # Stop any currently playing audio
        try:
            sd.stop()
        except Exception as e:
            print(f"Error stopping sounddevice: {e}")
        
        print("Audio playback stopped")
    
    def play(self, audio_data):
        """
        Queue audio data for playback.
        
        Args:
            audio_data: NumPy array of audio samples
        """
        # Check if audio data is not empty
        if audio_data is not None and len(audio_data) > 0:
            # Check for NaN or Inf values
            if np.isnan(audio_data).any() or np.isinf(audio_data).any():
                print("WARNING: NaN or Inf values detected in audio output")
                audio_data = np.nan_to_num(audio_data)
                
            # Add some diagnostics
            print(f"Queuing audio for playback: {len(audio_data)} samples, "
                  f"min={np.min(audio_data):.3f}, max={np.max(audio_data):.3f}")
            
            self.audio_queue.put(audio_data)
    
    def _playback_worker(self):
        """
        Worker thread that continuously plays audio chunks from the queue.
        """
        while self.running:
            try:
                # Get audio from queue with reduced timeout for faster response
                audio_data = self.audio_queue.get(timeout=0.2)
                
                # Set currently playing flag
                self.currently_playing = True
                
                # Play audio
                try:
                    sd.play(audio_data, self.sample_rate, device=self.device)
                    sd.wait()  # Wait until audio is done playing
                    self.chunks_played += 1
                except Exception as e:
                    self.last_error = str(e)
                    print(f"Error during audio playback: {e}")
                
                # Mark task as done
                self.audio_queue.task_done()
                self.currently_playing = False
                
            except queue.Empty:
                # Queue timeout, continue loop
                self.currently_playing = False
                continue
            except Exception as e:
                self.last_error = str(e)
                print(f"Error in playback worker: {e}")
                self.currently_playing = False
    
    def get_diagnostics(self):
        """
        Get diagnostic information about playback.
        
        Returns:
            dict: Diagnostic information
        """
        return {
            "chunks_played": self.chunks_played,
            "queue_size": self.audio_queue.qsize(),
            "currently_playing": self.currently_playing,
            "last_error": self.last_error,
            "device": self.device
        }