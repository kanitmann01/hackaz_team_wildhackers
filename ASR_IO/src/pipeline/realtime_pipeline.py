import threading
import time
import queue
from threading import Thread


class RealTimeTranslationPipeline:
    """
    Integrated pipeline for real-time speech-to-speech translation.
    
    This class connects all components (audio capture, ASR, MT, TTS)
    into a cohesive streaming pipeline that processes speech continuously
    and produces translated speech with minimal latency.
    """
    
    def __init__(self, asr_model, mt_model, tts_model, audio_capture, audio_playback):
        """
        Initialize the real-time translation pipeline.
        
        Args:
            asr_model: Streaming ASR model instance
            mt_model: Streaming MT model instance
            tts_model: Streaming TTS model instance
            audio_capture: Audio capture component
            audio_playback: Audio playback component
        """
        # Store component references
        self.asr = asr_model
        self.translator = mt_model
        self.tts = tts_model
        self.audio_capture = audio_capture
        self.audio_playback = audio_playback
        
        # State tracking
        self.running = False
        self.paused = False
        self.lock = threading.RLock()
        
        # Create communication queues
        self.asr_queue = queue.Queue()
        self.mt_queue = queue.Queue()
        self.tts_queue = queue.Queue()
        
        # UI update tracking
        self.current_source_text = ""
        self.current_translated_text = ""
        self.update_callback = None
        
        # Background workers
        self.asr_worker = None
        self.mt_worker = None
        self.tts_worker = None
        
        # Performance tracking
        self.start_time = None
        self.stats = {
            "audio_chunks_processed": 0,
            "asr_chunks_processed": 0,
            "mt_chunks_processed": 0,
            "tts_chunks_processed": 0,
            "total_latency": 0
        }
    
    def set_update_callback(self, callback):
        """
        Set callback function for UI updates.
        
        Args:
            callback: Function to call with updated text
        """
        self.update_callback = callback
    
    def set_languages(self, source_lang, target_lang):
        """
        Set source and target languages.
        
        Args:
            source_lang: Source language code
            target_lang: Target language code
        """
        with self.lock:
            self.asr.set_language(source_lang)
            self.translator.set_languages(source_lang, target_lang)
    
    def reset(self):
        """Reset all component states and clear buffers."""
        with self.lock:
            self.asr.reset_context()
            self.translator.reset_context()
            self.current_source_text = ""
            self.current_translated_text = ""
            
            # Clear all queues
            self._clear_queue(self.asr_queue)
            self._clear_queue(self.mt_queue)
            self._clear_queue(self.tts_queue)
            
            # Reset performance tracking
            self.stats = {
                "audio_chunks_processed": 0,
                "asr_chunks_processed": 0,
                "mt_chunks_processed": 0,
                "tts_chunks_processed": 0,
                "total_latency": 0
            }
    
    def _clear_queue(self, q):
        """
        Clear a queue safely.
        
        Args:
            q: Queue to clear
        """
        try:
            while True:
                q.get_nowait()
                q.task_done()
        except queue.Empty:
            pass
    
    def process_audio_chunk(self, audio_chunk, sample_rate):
        """
        Process an audio chunk from the capture device.
        
        This is the entry point for audio data into the pipeline.
        
        Args:
            audio_chunk: NumPy array of audio samples
            sample_rate: Sample rate of the audio
        """
        # Skip if paused
        if self.paused:
            return
            
        # Add chunk timestamp for latency tracking
        chunk_data = {
            'audio': audio_chunk,
            'sample_rate': sample_rate,
            'timestamp': time.time()
        }
        
        # Queue for ASR processing
        self.asr_queue.put(chunk_data)
        
        # Update stats
        with self.lock:
            self.stats["audio_chunks_processed"] += 1
    
    def _asr_worker_thread(self):
        """Worker thread for ASR processing."""
        while self.running:
            try:
                # Get chunk from queue with timeout
                chunk_data = self.asr_queue.get(timeout=0.5)
                
                # Skip if paused
                if self.paused:
                    self.asr_queue.task_done()
                    continue
                
                # Process audio chunk
                asr_result = self.asr.transcribe_chunk(
                    chunk_data['audio'], 
                    chunk_data['sample_rate']
                )
                
                # Only process if new text detected
                if asr_result['text']:
                    # Update source text
                    with self.lock:
                        self.current_source_text = asr_result['full_text']
                    
                    # Add to translation queue with timestamp
                    self.mt_queue.put({
                        'text': asr_result['text'],
                        'timestamp': chunk_data['timestamp']
                    })
                    
                    # Update stats
                    with self.lock:
                        self.stats["asr_chunks_processed"] += 1
                
                # Call update callback if set
                if self.update_callback:
                    self.update_callback(
                        self.current_source_text, 
                        self.current_translated_text
                    )
                
                # Mark task as done
                self.asr_queue.task_done()
                
            except queue.Empty:
                # Queue timeout, continue loop
                continue
            except Exception as e:
                print(f"Error in ASR worker: {e}")
                # Mark task as done to avoid blocking
                try:
                    self.asr_queue.task_done()
                except:
                    pass
    
    def _mt_worker_thread(self):
        """Worker thread for MT processing."""
        while self.running:
            try:
                # Get chunk from queue with timeout
                chunk_data = self.mt_queue.get(timeout=0.5)
                
                # Skip if paused
                if self.paused:
                    self.mt_queue.task_done()
                    continue
                
                # Process text chunk
                mt_result = self.translator.translate_chunk(chunk_data['text'])
                
                # Only process if new translation detected
                if mt_result['text']:
                    # Update translated text
                    with self.lock:
                        self.current_translated_text = mt_result['full_text']
                    
                    # Add to TTS queue with timestamp
                    self.tts_queue.put({
                        'text': mt_result['text'],
                        'timestamp': chunk_data['timestamp']
                    })
                    
                    # Update stats
                    with self.lock:
                        self.stats["mt_chunks_processed"] += 1
                
                # Call update callback if set
                if self.update_callback:
                    self.update_callback(
                        self.current_source_text, 
                        self.current_translated_text
                    )
                
                # Mark task as done
                self.mt_queue.task_done()
                
            except queue.Empty:
                # Queue timeout, continue loop
                continue
            except Exception as e:
                print(f"Error in MT worker: {e}")
                # Mark task as done to avoid blocking
                try:
                    self.mt_queue.task_done()
                except:
                    pass
    
    def _tts_worker_thread(self):
        """Worker thread for TTS processing."""
        while self.running:
            try:
                # Get chunk from queue with timeout
                chunk_data = self.tts_queue.get(timeout=0.5)
                
                # Skip if paused
                if self.paused:
                    self.tts_queue.task_done()
                    continue
                
                # Process text chunk and play audio
                audio_data = self.tts.synthesize_speech(
                    chunk_data['text'],
                    play_callback=self.audio_playback.play
                )
                
                # Calculate and track end-to-end latency
                if chunk_data['timestamp'] > 0:
                    latency = time.time() - chunk_data['timestamp']
                    with self.lock:
                        self.stats["total_latency"] += latency
                        self.stats["tts_chunks_processed"] += 1
                
                # Mark task as done
                self.tts_queue.task_done()
                
            except queue.Empty:
                # Queue timeout, continue loop
                continue
            except Exception as e:
                print(f"Error in TTS worker: {e}")
                # Mark task as done to avoid blocking
                try:
                    self.tts_queue.task_done()
                except:
                    pass
    
    def start(self):
        """Start the translation pipeline."""
        if self.running:
            print("Pipeline already running")
            return
        
        print("Starting real-time translation pipeline...")
        
        # Set running state
        self.running = True
        self.paused = False
        
        # Reset state
        self.reset()
        
        # Start background workers
        self.asr_worker = Thread(
            target=self._asr_worker_thread,
            daemon=True,
            name="ASR-Worker"
        )
        self.asr_worker.start()
        
        self.mt_worker = Thread(
            target=self._mt_worker_thread,
            daemon=True,
            name="MT-Worker"
        )
        self.mt_worker.start()
        
        self.tts_worker = Thread(
            target=self._tts_worker_thread,
            daemon=True,
            name="TTS-Worker"
        )
        self.tts_worker.start()
        
        # Start audio capture and playback
        self.audio_playback.start()
        self.audio_capture.callback = self.process_audio_chunk
        self.audio_capture.start()
        
        # Set start time for stats
        self.start_time = time.time()
        
        print("Pipeline started successfully")
    
    def stop(self):
        """Stop the translation pipeline."""
        if not self.running:
            return
        
        print("Stopping translation pipeline...")
        
        # Set running state
        self.running = False
        
        # Stop audio capture and playback
        self.audio_capture.stop()
        self.audio_playback.stop()
        
        # Wait for workers to finish
        if self.asr_worker and self.asr_worker.is_alive():
            self.asr_worker.join(timeout=1)
        
        if self.mt_worker and self.mt_worker.is_alive():
            self.mt_worker.join(timeout=1)
        
        if self.tts_worker and self.tts_worker.is_alive():
            self.tts_worker.join(timeout=1)
        
        print("Pipeline stopped")
    
    def pause(self):
        """Pause the translation pipeline."""
        if not self.running or self.paused:
            return
        
        with self.lock:
            self.paused = True
        
        print("Pipeline paused")
    
    def resume(self):
        """Resume the translation pipeline."""
        if not self.running or not self.paused:
            return
        
        with self.lock:
            self.paused = False
        
        print("Pipeline resumed")
    
    def get_stats(self):
        """
        Get pipeline performance statistics.
        
        Returns:
            dict: Performance statistics
        """
        with self.lock:
            stats_copy = self.stats.copy()
            
            # Calculate averages
            if stats_copy["tts_chunks_processed"] > 0:
                stats_copy["avg_latency"] = stats_copy["total_latency"] / stats_copy["tts_chunks_processed"]
            else:
                stats_copy["avg_latency"] = 0
            
            # Add component-specific stats
            stats_copy["asr_stats"] = self.asr.get_stats()
            stats_copy["mt_stats"] = self.translator.get_stats()
            stats_copy["tts_stats"] = self.tts.get_stats()
            
            # Add uptime
            if self.start_time:
                stats_copy["uptime"] = time.time() - self.start_time
            else:
                stats_copy["uptime"] = 0
            
            return stats_copy