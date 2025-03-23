import gradio as gr
import torch
import time
import threading
import os
import sys
import numpy as np
import traceback
import json
import wave

# Add parent directory to path to allow for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

# Ensure data directories exist
os.makedirs("data/samples", exist_ok=True)
os.makedirs("data/models", exist_ok=True)

# Initialize console log
console_output = []

def log(message):
    """Add a message to the console log with timestamp."""
    timestamp = time.strftime("%H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    console_output.append(log_message)
    print(log_message)
    return "\n".join(console_output)

# Initialize global component variables
latest_audio_chunk = None
audio_capture_component = None
asr_model_component = None
mt_model_component = None
tts_model_component = None
audio_playback_component = None
pipeline_component = None

# ===== AUDIO CAPTURE TESTING =====

def init_audio_capture(chunk_duration, overlap):
    """Initialize audio capture component."""
    try:
        from src.audio.streaming_audio import StreamingAudioCapture
        
        log(f"Initializing audio capture (chunk_duration={chunk_duration}s, overlap={overlap})")
        
        # Create audio capture with a callback that logs audio levels
        def audio_callback(audio_chunk, sample_rate):
            level = np.mean(np.abs(audio_chunk))
            peak = np.max(np.abs(audio_chunk))
            log(f"Audio received: {len(audio_chunk)} samples, level={level:.4f}, peak={peak:.4f}")
            
            # Save reference to latest audio chunk
            global latest_audio_chunk
            latest_audio_chunk = audio_chunk
        
        # Create the capture component
        global audio_capture_component
        audio_capture_component = StreamingAudioCapture(
            callback=audio_callback,
            chunk_duration=chunk_duration,
            overlap=overlap
        )
        
        log("Audio capture initialized successfully")
        return "Success"
    except Exception as e:
        error_message = f"Error initializing audio capture: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return error_message

def start_audio_capture():
    """Start audio capture."""
    global audio_capture_component
    
    if not audio_capture_component:
        return log("Error: Audio capture not initialized")
    
    try:
        audio_capture_component.start()
        return log("Audio capture started")
    except Exception as e:
        error_message = f"Error starting audio capture: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return log(error_message)

def stop_audio_capture():
    """Stop audio capture."""
    global audio_capture_component
    
    if not audio_capture_component:
        return log("Error: Audio capture not initialized")
    
    try:
        audio_capture_component.stop()
        return log("Audio capture stopped")
    except Exception as e:
        error_message = f"Error stopping audio capture: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return log(error_message)

def get_audio_levels():
    """Get current audio levels."""
    global audio_capture_component
    
    if not audio_capture_component:
        return 0, 0
    
    try:
        current, peak = audio_capture_component.get_audio_level()
        return current, peak
    except Exception as e:
        log(f"Error getting audio levels: {str(e)}")
        return 0, 0

def save_audio_sample(filename):
    """Save current audio buffer to a file."""
    global audio_capture_component, latest_audio_chunk
    
    if not audio_capture_component:
        return log("Error: Audio capture not initialized")
    
    if not filename:
        filename = f"data/samples/audio_sample_{int(time.time())}.wav"
    
    try:
        # Ensure filename has .wav extension
        if not filename.endswith(".wav"):
            filename += ".wav"
        
        # Ensure it's in data/samples directory
        if not filename.startswith("data/samples/"):
            filename = f"data/samples/{os.path.basename(filename)}"
        
        # Get the latest audio chunk
        if latest_audio_chunk is None or len(latest_audio_chunk) == 0:
            return log("No audio data available to save")
        
        # Save to file
        audio_int16 = (latest_audio_chunk * 32767).astype(np.int16)
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(16000)  # Using standard Whisper sample rate
            wf.writeframes(audio_int16.tobytes())
        
        return log(f"Audio saved to {filename}")
    except Exception as e:
        error_message = f"Error saving audio: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return log(error_message)

# ===== ASR (SPEECH RECOGNITION) TESTING =====

def init_asr_model(model_name, use_gpu):
    """Initialize ASR model."""
    try:
        from src.asr.whisper_asr import WhisperASR
        
        # Set device
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            log(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            log("Using CPU for ASR")
        
        log(f"Initializing Whisper ASR model: {model_name}")
        
        # Create the model
        global asr_model_component
        asr_model_component = WhisperASR(model_name=model_name, device=device)
        
        log("ASR model initialized successfully")
        return "Success"
    except Exception as e:
        error_message = f"Error initializing ASR model: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return error_message

def test_asr_model():
    """Run a basic test of the ASR model."""
    global asr_model_component
    
    if not asr_model_component:
        return log("Error: ASR model not initialized")
    
    try:
        log("Running ASR model test...")
        start_time = time.time()
        
        # Use the test_transcription method if available
        if hasattr(asr_model_component, 'test_transcription'):
            test_result = asr_model_component.test_transcription()
            end_time = time.time()
            
            if test_result:
                log(f"ASR test successful (took {end_time - start_time:.2f}s)")
            else:
                log(f"ASR test failed (took {end_time - start_time:.2f}s)")
        else:
            # Manual test with a sine wave if test method not available
            sample_rate = 16000
            duration = 2  # seconds
            t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
            audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
            
            result = asr_model_component.transcribe_chunk(audio, sample_rate)
            end_time = time.time()
            
            log(f"ASR test with sine wave completed in {end_time - start_time:.2f}s")
            log(f"Result: {result['full_text']}")
        
        return log("ASR test complete")
    except Exception as e:
        error_message = f"Error testing ASR model: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return log(error_message)

def transcribe_file(file_path):
    """Transcribe an audio file."""
    global asr_model_component
    
    if not asr_model_component:
        return log("Error: ASR model not initialized")
    
    if not file_path or not os.path.exists(file_path):
        return log(f"Error: File not found - {file_path}")
    
    try:
        log(f"Transcribing file: {file_path}")
        start_time = time.time()
        
        # Load audio file
        with wave.open(file_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        
        # Process with ASR
        result = asr_model_component.transcribe_chunk(audio, sample_rate)
        end_time = time.time()
        
        # Log result
        log(f"Transcription completed in {end_time - start_time:.2f}s")
        log(f"Transcription: {result['full_text']}")
        
        return log("Transcription complete")
    except Exception as e:
        error_message = f"Error transcribing file: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return log(error_message)

def transcribe_live(duration=5):
    """Transcribe live audio from capture device."""
    global asr_model_component, audio_capture_component
    
    if not asr_model_component:
        return log("Error: ASR model not initialized")
    
    if not audio_capture_component:
        return log("Error: Audio capture not initialized")
    
    try:
        log(f"Starting live transcription for {duration} seconds...")
        
        # Create a thread to run the transcription
        def transcribe_thread():
            # Reset ASR context
            asr_model_component.reset_context()
            
            # Create a new callback that processes with ASR
            original_callback = audio_capture_component.callback
            
            def asr_callback(audio_chunk, sample_rate):
                # Call original callback for logging
                original_callback(audio_chunk, sample_rate)
                
                # Process with ASR
                result = asr_model_component.transcribe_chunk(audio_chunk, sample_rate)
                
                # Log result if there's new text
                if result['text']:
                    log(f"Transcribed: {result['text']}")
                    log(f"Full text: {result['full_text']}")
            
            # Set the new callback
            audio_capture_component.callback = asr_callback
            
            # Wait for the specified duration
            time.sleep(duration)
            
            # Restore original callback
            audio_capture_component.callback = original_callback
            
            log("Live transcription complete")
        
        # Start transcription thread
        thread = threading.Thread(target=transcribe_thread)
        thread.daemon = True
        thread.start()
        
        return log(f"Live transcription started for {duration} seconds")
    except Exception as e:
        error_message = f"Error in live transcription: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return log(error_message)

# ===== MT (TRANSLATION) TESTING =====

# Fix for the MT component in the diagnostic tool
# Replace the init_mt_model function with this corrected version

def init_mt_model(model_name, source_lang, target_lang, use_gpu):
    """Initialize MT model."""
    try:
        from src.mt.streaming_mt import StreamingTranslator
        from src.ui.language_utils import LanguageMapper
        
        # Convert language names to NLLB codes
        source_nllb = LanguageMapper.iso_to_nllb(source_lang)
        target_nllb = LanguageMapper.iso_to_nllb(target_lang)
        
        # Set device
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            log(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            log("Using CPU for MT")
        
        log(f"Initializing MT model: {model_name}")
        log(f"Source language: {source_lang} ({source_nllb})")
        log(f"Target language: {target_lang} ({target_nllb})")
        
        # Now, modify the StreamingTranslator class to use the correct method for target language
        # This is a temporary monkey patch to fix the issue without changing the original file
        
        # Save the original __init__ method
        original_init = StreamingTranslator.__init__
        
        # Define a new translate_chunk method that uses convert_tokens_to_ids
        def fixed_translate_chunk(self, text_chunk, force=False):
            """
            Translate a chunk of text incrementally.
            
            Args:
                text_chunk: New text to translate
                force: Force translation even for small chunks
                
            Returns:
                dict: Containing 'text' (new translation) and 'full_text' (complete translation)
            """
            start_time = time.time()
            
            # Skip empty chunks
            if not text_chunk or not text_chunk.strip():
                return {
                    'text': '',
                    'full_text': self.translated_buffer,
                    'processing_time': 0
                }
            
            # Update source buffer
            complete_source = self.source_buffer + " " + text_chunk if self.source_buffer else text_chunk
            complete_source = complete_source.strip()
            self.source_buffer = complete_source
            
            # Skip translation if chunk is too small (unless forced)
            words = complete_source.split()
            is_end_of_sentence = any(p in text_chunk for p in ['.', '!', '?', '。', '！', '？'])
            if len(words) < self.min_chunk_size and not force and not is_end_of_sentence:
                return {
                    'text': '',
                    'full_text': self.translated_buffer,
                    'processing_time': 0
                }
            
            # Tokenize with source language
            inputs = self.tokenizer(
                complete_source, 
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            # Generate translation with fixed approach for language code
            with torch.no_grad():
                # Use convert_tokens_to_ids instead of lang_code_to_id
                target_lang_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)
                
                translated_ids = self.model.generate(
                    **inputs,
                    forced_bos_token_id=target_lang_id,
                    max_length=512,  # Increased for longer context
                    num_beams=2,  # Faster beam search
                    length_penalty=1.0  # Balanced length penalty
                )
            
            # Decode translation
            translation = self.tokenizer.batch_decode(
                translated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Determine new translated text
            if self.translated_buffer and translation.startswith(self.translated_buffer):
                new_translation = translation[len(self.translated_buffer):].strip()
            else:
                new_translation = translation
            
            # Update translation buffer
            self.translated_buffer = translation
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.chunk_count += 1
            avg_processing_time = self.total_processing_time / self.chunk_count if self.chunk_count > 0 else 0
            
            return {
                'text': new_translation,
                'full_text': translation,
                'processing_time': processing_time,
                'avg_processing_time': avg_processing_time
            }
        
        # Replace the method in the class
        StreamingTranslator.translate_chunk = fixed_translate_chunk
        
        # Create the model
        global mt_model_component
        mt_model_component = StreamingTranslator(
            source_lang=source_nllb,
            target_lang=target_nllb,
            device=device,
            model_name=model_name
        )
        
        log("MT model initialized successfully")
        return "Success"
    except Exception as e:
        error_message = f"Error initializing MT model: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return error_message
    
def diagnose_tts_output(text, language_code):
    """Diagnose TTS output by examining the audio directly."""
    try:
        from src.tts.streaming_tts import StreamingTTS
        import numpy as np
        import wave
        import os
        
        log(f"Diagnosing TTS output for text: '{text}' in language: {language_code}")
        
        # Create a TTS model with CPU for simplicity
        tts = StreamingTTS(device=torch.device("cpu"))
        tts.set_language(language_code)
        
        # Generate speech without playback
        log("Generating speech...")
        audio = tts.synthesize_speech(text, play_callback=None)
        
        if audio is None:
            log("ERROR: TTS generated None instead of audio data")
            return "Failed: No audio generated"
        
        # Analyze the audio data
        log(f"Audio data shape: {audio.shape if hasattr(audio, 'shape') else 'unknown'}")
        log(f"Audio data type: {type(audio)}")
        
        if isinstance(audio, np.ndarray):
            log(f"Audio statistics:")
            log(f"  Length: {len(audio)} samples")
            log(f"  Min value: {np.min(audio):.8f}")
            log(f"  Max value: {np.max(audio):.8f}")
            log(f"  Mean value: {np.mean(audio):.8f}")
            log(f"  Std deviation: {np.std(audio):.8f}")
            
            # Check if the audio is mostly silent
            if np.max(np.abs(audio)) < 0.01:
                log("WARNING: Audio appears to be mostly silent")
                
                # Try amplifying and saving
                amplified = audio * 100  # Amplify by 100x
                filename = "data/samples/tts_amplified.wav"
                amplified_int16 = (amplified * 32767).astype(np.int16)
                
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(16000)
                    wf.writeframes(amplified_int16.tobytes())
                
                log(f"Saved amplified audio to {filename}")
        
        # Save raw audio for inspection
        filename = "data/samples/tts_diagnostic.wav"
        audio_int16 = (audio * 32767).astype(np.int16)
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_int16.tobytes())
        
        log(f"Saved diagnostic audio to {filename}")
        
        # Examine TTS model parameters
        log("TTS model parameters:")
        log(f"  Model name: {tts.model.__class__.__name__ if hasattr(tts, 'model') else 'unknown'}")
        log(f"  Language code: {tts.language_code}")
        
        log("TTS diagnosis complete")
        return "TTS diagnosis complete - check logs for details"
        
    except Exception as e:
        error_message = f"Error in TTS diagnosis: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return error_message


def translate_text(text):
    """Translate a piece of text."""
    global mt_model_component
    
    if not mt_model_component:
        return log("Error: MT model not initialized")
    
    if not text:
        return log("Error: No text provided for translation")
    
    try:
        log(f"Translating text: {text}")
        start_time = time.time()
        
        # Translate text
        result = mt_model_component.translate_chunk(text)
        end_time = time.time()
        
        # Log result
        log(f"Translation completed in {end_time - start_time:.2f}s")
        log(f"Translation: {result['full_text']}")
        
        return log("Translation complete")
    except Exception as e:
        error_message = f"Error translating text: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return log(error_message)

def test_complete_audio_system():
    """Test the complete audio generation and playback system step by step."""
    try:
        import sounddevice as sd
        import numpy as np
        import time
        import wave
        import os
        from src.audio.streaming_audio import StreamingAudioPlayback
        
        log("============ COMPLETE AUDIO SYSTEM TEST ============")
        
        # Step 1: Test basic sound device availability
        log("Step 1: Testing sound device availability")
        devices = sd.query_devices()
        
        output_devices = []
        for i, dev in enumerate(devices):
            if dev['max_output_channels'] > 0:
                output_devices.append((i, dev['name']))
                log(f"  Output device {i}: {dev['name']} ({dev['max_output_channels']} channels)")
        
        if not output_devices:
            log("ERROR: No output devices found!")
            return "Failed: No output devices"
        
        # Step 2: Generate a test sine wave
        log("Step 2: Generating test audio")
        sample_rate = 16000
        duration = 2  # seconds
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
        
        # Generate a more noticeable sound (sweeping frequency)
        audio = np.zeros_like(t)
        for freq in [264, 330, 396, 440]:  # Simple musical notes
            audio += 0.2 * np.sin(2 * np.pi * freq * t)
        
        # Normalize
        audio = audio / np.max(np.abs(audio)) * 0.9
        
        log(f"  Generated {len(audio)} samples at {sample_rate}Hz")
        log(f"  Audio range: min={np.min(audio):.4f}, max={np.max(audio):.4f}")
        
        # Step 3: Save to file
        log("Step 3: Saving audio to file")
        test_file = "data/samples/audio_test.wav"
        audio_int16 = (audio * 32767).astype(np.int16)
        
        with wave.open(test_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 2 bytes for int16
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        file_size = os.path.getsize(test_file)
        log(f"  Saved to {test_file} ({file_size} bytes)")
        
        # Step 4: Test direct playback using sounddevice
        log("Step 4: Testing direct audio playback with sounddevice")
        try:
            sd.play(audio, sample_rate)
            log("  Audio playback started - you should hear a musical note sequence")
            sd.wait()
            log("  Direct playback completed")
        except Exception as e:
            log(f"  ERROR in direct playback: {e}")
        
        # Step 5: Test with StreamingAudioPlayback class
        log("Step 5: Testing StreamingAudioPlayback class")
        try:
            playback = StreamingAudioPlayback()
            playback.start()
            log("  StreamingAudioPlayback started")
            
            playback.play(audio)
            log("  Audio queued for playback - you should hear the same notes again")
            
            # Wait for playback to complete
            time.sleep(3)
            
            # Stop playback
            playback.stop()
            log("  StreamingAudioPlayback stopped")
        except Exception as e:
            log(f"  ERROR in StreamingAudioPlayback: {e}")
        
        # Step 6: Test audio file validation
        log("Step 6: Testing audio file validation")
        try:
            with wave.open(test_file, 'rb') as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                frame_rate = wf.getframerate()
                n_frames = wf.getnframes()
                
                log(f"  WAV format: {channels} channels, {sample_width * 8} bits, {frame_rate}Hz")
                log(f"  Duration: {n_frames / frame_rate:.2f} seconds")
                
                # Read all frames
                frames = wf.readframes(n_frames)
                file_audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
                
                log(f"  File data: {len(file_audio)} samples")
                log(f"  File audio range: min={np.min(file_audio):.4f}, max={np.max(file_audio):.4f}")
                
                # Check if original and file data match
                if len(audio) == len(file_audio):
                    diff = np.mean(np.abs(audio - file_audio))
                    log(f"  Average difference between original and file data: {diff:.8f}")
                else:
                    log(f"  WARNING: Original audio length ({len(audio)}) != file audio length ({len(file_audio)})")
        except Exception as e:
            log(f"  ERROR in file validation: {e}")
        
        log("============ AUDIO TEST COMPLETE ============")
        return "Audio test complete - check logs for detailed results"
    
    except Exception as e:
        error_message = f"Error in audio system test: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return error_message


def test_incremental_translation(sentences):
    """Test incremental translation with a series of sentences."""
    global mt_model_component
    
    if not mt_model_component:
        return log("Error: MT model not initialized")
    
    if not sentences:
        # Default test sentences
        sentences = [
            "Hello, ",
            "my name is John. ",
            "I am testing the translation system. ",
            "This is a streaming test with multiple chunks."
        ]
    else:
        # Split the input into sentences
        sentences = sentences.split(". ")
        sentences = [s.strip() + ". " if not s.endswith(".") else s for s in sentences]
    
    try:
        log("Testing incremental translation...")
        
        # Reset translator context
        mt_model_component.reset_context()
        
        # Process each sentence incrementally
        for i, sentence in enumerate(sentences):
            log(f"Chunk {i+1}: {sentence}")
            
            start_time = time.time()
            result = mt_model_component.translate_chunk(sentence)
            end_time = time.time()
            
            log(f"Incremental translation (took {end_time - start_time:.2f}s): {result['full_text']}")
            
            # Small delay to simulate streaming
            time.sleep(0.5)
        
        return log("Incremental translation test complete")
    except Exception as e:
        error_message = f"Error in incremental translation: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return log(error_message)

# ===== TTS (TEXT-TO-SPEECH) TESTING =====

def init_tts_model(language, use_gpu):
    """Initialize TTS model."""
    try:
        from src.tts.streaming_tts import StreamingTTS
        from src.ui.language_utils import LanguageMapper
        
        # Get language code for TTS
        language_code, model_name = LanguageMapper.iso_to_mms(language)
        
        log(f"Language mapping: {language} -> code={language_code}, model={model_name}")
        
        # Set device
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            log(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            log("Using CPU for TTS")
        
        log(f"Initializing TTS model for language: {language} ({language_code})")
        log(f"Model: {model_name}")
        
        # Create the model with verbose logging
        log("About to create TTS model")
        
        # Diagnostic: Import and check model classes directly
        try:
            from transformers import AutoProcessor, AutoModel
            log("Successfully imported transformers classes")
            
            # Check if model exists in Hugging Face
            log(f"Checking if model exists: {model_name}")
            processor = AutoProcessor.from_pretrained(model_name)
            log("Processor loaded successfully")
            
            model = AutoModel.from_pretrained(model_name)
            log("Model loaded successfully")
            
        except Exception as model_check_error:
            log(f"Error checking model directly: {str(model_check_error)}")
        
        # Create the actual component
        global tts_model_component
        tts_model_component = StreamingTTS(device=device, model_name=model_name)
        tts_model_component.set_language(language_code)
        
        log("TTS model initialized successfully")
        return "Success"
    except Exception as e:
        error_message = f"Error initializing TTS model: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return error_message
    
def init_audio_playback():
    """Initialize audio playback component."""
    try:
        from src.audio.streaming_audio import StreamingAudioPlayback
        
        log("Initializing audio playback")
        
        # Create the playback component
        global audio_playback_component
        audio_playback_component = StreamingAudioPlayback()
        audio_playback_component.start()
        
        log("Audio playback initialized successfully")
        return "Success"
    except Exception as e:
        error_message = f"Error initializing audio playback: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return error_message

def synthesize_speech(text):
    """Synthesize speech from text."""
    global tts_model_component, audio_playback_component
    
    if not tts_model_component:
        return log("Error: TTS model not initialized")
    
    if not audio_playback_component:
        return log("Error: Audio playback not initialized")
    
    if not text:
        return log("Error: No text provided for synthesis")
    
    try:
        log(f"Synthesizing speech for text: {text}")
        log(f"Using language code: {tts_model_component.language_code}")
        log(f"TTS model type: {type(tts_model_component.model).__name__}")
        
        start_time = time.time()
        
        # Synthesize speech with playback
        audio = tts_model_component.synthesize_speech(text, play_callback=audio_playback_component.play)
        end_time = time.time()
        
        if audio is not None:
            # Log result
            log(f"Speech synthesis completed in {end_time - start_time:.2f}s")
            log(f"Generated {len(audio)} audio samples")
            
            # Save to file
            filename = f"data/samples/tts_output_{int(time.time())}.wav"
            audio_int16 = (audio * 32767).astype(np.int16)
            with wave.open(filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 2 bytes for int16
                wf.setframerate(16000)  # Standard rate for TTS
                wf.writeframes(audio_int16.tobytes())
            
            log(f"Speech saved to {filename}")
        else:
            log("Failed to generate speech audio - audio is None")
        
        return log("Speech synthesis complete")
    except Exception as e:
        error_message = f"Error synthesizing speech: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return log(error_message)
    
def stop_audio_playback():
    """Stop audio playback."""
    global audio_playback_component
    
    if not audio_playback_component:
        return log("Error: Audio playback not initialized")
    
    try:
        audio_playback_component.stop()
        return log("Audio playback stopped")
    except Exception as e:
        error_message = f"Error stopping audio playback: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return log(error_message)

# ===== FULL PIPELINE TESTING =====

def init_pipeline(source_lang, target_lang, use_gpu):
    """Initialize the full translation pipeline."""
    try:
        from src.asr.whisper_asr import WhisperASR
        from src.mt.streaming_mt import StreamingTranslator
        from src.tts.streaming_tts import StreamingTTS
        from src.audio.streaming_audio import StreamingAudioCapture, StreamingAudioPlayback
        from src.pipeline.realtime_pipeline import RealTimeTranslationPipeline
        from src.ui.language_utils import LanguageMapper
        
        # Set device
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            log(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            log("Using CPU for pipeline")
        
        # Get language codes
        source_nllb = LanguageMapper.iso_to_nllb(source_lang)
        target_nllb = LanguageMapper.iso_to_nllb(target_lang)
        target_mms_code, target_mms_model = LanguageMapper.iso_to_mms(target_lang)
        
        log(f"Initializing pipeline with {source_lang} -> {target_lang}")
        log(f"NLLB codes: {source_nllb} -> {target_nllb}")
        log(f"TTS code: {target_mms_code}")
        
        # Initialize components
        log("Creating ASR model (Whisper tiny)...")
        asr_model = WhisperASR(model_name="tiny", device=device, language=source_lang)
        
        log("Creating MT model (NLLB-200)...")
        mt_model = StreamingTranslator(
            source_lang=source_nllb,
            target_lang=target_nllb,
            device=device
        )
        
        log("Creating TTS model (MMS-TTS)...")
        tts_model = StreamingTTS(device=device)
        tts_model.set_language(target_mms_code)
        
        log("Creating audio components...")
        audio_playback = StreamingAudioPlayback()
        
        # Create audio capture with a callback for pipeline
        audio_capture = StreamingAudioCapture(
            callback=lambda x, y: None,  # Will be replaced by pipeline
            chunk_duration=1.0,
            overlap=0.25
        )
        
        # Create pipeline
        log("Creating pipeline...")
        global pipeline_component
        pipeline_component = RealTimeTranslationPipeline(
            asr_model=asr_model,
            mt_model=mt_model,
            tts_model=tts_model,
            audio_capture=audio_capture,
            audio_playback=audio_playback
        )
        
        # Set callback for updates
        def update_callback(source_text, translated_text):
            log(f"Pipeline update:")
            log(f"Source: {source_text}")
            log(f"Translation: {translated_text}")
        
        pipeline_component.set_update_callback(update_callback)
        
        log("Pipeline initialized successfully")
        return "Success"
    except Exception as e:
        error_message = f"Error initializing pipeline: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return error_message

def start_pipeline():
    """Start the translation pipeline."""
    global pipeline_component
    
    if not pipeline_component:
        return log("Error: Pipeline not initialized")
    
    try:
        log("Starting pipeline...")
        pipeline_component.start()
        return log("Pipeline started successfully")
    except Exception as e:
        error_message = f"Error starting pipeline: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return log(error_message)

def stop_pipeline():
    """Stop the translation pipeline."""
    global pipeline_component
    
    if not pipeline_component:
        return log("Error: Pipeline not initialized")
    
    try:
        log("Stopping pipeline...")
        pipeline_component.stop()
        return log("Pipeline stopped successfully")
    except Exception as e:
        error_message = f"Error stopping pipeline: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return log(error_message)

def reset_pipeline():
    """Reset the translation pipeline."""
    global pipeline_component
    
    if not pipeline_component:
        return log("Error: Pipeline not initialized")
    
    try:
        log("Resetting pipeline...")
        pipeline_component.reset()
        return log("Pipeline reset successfully")
    except Exception as e:
        error_message = f"Error resetting pipeline: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return log(error_message)


def process_audio_file_with_pipeline(file_path):
    """Process an audio file through the pipeline."""
    global pipeline_component
    
    if not pipeline_component:
        return log("Error: Pipeline not initialized")
    
    if not file_path or not os.path.exists(file_path):
        return log(f"Error: File not found - {file_path}")
    
    try:
        log(f"Processing audio file with pipeline: {file_path}")
        
        # Load audio file
        with wave.open(file_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
        
        log(f"Loaded {len(audio_data)} samples at {sample_rate}Hz")
        
        # Process in chunks to simulate streaming
        chunk_size = int(sample_rate * 0.5)  # 500ms chunks
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            if len(chunk) == chunk_size:  # Skip partial chunks
                # Process through pipeline
                pipeline_component.process_audio_chunk(chunk, sample_rate)
                log(f"Processed chunk {i//chunk_size + 1}/{len(audio_data)//chunk_size}")
                time.sleep(0.1)  # Small delay to simulate real-time
        
        return log("Audio file processing complete")
    except Exception as e:
        error_message = f"Error processing audio file: {str(e)}"
        log(error_message)
        log(traceback.format_exc())
        return log(error_message)

# ===== SYSTEM INFO =====

def get_system_info():
    """Get system information."""
    info = {
        "python_version": sys.version,
        "platform": sys.platform,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__
    }
    
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
    
    return json.dumps(info, indent=2)

def get_installed_packages():
    """Get list of installed packages."""
    import pkg_resources
    packages = sorted([f"{p.key}=={p.version}" for p in pkg_resources.working_set])
    return "\n".join(packages)

# ===== MAIN UI =====

def create_ui():
    """Create the diagnostic UI."""
    with gr.Blocks(title="Translation System Diagnostics") as demo:
        gr.Markdown("# Translation System Component Diagnostics")
        gr.Markdown("Test each component individually to diagnose issues")
        
        # Console output
        with gr.Row():
            console_log = gr.Textbox(
                label="Console Log",
                value="",
                lines=20,
                max_lines=100,
                interactive=False
            )
        
        with gr.Tabs():
            # ===== AUDIO CAPTURE TAB =====
            with gr.TabItem("Audio Capture"):
                gr.Markdown("## Audio Capture Testing")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            chunk_duration = gr.Slider(
                                minimum=0.5,
                                maximum=3.0,
                                value=1.0,
                                step=0.1,
                                label="Chunk Duration (seconds)"
                            )
                            overlap = gr.Slider(
                                minimum=0.1,
                                maximum=0.5,
                                value=0.25,
                                step=0.05,
                                label="Overlap Fraction"
                            )
                        
                        with gr.Row():
                            init_audio_btn = gr.Button("Initialize Audio Capture")
                            start_audio_btn = gr.Button("Start Capture")
                            stop_audio_btn = gr.Button("Stop Capture")
                            save_audio_btn = gr.Button("Save Sample")
                        
                        audio_filename = gr.Textbox(
                            label="Save Filename",
                            value="data/samples/audio_sample.wav",
                            interactive=True
                        )
                        
                    with gr.Column(scale=1):
                        input_level = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0,
                            label="Input Level",
                            interactive=False
                        )
                        
                        peak_level = gr.Slider(
                            minimum=0,
                            maximum=1,
                            value=0,
                            label="Peak Level",
                            interactive=False
                        )
                
                # Event handlers
                init_audio_btn.click(
                    fn=init_audio_capture,
                    inputs=[chunk_duration, overlap],
                    outputs=[console_log]
                )
                
                start_audio_btn.click(
                    fn=start_audio_capture,
                    outputs=[console_log]
                )
                
                stop_audio_btn.click(
                    fn=stop_audio_capture,
                    outputs=[console_log]
                )
                
                save_audio_btn.click(
                    fn=save_audio_sample,
                    inputs=[audio_filename],
                    outputs=[console_log]
                )
                
                # Update audio levels
                demo.load(
                    fn=get_audio_levels,
                    inputs=None,
                    outputs=[input_level, peak_level],
                    every=0.1
                )
            
            # ===== ASR TAB =====
            with gr.TabItem("ASR (Speech Recognition)"):
                gr.Markdown("## ASR Testing")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Row():
                            asr_model_name = gr.Dropdown(
                                ["tiny", "base", "small", "medium", "large"],
                                label="Whisper Model Size",
                                value="tiny"
                            )
                            asr_use_gpu = gr.Checkbox(
                                label="Use GPU if available",
                                value=True
                            )
                        
                        with gr.Row():
                            init_asr_btn = gr.Button("Initialize ASR Model")
                            test_asr_btn = gr.Button("Test ASR Model")
                        
                        with gr.Row():
                            asr_file_input = gr.Textbox(
                                label="Audio File Path",
                                value="data/samples/test_en.wav",
                                interactive=True
                            )
                            transcribe_file_btn = gr.Button("Transcribe File")
                        
                        with gr.Row():
                            live_duration = gr.Slider(
                                minimum=1,
                                maximum=30,
                                value=5,
                                step=1,
                                label="Live Duration (seconds)"
                            )
                            transcribe_live_btn = gr.Button("Transcribe Live Audio")
                
                # Event handlers
                init_asr_btn.click(
                    fn=init_asr_model,
                    inputs=[asr_model_name, asr_use_gpu],
                    outputs=[console_log]
                )
                
                test_asr_btn.click(
                    fn=test_asr_model,
                    outputs=[console_log]
                )
                
                transcribe_file_btn.click(
                    fn=transcribe_file,
                    inputs=[asr_file_input],
                    outputs=[console_log]
                )
                
                transcribe_live_btn.click(
                    fn=transcribe_live,
                    inputs=[live_duration],
                    outputs=[console_log]
                )
            
            # ===== MT TAB =====
            with gr.TabItem("MT (Translation)"):
                gr.Markdown("## MT Testing")
                
                with gr.Row():
                    with gr.Column():
                        mt_model_name = gr.Dropdown(
                            ["facebook/nllb-200-distilled-600M", "facebook/nllb-200-1.3B", "facebook/nllb-200-3.3B"],
                            label="Translation Model",
                            value="facebook/nllb-200-distilled-600M"
                        )
                        
                        with gr.Row():
                            mt_source_lang = gr.Dropdown(
                                list(sorted(["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"])),
                                label="Source Language",
                                value="en"
                            )
                            mt_target_lang = gr.Dropdown(
                                list(sorted(["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"])),
                                label="Target Language",
                                value="es"
                            )
                            
                        mt_use_gpu = gr.Checkbox(
                            label="Use GPU if available",
                            value=True
                        )
                        
                        init_mt_btn = gr.Button("Initialize MT Model")
                        
                        mt_input_text = gr.Textbox(
                            label="Text to Translate",
                            value="This is a test of the translation system.",
                            lines=3,
                            interactive=True
                        )
                        
                        translate_btn = gr.Button("Translate Text")
                        
                        mt_incremental_input = gr.Textbox(
                            label="Incremental Translation Text",
                            value="Hello. My name is John. I am testing the translation system.",
                            lines=3,
                            interactive=True
                        )
                        
                        test_incremental_btn = gr.Button("Test Incremental Translation")
                
                # Event handlers
                init_mt_btn.click(
                    fn=init_mt_model,
                    inputs=[mt_model_name, mt_source_lang, mt_target_lang, mt_use_gpu],
                    outputs=[console_log]
                )
                
                translate_btn.click(
                    fn=translate_text,
                    inputs=[mt_input_text],
                    outputs=[console_log]
                )
                
                test_incremental_btn.click(
                    fn=test_incremental_translation,
                    inputs=[mt_incremental_input],
                    outputs=[console_log]
                )
            
            # ===== TTS TAB =====
            with gr.TabItem("TTS (Text-to-Speech)"):
                gr.Markdown("## TTS Testing")
                
                with gr.Row():
                    with gr.Column():
                        tts_language = gr.Dropdown(
                            list(sorted(["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"])),
                            label="TTS Language",
                            value="en"
                        )
                        
                        tts_use_gpu = gr.Checkbox(
                            label="Use GPU if available",
                            value=True
                        )
                        
                        with gr.Row():
                            init_tts_btn = gr.Button("Initialize TTS Model")
                            init_playback_btn = gr.Button("Initialize Audio Playback")
                            stop_playback_btn = gr.Button("Stop Playback")
                        
                        tts_input_text = gr.Textbox(
                            label="Text to Speak",
                            value="This is a test of the speech synthesis system.",
                            lines=3,
                            interactive=True
                        )
                        
                        synthesize_btn = gr.Button("Synthesize Speech")
                        test_audio_btn = gr.Button("Test Complete Audio System")
                        diagnose_tts_btn = gr.Button("Diagnose TTS Output")

                
                # Event handlers
                test_audio_btn.click(
                    fn=test_complete_audio_system,
                    outputs=[console_log]
                )

                diagnose_tts_btn.click(
                fn=lambda: diagnose_tts_output("This is a test of the speech synthesis system.", "eng"),
                outputs=[console_log]
            )

                init_tts_btn.click(
                    fn=init_tts_model,
                    inputs=[tts_language, tts_use_gpu],
                    outputs=[console_log]
                )
                
                init_playback_btn.click(
                    fn=init_audio_playback,
                    outputs=[console_log]
                )
                
                stop_playback_btn.click(
                    fn=stop_audio_playback,
                    outputs=[console_log]
                )
                
                synthesize_btn.click(
                    fn=synthesize_speech,
                    inputs=[tts_input_text],
                    outputs=[console_log]
                )
            
            # ===== PIPELINE TAB =====
            with gr.TabItem("Full Pipeline"):
                gr.Markdown("## Full Pipeline Testing")
                
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            pipeline_source_lang = gr.Dropdown(
                                list(sorted(["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"])),
                                label="Source Language",
                                value="en"
                            )
                            pipeline_target_lang = gr.Dropdown(
                                list(sorted(["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi"])),
                                label="Target Language",
                                value="es"
                            )
                        
                        pipeline_use_gpu = gr.Checkbox(
                            label="Use GPU if available",
                            value=True
                        )
                        
                        with gr.Row():
                            init_pipeline_btn = gr.Button("Initialize Pipeline")
                            start_pipeline_btn = gr.Button("Start Pipeline")
                            stop_pipeline_btn = gr.Button("Stop Pipeline")
                            reset_pipeline_btn = gr.Button("Reset Pipeline")
                        
                        pipeline_file_input = gr.Textbox(
                            label="Audio File Path",
                            value="data/samples/test_en.wav",
                            interactive=True
                        )
                        
                        process_file_btn = gr.Button("Process Audio File")
                
                # Event handlers
                init_pipeline_btn.click(
                    fn=init_pipeline,
                    inputs=[pipeline_source_lang, pipeline_target_lang, pipeline_use_gpu],
                    outputs=[console_log]
                )
                
                start_pipeline_btn.click(
                    fn=start_pipeline,
                    outputs=[console_log]
                )
                
                stop_pipeline_btn.click(
                    fn=stop_pipeline,
                    outputs=[console_log]
                )
                
                reset_pipeline_btn.click(
                    fn=reset_pipeline,
                    outputs=[console_log]
                )
                
                process_file_btn.click(
                    fn=process_audio_file_with_pipeline,
                    inputs=[pipeline_file_input],
                    outputs=[console_log]
                )
            
            # ===== SYSTEM INFO TAB =====
            with gr.TabItem("System Info"):
                gr.Markdown("## System Information")
                
                with gr.Row():
                    with gr.Column():
                        system_info = gr.JSON(
                            label="System Information",
                            value=get_system_info()
                        )
                        
                        refresh_info_btn = gr.Button("Refresh System Info")
                        
                        installed_packages = gr.Textbox(
                            label="Installed Packages",
                            value=get_installed_packages(),
                            lines=20,
                            interactive=False
                        )
                
                # Event handlers
                refresh_info_btn.click(
                    fn=get_system_info,
                    outputs=[system_info]
                )
        
        # Initial log message
        log("Diagnostic tool loaded. Select a tab to test specific components.")
    
    return demo

if __name__ == "__main__":
    ui = create_ui()
    ui.launch(share=True)
    print("Diagnostic UI started. Press Ctrl+C to exit.")