import gradio as gr
import torch
import time
import threading
import os
import sys
import numpy as np
import traceback

# Add parent directory to path to allow for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.asr.streaming_asr import StreamingASR
from src.mt.streaming_mt import StreamingTranslator
from src.tts.streaming_tts import StreamingTTS
from src.audio.streaming_audio import StreamingAudioCapture, StreamingAudioPlayback
from src.pipeline.realtime_pipeline import RealTimeTranslationPipeline
from src.ui.language_utils import LanguageMapper, ISO_LANGUAGES

# Define available languages
LANGUAGES = LanguageMapper.get_supported_languages()

# Global variables
pipeline = None
source_text_global = ""
translated_text_global = ""
input_level_global = 0
output_level_global = 0
stats_global = {}
update_ui_event = threading.Event()
running_global = False
warning_shown = False

def initialize_components():
    """Initialize all system components."""
    global pipeline, warning_shown
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not warning_shown:
        print("WARNING: CUDA not available. Using CPU for processing, which may be slow.")
        warning_shown = True
    
    try:
        # Create components with Whisper (ASR), NLLB (MT), and VITS (TTS)
        print("Initializing Whisper ASR model...")
        from src.asr.whisper_asr import WhisperASR
        asr_model = WhisperASR(model_name="tiny", device=device)
        
        print("Initializing NLLB-200 MT model...")
        mt_model_name = "facebook/nllb-200-distilled-600M"
        print(f"Using MT model: {mt_model_name}")
        mt_model = StreamingTranslator(
            source_lang="eng_Latn", 
            target_lang="spa_Latn", 
            device=device, 
            model_name=mt_model_name
        )
        
        print("Initializing VITS TTS model...")
        tts_model_name = "facebook/mms-tts-eng"
        print(f"Using TTS model: {tts_model_name}")
        tts_model = StreamingTTS(device=device, model_name=tts_model_name)
        
        print("Initializing audio components...")
        audio_playback = StreamingAudioPlayback()
        
        # Increase chunk duration for better recognition
        audio_capture = StreamingAudioCapture(
            callback=lambda x, y: None,  # Placeholder callback
            chunk_duration=2.0,  # Longer chunks for better speech recognition with Whisper
            overlap=0.2  # Higher overlap for better continuity
        )
        
        # Create pipeline
        print("Creating pipeline...")
        pipeline = RealTimeTranslationPipeline(
            asr_model=asr_model,
            mt_model=mt_model,
            tts_model=tts_model,
            audio_capture=audio_capture,
            audio_playback=audio_playback
        )
        
        # Set callback for UI updates
        pipeline.set_update_callback(ui_update_callback)
        
        print("System initialized successfully")
        
    except Exception as e:
        import traceback
        print(f"Error initializing components: {e}")
        print(traceback.format_exc())
        raise

def ui_update_callback(source_text, translated_text):
    """Callback function for pipeline updates."""
    global source_text_global, translated_text_global, update_ui_event
    
    print(f"UI Update: Source: '{source_text}', Translation: '{translated_text}'")
    source_text_global = source_text
    translated_text_global = translated_text
    
    # Signal UI update
    update_ui_event.set()

def start_translation(source_lang, target_lang):
    """Start the translation pipeline."""
    global pipeline, running_global
    
    if running_global:
        return "Translation already running. Stop it first."
    
    try:
        # Convert language names to ISO codes
        source_iso = None
        target_iso = None
        
        print(f"Looking up ISO codes for {source_lang} and {target_lang}")
        
        # Find ISO codes based on language name
        for iso, name in ISO_LANGUAGES.items():
            if name.lower() == source_lang.lower():
                source_iso = iso
            if name.lower() == target_lang.lower():
                target_iso = iso
        
        print(f"Found ISO codes: {source_iso} and {target_iso}")
        
        # If ISO codes not found, return error
        if source_iso is None or target_iso is None:
            return f"❌ Error: Could not find ISO codes for {source_lang} or {target_lang}"
        
        # Create pipeline if not exists
        if pipeline is None:
            print("Initializing components...")
            initialize_components()
        
        # Get model-specific language codes
        print(f"Getting NLLB codes for {source_iso}, {target_iso}")
        source_nllb = LanguageMapper.iso_to_nllb(source_iso)
        target_nllb = LanguageMapper.iso_to_nllb(target_iso)
        
        print(f"Getting MMS code for {target_iso}")
        target_mms_code, target_mms_model = LanguageMapper.iso_to_mms(target_iso)
        
        print(f"NLLB codes: {source_nllb} -> {target_nllb}")
        print(f"MMS-TTS code: {target_mms_code}, model: {target_mms_model}")
        
        # Set languages
        print("Setting languages in pipeline...")
        print(f"Setting pipeline languages to {source_iso} -> {target_iso}")
        pipeline.set_languages(source_iso, target_iso)
        
        # Update TTS language explicitly
        print(f"Setting TTS language to {target_mms_code}")
        pipeline.tts.set_language(target_mms_code)
        
        # Reset and start
        print("Resetting pipeline...")
        pipeline.reset()
        
        print("Starting pipeline...")
        pipeline.start()
        
        running_global = True
        
        return f"✅ Started translating from {source_lang} to {target_lang}"
        
    except Exception as e:
        error_msg = f"❌ Error starting translation: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return error_msg

def stop_translation():
    """Stop the translation pipeline."""
    global pipeline, running_global
    
    if not running_global:
        return "Translation not running."
    
    try:
        if pipeline:
            pipeline.stop()
        
        running_global = False
        
        return "✅ Translation stopped"
        
    except Exception as e:
        error_msg = f"❌ Error stopping translation: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return error_msg

def get_updates():
    """Get the latest text updates for UI."""
    global source_text_global, translated_text_global, update_ui_event
    
    # Clear the event
    update_ui_event.clear()
    
    return source_text_global, translated_text_global

def get_audio_levels():
    """Get the current audio input/output levels for UI."""
    global pipeline, input_level_global, output_level_global
    
    if pipeline and running_global:
        try:
            # Get input level from audio capture
            current, peak = pipeline.audio_capture.get_audio_level()
            input_level_global = current
            
            # Output level is simulated for now
            output_level_global = max(0, output_level_global - 0.05)
            if len(translated_text_global) > len(source_text_global) - 10:
                output_level_global = min(0.8, output_level_global + 0.2)
        except Exception as e:
            print(f"Error getting audio levels: {e}")
    
    return input_level_global, output_level_global

def get_stats():
    """Get the latest performance statistics."""
    global pipeline, stats_global
    
    if pipeline and running_global:
        try:
            stats_global = pipeline.get_stats()
            
            # Format for display
            formatted_stats = f"""
**System Statistics:**
- Total audio chunks processed: {stats_global.get('audio_chunks_processed', 0)}
- ASR chunks processed: {stats_global.get('asr_chunks_processed', 0)}
- MT chunks processed: {stats_global.get('mt_chunks_processed', 0)}
- TTS chunks processed: {stats_global.get('tts_chunks_processed', 0)}
- Average latency: {stats_global.get('avg_latency', 0):.2f}s
- Uptime: {stats_global.get('uptime', 0):.1f}s
            """
            
            return formatted_stats
        except Exception as e:
            print(f"Error getting stats: {e}")
    
    return "Statistics not available"

def create_ui():
    """Create the Gradio UI."""
    with gr.Blocks(title="Real-Time P2P Audio Translation") as demo:
        gr.Markdown("# Real-Time P2P Audio Translation")
        gr.Markdown("#### Translate speech between languages in real-time with minimal latency")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Controls")
                
                # Convert the language dictionary to a list of display names for dropdowns
                language_names = list(LANGUAGES.values())
                
                source_lang = gr.Dropdown(
                    language_names, 
                    label="Source Language",
                    value="English"
                )
                
                target_lang = gr.Dropdown(
                    language_names, 
                    label="Target Language",
                    value="Spanish"
                )
                
                with gr.Row():
                    start_btn = gr.Button("▶️ Start Translation", variant="primary")
                    stop_btn = gr.Button("⏹️ Stop Translation", variant="stop")
                
                status = gr.Textbox(label="Status", value="Ready")
                
                gr.Markdown("### Performance")
                performance_stats = gr.Markdown("Statistics will appear when translation starts")
                
            with gr.Column(scale=2):
                gr.Markdown("### Live Translation")
                
                with gr.Row():
                    input_level = gr.Slider(
                        minimum=0, 
                        maximum=1, 
                        value=0, 
                        label="Input Audio Level",
                        interactive=False
                    )
                    
                    output_level = gr.Slider(
                        minimum=0, 
                        maximum=1, 
                        value=0, 
                        label="Output Audio Level",
                        interactive=False
                    )
                
                with gr.Row():
                    transcription = gr.Textbox(
                        label="Speech Recognition (Source)",
                        placeholder="Your speech will appear here...",
                        lines=5
                    )
                    
                    translation = gr.Textbox(
                        label="Translation (Target)",
                        placeholder="Translation will appear here...",
                        lines=5
                    )
                
                gr.Markdown("""
                ### Instructions
                
                1. Select source and target languages
                2. Click "Start Translation" 
                3. Speak into your microphone
                4. The translation will be spoken out loud and displayed above
                5. Click "Stop Translation" when done
                
                **Note:** First-time startup may take a few moments as models are downloaded.
                """)
                
                # Add information about the models
                gr.Markdown("""
                ### About the Technology
                
                This system uses state-of-the-art AI models:
                
                - **ASR**: Wav2Vec 2.0 for high-quality speech recognition
                - **MT**: NLLB-200 for accurate translation in 200+ languages
                - **TTS**: VITS (MMS-TTS) for natural-sounding speech synthesis
                
                All components operate in a streaming fashion for real-time performance.
                """)
        
        # Set up event handlers
        start_btn.click(
            fn=start_translation,
            inputs=[source_lang, target_lang],
            outputs=status
        )
        
        stop_btn.click(
            fn=stop_translation,
            inputs=[],
            outputs=status
        )
        
        # Periodic UI updates
        demo.load(lambda: ("", ""))
        
        # Status updates
        demo.load(lambda: "Ready", None, status, every=10)
        
        # Audio level updates
        demo.load(
            fn=get_audio_levels,
            inputs=None,
            outputs=[input_level, output_level],
            every=0.1
        )
        
        # Text updates
        demo.load(
            fn=get_updates,
            inputs=None,
            outputs=[transcription, translation],
            every=0.2
        )
        
        # Stats updates
        demo.load(
            fn=get_stats,
            inputs=None,
            outputs=performance_stats,
            every=1.0
        )
        
    return demo

if __name__ == "__main__":
    ui = create_ui()
    ui.launch(share=True)
    print("UI started. Press Ctrl+C to exit.")