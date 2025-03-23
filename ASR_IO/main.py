import os
import sys
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "."))

def main():
    """
    Main entry point for the Real-Time P2P Audio Translation application.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Real-Time P2P Audio Translation")
    parser.add_argument(
        "--mode", 
        choices=["ui", "demo", "test"], 
        default="ui",
        help="Application mode (ui, demo, or test)"
    )
    parser.add_argument(
        "--source-lang", 
        default="en",
        help="Source language code (e.g., 'en', 'es', 'fr')"
    )
    parser.add_argument(
        "--target-lang", 
        default="es",
        help="Target language code (e.g., 'en', 'es', 'fr')"
    )
    parser.add_argument(
        "--no-gpu", 
        action="store_true",
        help="Disable GPU usage even if available"
    )
    parser.add_argument(
        "--share", 
        action="store_true",
        help="Share the UI through a public URL"
    )
    
    args = parser.parse_args()
    
    # Force CPU if requested
    if args.no_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Run in selected mode
    if args.mode == "ui":
        # Run the UI
        from src.ui.app import create_ui
        ui = create_ui()
        ui.launch(share=args.share)
        print("UI started. Press Ctrl+C to exit.")
        
    elif args.mode == "demo":
        # Run the demo mode
        from src.pipeline.realtime_pipeline import RealTimeTranslationPipeline
        from src.asr.streaming_asr import StreamingASR
        from src.mt.streaming_mt import StreamingTranslator
        from src.tts.streaming_tts import StreamingTTS
        from src.audio.streaming_audio import StreamingAudioCapture, StreamingAudioPlayback
        from src.ui.language_utils import LanguageMapper
        import torch
        import time
        
        print(f"Starting demo mode with {args.source_lang} -> {args.target_lang}")
        
        # Initialize components
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_gpu else "cpu")
        print(f"Using device: {device}")
        
        # Get appropriate model names for languages
        wav2vec_model = LanguageMapper.iso_to_wav2vec(args.source_lang)
        source_nllb = LanguageMapper.iso_to_nllb(args.source_lang)
        target_nllb = LanguageMapper.iso_to_nllb(args.target_lang)
        target_mms_code, target_mms_model = LanguageMapper.iso_to_mms(args.target_lang)
        
        print(f"Loading ASR model: {wav2vec_model}")
        asr_model = StreamingASR(model_name=wav2vec_model, device=device, language=args.source_lang)
        
        print(f"Loading MT model with {source_nllb} -> {target_nllb}")
        mt_model = StreamingTranslator(
            source_lang=source_nllb, 
            target_lang=target_nllb, 
            device=device
        )
        
        print(f"Loading TTS model: {target_mms_model}")
        tts_model = StreamingTTS(device=device, model_name=target_mms_model)
        tts_model.set_language(target_mms_code)
        
        # Create pipeline
        audio_playback = StreamingAudioPlayback()
        
        def demo_callback(source_text, translated_text):
            """Print updates to console."""
            print(f"\nSource: {source_text}")
            print(f"Translation: {translated_text}")
            print("-" * 40)
        
        audio_capture = StreamingAudioCapture(callback=lambda x, y: None)
        
        pipeline = RealTimeTranslationPipeline(
            asr_model=asr_model,
            mt_model=mt_model,
            tts_model=tts_model,
            audio_capture=audio_capture,
            audio_playback=audio_playback
        )
        
        # Set callback
        pipeline.set_update_callback(demo_callback)
        
        # Start pipeline
        pipeline.start()
        
        print("\nDemo mode started. Speak into your microphone.")
        print("Press Ctrl+C to exit.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping demo...")
            pipeline.stop()
            print("Demo stopped.")
        
    elif args.mode == "test":
        # Run the test mode
        from scripts.run_tests import run_tests
        run_tests()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())