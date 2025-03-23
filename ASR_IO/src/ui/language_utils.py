"""
Simplified language utility functions for the real-time translation system.
Focuses on Whisper ASR model and provides mappings for NLLB and MMS-TTS.
"""

# ISO language codes to human-readable names
ISO_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "bn": "Bengali",
    "sw": "Swahili",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "te": "Telugu",
    "ta": "Tamil",
    "mr": "Marathi",
    "th": "Thai"
}

class LanguageMapper:
    """
    Simplified language code mappings optimized for Whisper.
    """
    
    @staticmethod
    def get_supported_languages():
        """Get list of supported languages with ISO codes."""
        return ISO_LANGUAGES
    
    @staticmethod
    def get_whisper_language(iso_code):
        """
        Get Whisper language code from ISO code.
        Whisper uses the same codes as ISO 639-1 in most cases.
        
        Args:
            iso_code: ISO language code (e.g., "en")
            
        Returns:
            str: Whisper language code
        """
        # Whisper uses ISO 639-1 codes directly in most cases
        return iso_code.lower()
    
    @staticmethod
    def iso_to_nllb(iso_code):
        """
        Convert ISO code to NLLB-200 language code.
        
        Args:
            iso_code: ISO language code (e.g., "en")
            
        Returns:
            str: NLLB-200 language code
        """
        nllb_codes = {
            "en": "eng_Latn",
            "es": "spa_Latn",
            "fr": "fra_Latn",
            "de": "deu_Latn",
            "it": "ita_Latn",
            "pt": "por_Latn",
            "ru": "rus_Cyrl",
            "zh": "zho_Hans",
            "ja": "jpn_Jpan",
            "ko": "kor_Hang",
            "ar": "ara_Arab",
            "hi": "hin_Deva",
            "bn": "ben_Beng",
            "sw": "swh_Latn",
            "ur": "urd_Arab",
            "vi": "vie_Latn",
            "te": "tel_Telu",
            "ta": "tam_Taml",
            "mr": "mar_Deva",
            "th": "tha_Thai"
        }
        
        # If already in NLLB format (contains underscore), return as is
        if "_" in iso_code:
            return iso_code
            
        # Always return a valid code, defaulting to English for unknown languages
        return nllb_codes.get(iso_code.lower(), "eng_Latn")
    
    @staticmethod
    def iso_to_mms(iso_code):
        """
        Convert ISO code to MMS-TTS language code.
        
        Args:
            iso_code: ISO language code (e.g., "en")
            
        Returns:
            tuple: (MMS-TTS language code, model name)
        """
        mms_codes = {
            "en": ("eng", "facebook/mms-tts-eng"),
            "es": ("spa", "facebook/mms-tts-spa"),
            "fr": ("fra", "facebook/mms-tts-fra"),
            "de": ("deu", "facebook/mms-tts-deu"),
            "it": ("ita", "facebook/mms-tts-ita"),
            "pt": ("por", "facebook/mms-tts-por"),
            "ru": ("rus", "facebook/mms-tts-rus"),
            "zh": ("cmn", "facebook/mms-tts-cmn"),
            "ja": ("jpn", "facebook/mms-tts-jpn"),
            "ko": ("kor", "facebook/mms-tts-eng"),  # Use English if no Korean model
            "ar": ("ara", "facebook/mms-tts-ara"),
            "hi": ("hin", "facebook/mms-tts-hin"),
            "bn": ("eng", "facebook/mms-tts-eng"),  # Use English for Bengali if no Bengali model
            "sw": ("eng", "facebook/mms-tts-eng"),  # Default to English for other languages
            "ur": ("eng", "facebook/mms-tts-eng"),
            "vi": ("eng", "facebook/mms-tts-eng"),
            "te": ("eng", "facebook/mms-tts-eng"),
            "ta": ("eng", "facebook/mms-tts-eng"),
            "mr": ("eng", "facebook/mms-tts-eng"),
            "th": ("eng", "facebook/mms-tts-eng")
        }
        
        # Always return a valid tuple, defaulting to English for unknown languages
        return mms_codes.get(iso_code.lower(), ("eng", "facebook/mms-tts-eng"))