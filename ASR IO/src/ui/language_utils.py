"""
Language utility functions for the real-time translation system.
This module handles language code mappings and conversions between different models.
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

# Language code mappings for different models
class LanguageMapper:
    """
    Handles language code mappings between different model formats.
    """
    
    @staticmethod
    def get_supported_languages():
        """Get list of supported languages with ISO codes."""
        return ISO_LANGUAGES
    
    @staticmethod
    def iso_to_wav2vec(iso_code):
        """
        Convert ISO code to appropriate Wav2Vec 2.0 model name.
        
        Args:
            iso_code: ISO language code (e.g., "en")
            
        Returns:
            str: Wav2Vec model name for the language
        """
        wav2vec_models = {
            "en": "facebook/wav2vec2-large-960h-lv60-self",
            # Use XLSR for non-English languages
            "es": "facebook/wav2vec2-large-xlsr-53",
            "fr": "facebook/wav2vec2-large-xlsr-53",
            "de": "facebook/wav2vec2-large-xlsr-53",
            "it": "facebook/wav2vec2-large-xlsr-53",
            "pt": "facebook/wav2vec2-large-xlsr-53",
            "ru": "facebook/wav2vec2-large-xlsr-53",
            "zh": "facebook/wav2vec2-large-xlsr-53",
            "ja": "facebook/wav2vec2-large-xlsr-53",
            "ar": "facebook/wav2vec2-large-xlsr-53",
            "hi": "facebook/wav2vec2-large-xlsr-53"
        }
        
        return wav2vec_models.get(iso_code, "facebook/wav2vec2-large-xlsr-53")
    
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
            
        return nllb_codes.get(iso_code, "eng_Latn")  # Default to English
    
    @staticmethod
    def iso_to_mms(iso_code):
        """
        Convert ISO code to MMS-TTS language code.
        
        Args:
            iso_code: ISO language code (e.g., "en")
            
        Returns:
            str: MMS-TTS language code and model name
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
            "ar": ("ara", "facebook/mms-tts-ara"),
            "hi": ("hin", "facebook/mms-tts-hin")
        }
        
        return mms_codes.get(iso_code, ("eng", "facebook/mms-tts-eng"))  # Default to English