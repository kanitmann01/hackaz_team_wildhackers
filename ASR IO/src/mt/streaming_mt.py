import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import time

class StreamingTranslator:
    """
    Real-time streaming machine translation using NLLB-200.
    
    This class implements incremental text translation, processing
    text chunks as they arrive from the ASR system while maintaining
    context to ensure coherent translation output.
    """
    
    def __init__(self, source_lang="eng_Latn", target_lang="spa_Latn", device=None,
                 model_name="facebook/nllb-200-distilled-600M"):
        """
        Initialize the streaming translator with NLLB-200.
        
        Args:
            source_lang: Source language code (default: "eng_Latn" for English)
            target_lang: Target language code (default: "spa_Latn" for Spanish)
            device: Computation device ('cuda' or 'cpu')
            model_name: Translation model to use
        """
        # Set device (use CUDA if available unless specified otherwise)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model and tokenizer
        print(f"Loading NLLB-200 translation model '{model_name}' on {self.device}...")
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # Configure for better streaming performance
        # Half-precision for faster processing on GPU
        if self.device.type == 'cuda':
            self.model = self.model.half()
        print("Translation model loaded successfully!")
        
        # Set languages
        self.set_languages(source_lang, target_lang)
        
        # Context tracking
        self.source_buffer = ""
        self.translated_buffer = ""
        
        # Performance tracking
        self.total_processing_time = 0
        self.chunk_count = 0
        
        # Configure for streaming
        self.min_chunk_size = 3  # Minimum number of words to trigger translation
    
    def _convert_language_code(self, iso_code):
        """
        Convert ISO language code to NLLB-200 format.
        
        Args:
            iso_code: ISO language code (e.g., "en", "es")
            
        Returns:
            str: NLLB-200 language code
        """
        # NLLB-200 uses language codes in format lang_script
        iso_to_nllb = {
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
            "hi": "hin_Deva"
        }
        
        # If already in NLLB format, return as is
        if "_" in iso_code:
            return iso_code
            
        # Convert from ISO to NLLB format
        return iso_to_nllb.get(iso_code, "eng_Latn")  # Default to English if unknown
    
    def set_languages(self, source_lang, target_lang):
        """
        Set the source and target languages for translation.
        
        Args:
            source_lang: Source language code (ISO or NLLB format)
            target_lang: Target language code (ISO or NLLB format)
        """
        # Convert language codes if needed
        self.source_lang = self._convert_language_code(source_lang)
        self.target_lang = self._convert_language_code(target_lang)
    
    def reset_context(self):
        """Reset the translation context."""
        self.source_buffer = ""
        self.translated_buffer = ""
        self.total_processing_time = 0
        self.chunk_count = 0
    
    def translate_chunk(self, text_chunk, force=False):
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
        
        # Generate translation
        with torch.no_grad():
            translated_ids = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[self.target_lang],
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
    
    def get_supported_languages(self):
        """
        Get a list of supported language pairs.
        
        Returns:
            dict: Dictionary of supported language codes and names
        """
        # NLLB-200 supports 200+ languages with higher quality than M2M-100
        return {
            "eng_Latn": "English",
            "spa_Latn": "Spanish",
            "fra_Latn": "French",
            "deu_Latn": "German",
            "ita_Latn": "Italian",
            "por_Latn": "Portuguese",
            "rus_Cyrl": "Russian",
            "zho_Hans": "Chinese (Simplified)",
            "jpn_Jpan": "Japanese",
            "kor_Hang": "Korean",
            "ara_Arab": "Arabic",
            "hin_Deva": "Hindi",
            "ben_Beng": "Bengali",
            "swh_Latn": "Swahili",
            "urd_Arab": "Urdu",
            "vie_Latn": "Vietnamese",
            "tel_Telu": "Telugu",
            "tam_Taml": "Tamil",
            "mar_Deva": "Marathi",
            "tha_Thai": "Thai"
        }
    
    def get_stats(self):
        """
        Get performance statistics.
        
        Returns:
            dict: Performance statistics
        """
        return {
            'total_chunks': self.chunk_count,
            'total_processing_time': self.total_processing_time,
            'avg_processing_time': self.total_processing_time / self.chunk_count if self.chunk_count > 0 else 0
        }