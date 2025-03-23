import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Use this function to test MT in isolation
def test_mt_directly():
    print("Testing MT component directly...")
    
    # Set device
    device = torch.device("cpu")  # Start with CPU for simplicity
    
    # Model name
    model_name = "facebook/nllb-200-distilled-600M"
    
    # Languages
    source_lang = "eng_Latn"
    target_lang = "spa_Latn"
    
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    
    # Test text
    text = "This is a test of the translation system."
    print(f"Translating: '{text}'")
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Print available functions in tokenizer to debug
    print("Available tokenizer functions:", dir(tokenizer))
    
    # Check if lang_code_to_id is a method or attribute
    if hasattr(tokenizer, 'lang_code_to_id'):
        if callable(tokenizer.lang_code_to_id):
            print("lang_code_to_id is a method")
            target_lang_id = tokenizer.lang_code_to_id(target_lang)
        else:
            print("lang_code_to_id is an attribute")
            target_lang_id = tokenizer.lang_code_to_id.get(target_lang)
    else:
        print("lang_code_to_id not found. Trying alternative approach.")
        # Alternative approach
        target_lang_id = tokenizer.convert_tokens_to_ids(target_lang)
    
    print(f"Target language ID: {target_lang_id}")
    
    # Generate translation
    with torch.no_grad():
        try:
            translated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=target_lang_id
            )
            
            # Decode translation
            translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            print(f"Translation result: '{translation}'")
            
        except Exception as e:
            print(f"Error during translation: {e}")
            print("Trying alternative method...")
            
            # Try directly
            translated_tokens = model.generate(
                **inputs,
                decoder_start_token_id=target_lang_id
            )
            
            # Decode translation
            translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            print(f"Translation result (alternative method): '{translation}'")

if __name__ == "__main__":
    test_mt_directly()