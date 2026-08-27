import torch
from transformers import AutoModelForAudioClassification, AutoProcessor

def test():
    print("Loading pipeline...")
    
    model_id = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForAudioClassification.from_pretrained(model_id)
        
        # create dummy audio
        audio = torch.randn(16000 * 3) # 3 seconds
        
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        print("Logits shape:", outputs.logits.shape)
        print("Logits:", outputs.logits)
    except Exception as e:
        print("Error:", e)
    
if __name__ == "__main__":
    test()
