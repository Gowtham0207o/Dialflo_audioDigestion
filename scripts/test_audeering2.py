import torch
from transformers import AutoModelForAudioClassification, AutoProcessor

def test():
    model_id = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    try:
        model = AutoModelForAudioClassification.from_pretrained(model_id)
        print("Model architecture:", model)
    except Exception as e:
        print("Error:", e)
    
if __name__ == "__main__":
    test()
