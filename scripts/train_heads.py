import argparse
import logging
import sys
import os
from pathlib import Path
import numpy as np

# Add project root to sys.path so we can import app and evaluation modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from evaluation.adapter.common_voice import CommonVoiceAdapter
from app.audio.codec import AudioCodec
from app.audio.vad import VoiceActivityDetector
from app.audio.quality import AudioQualityAssessor
from app.audio.preprocessor import AudioPreprocessor
from app.inference.speech_encoder import SpeechEncoder
from app.inference.strategies.gender_classifier import GenderNet
from app.inference.strategies.age_estimator import AgeNet

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GENDER_MAP = {"male": 0, "female": 1}
AGE_MAP = {"18-30": 0, "31-45": 1, "46-60": 2, "60+": 3}

def extract_features(dataset_path: str, limit: int = None):
    dataset_path = Path(dataset_path)
    adapter = CommonVoiceAdapter(dataset_path)
    samples = adapter.load_samples(limit=limit)
    
    vad_detector = VoiceActivityDetector()
    quality_assessor = AudioQualityAssessor()
    preprocessor = AudioPreprocessor()
    
    encoder = SpeechEncoder()
    encoder.load()
    
    embeddings = []
    gender_labels = []
    age_labels = []
    
    logger.info("Extracting features (this may take a while)...")
    for idx, sample in enumerate(samples):
        if idx > 0 and idx % 100 == 0:
            logger.info(f"Processed {idx}/{len(samples)} samples...")
            
        try:
            if sample.audio_bytes is not None:
                audio_bytes = sample.audio_bytes
            else:
                audio_bytes = sample.audio_path.read_bytes()
                
            segment = AudioCodec.transcode_to_wav(audio_bytes)
            vad_res = vad_detector.detect(segment.waveform, segment.sample_rate)
            qual_res = quality_assessor.assess(segment.waveform, vad_res, segment.sample_rate)
            prep_input = preprocessor.prepare(segment, vad_res)
            
            if not prep_input.is_prepared_valid:
                continue
                
            emb_res = encoder.encode(prep_input)
            if not emb_res.is_valid:
                continue
                
            # Only use samples that have a valid gender/age ground truth for training
            if sample.gender in GENDER_MAP and sample.age_bracket in AGE_MAP:
                embeddings.append(emb_res.embedding)
                gender_labels.append(GENDER_MAP[sample.gender])
                age_labels.append(AGE_MAP[sample.age_bracket])
                
        except Exception as e:
            pass # ignore failures for training
            
    logger.info(f"Successfully extracted {len(embeddings)} valid training samples.")
    
    if not embeddings:
        logger.error("No valid samples extracted. Cannot train.")
        sys.exit(1)
        
    return torch.tensor(np.array(embeddings), dtype=torch.float32), torch.tensor(gender_labels, dtype=torch.long), torch.tensor(age_labels, dtype=torch.long)

def train_head(model, X_train, y_train, X_val, y_val, model_path: Path, epochs=50, lr=0.001, class_weights=None):
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32)
        criterion = nn.NLLLoss(weight=class_weights)
    else:
        criterion = nn.NLLLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    best_val_loss = float("inf")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            # Softmax is applied inside the model's forward pass
            # We use NLLLoss with log probabilities
            log_probs = torch.log(outputs + 1e-8)
            loss = criterion(log_probs, batch_y)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        scheduler.step()
            
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_log_probs = torch.log(val_outputs + 1e-8)
            val_loss = criterion(val_log_probs, y_val).item()
            
            preds = torch.argmax(val_outputs, dim=1)
            acc = (preds == y_val).float().mean().item()
            
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), model_path)
            
    logger.info(f"Finished training. Best model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser(description="Train custom classification heads")
    parser.add_argument("--dataset-path", required=True, type=str, help="Path to CV dataset")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    args = parser.parse_args()
    
    X, y_gender, y_age = extract_features(args.dataset_path, limit=args.limit)
    
    # Split Data
    X_train, X_val, y_gen_train, y_gen_val, y_age_train, y_age_val = train_test_split(
        X, y_gender, y_age, test_size=0.2, random_state=42
    )
    
    # Compute Class Weights to handle imbalance
    gen_weights = compute_class_weight('balanced', classes=np.unique(y_gen_train), y=y_gen_train.numpy())
    age_weights = compute_class_weight('balanced', classes=np.unique(y_age_train), y=y_age_train.numpy())
    
    logger.info("--- Training GenderNet ---")
    gender_model = GenderNet(embedding_dim=192)
    gender_path = Path("models/custom_heads/gender_head.pt")
    train_head(gender_model, X_train, y_gen_train, X_val, y_gen_val, gender_path, epochs=args.epochs, class_weights=gen_weights)
    
    logger.info("--- Training AgeNet ---")
    age_model = AgeNet(embedding_dim=192)
    age_path = Path("models/custom_heads/age_head.pt")
    train_head(age_model, X_train, y_age_train, X_val, y_age_val, age_path, epochs=args.epochs, class_weights=age_weights)
    
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
