"""Tests for Common Voice Adapter."""

import csv
import tempfile
from pathlib import Path

import pytest
from evaluation.adapter.common_voice import CommonVoiceAdapter


@pytest.fixture
def dummy_dataset_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        clips_dir = tmp_path / "clips"
        clips_dir.mkdir()
        
        # Create some dummy mp3 files
        (clips_dir / "sample1.mp3").touch()
        (clips_dir / "sample2.mp3").touch()
        (clips_dir / "sample3.mp3").touch()
        (clips_dir / "sample4.mp3").touch()
        
        tsv_path = tmp_path / "test.tsv"
        with open(tsv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["client_id", "path", "sentence", "up_votes", "down_votes", "age", "gender", "accents", "locale"])
            writer.writerow(["client_A", "sample1.mp3", "test", "1", "0", "twenties", "male", "", "en"])
            writer.writerow(["client_B", "sample2.mp3", "test", "1", "0", "forties", "female", "", "en"])
            writer.writerow(["client_C", "sample3.mp3", "test", "1", "0", "teens", "male", "", "en"])
            writer.writerow(["client_D", "sample4.mp3", "test", "1", "0", "unknown", "unknown_gender", "", "en"])
            
        yield tmp_path


def test_cv_adapter_loads_samples(dummy_dataset_dir):
    adapter = CommonVoiceAdapter(dummy_dataset_dir)
    samples = adapter.load_samples()
    
    assert len(samples) == 4
    
    # Sample 1: Valid age and gender
    assert samples[0].speaker_id == "client_A"
    assert samples[0].gender == "male"
    assert samples[0].age_bracket == "18-30"
    assert samples[0].skip_reason is None
    
    # Sample 2: Valid age and gender ("forties" -> "31-45")
    assert samples[1].gender == "female"
    assert samples[1].age_bracket == "31-45"
    assert samples[1].skip_reason is None
    
    # Sample 3: Skipped age ("teens") but usable gender
    assert samples[2].gender == "male"
    assert samples[2].age_bracket is None
    assert samples[2].skip_reason == "Excluded age: teens"
    
    # Sample 4: Unusable
    assert samples[3].gender is None
    assert samples[3].age_bracket is None
    assert samples[3].skip_reason == "Unsupported gender: unknown_gender"


def test_cv_adapter_limit_and_seed(dummy_dataset_dir):
    adapter = CommonVoiceAdapter(dummy_dataset_dir)
    samples1 = adapter.load_samples(limit=2, seed=42)
    assert len(samples1) == 2
    
    samples2 = adapter.load_samples(limit=2, seed=42)
    assert len(samples2) == 2
    
    assert [s.speaker_id for s in samples1] == [s.speaker_id for s in samples2]
