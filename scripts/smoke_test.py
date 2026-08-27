import sys
import httpx
from pathlib import Path
import json

def test_api():
    url = "http://localhost:8000/analyze"
    audio_file = Path("tests/fixtures/audio/sample_clean.wav")
    
    if not audio_file.exists():
        print(f"Error: Could not find test audio file at {audio_file}")
        sys.exit(1)
        
    print(f"Testing {url} with {audio_file}...")
    
    with open(audio_file, "rb") as f:
        files = {"file": ("sample_16k_clean.wav", f, "audio/wav")}
        
        try:
            response = httpx.post(url, files=files, timeout=10.0)
            response.raise_for_status()
            
            data = response.json()
            print("Success! Response JSON:")
            print(json.dumps(data, indent=2))
            
            # Verify exact JSON contract
            assert "contact_id" in data
            assert "gender" in data
            assert "age_bracket" in data
            assert "processing_ms" in data
            assert "audio_quality" in data
            
            assert "prediction" in data["gender"]
            assert "confidence" in data["gender"]
            
            print("JSON contract verified successfully.")
            
        except Exception as e:
            print(f"Test failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    test_api()
