"""Common Voice TSV Adapter.

Reads CV TSV dynamically, normalizes gender/age, and provides deterministic sampling.
"""

import csv
import logging
import random
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class CVSample:
    """A normalized Common Voice sample."""
    speaker_id: str
    audio_path: Path
    gender: str | None
    age_bracket: str | None
    skip_reason: str | None
    audio_bytes: bytes | None = None


class CommonVoiceAdapter:
    """Adapter for Mozilla Common Voice dataset."""

    AGE_MAP = {
        "teens": None,
        "twenties": "18-30",
        "thirties": "31-45",
        "fourties": "31-45",
        "fifties": "46-60",
        "sixties": "60+",
        "seventies": "60+",
        "eighties": "60+",
        "nineties": "60+",
    }

    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        if self.dataset_path.is_file():
            self.clips_dir = self.dataset_path.parent / "clips"
        else:
            self.clips_dir = self.dataset_path / "clips"

    def _find_data_file(self) -> Path:
        """Find the TSV or Parquet file to use."""
        if self.dataset_path.is_file():
            return self.dataset_path
            
        for name in ["validated.parquet", "test.parquet", "train.parquet", "validated.tsv", "test.tsv", "train.tsv"]:
            p = self.dataset_path / name
            if p.exists():
                return p
        raise FileNotFoundError(f"Could not find any TSV or Parquet file in {self.dataset_path}")

    def load_samples(self, limit: int | None = None, seed: int = 42) -> list[CVSample]:
        """Load and normalize samples from the TSV or Parquet."""
        data_path = self._find_data_file()
        
        all_samples = []
        
        if data_path.suffix == ".parquet":
            import pyarrow.parquet as pq
            table = pq.read_table(data_path)
            rows = table.to_pylist()
        else:
            with open(data_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f, delimiter="\t")
                rows = list(reader)
                
        for row in rows:
            # Detect path column
            path_col = "path" if "path" in row else "filename" if "filename" in row else None
            if not path_col or not row[path_col]:
                continue
            
            audio_bytes = None
            if "audio" in row and isinstance(row["audio"], dict) and "bytes" in row["audio"]:
                audio_bytes = row["audio"]["bytes"]
                
            raw_path = Path(row[path_col])
            
            if audio_bytes is not None:
                audio_path = Path(raw_path.name)
            else:
                audio_path = self.clips_dir / raw_path.name
                # Ensure .mp3 extension if missing
                if not audio_path.suffix:
                    audio_path = audio_path.with_suffix(".mp3")
                    
            client_id = row.get("client_id", "")
            raw_gender = row.get("gender", "").lower().strip()
            raw_age = row.get("age", "").lower().strip()
            
            # Normalize gender
            if raw_gender in ("male", "female"):
                gender = raw_gender
                skip_reason = None
            else:
                gender = None
                skip_reason = f"Unsupported gender: {raw_gender}"
            
            # Normalize age
            age_bracket = None
            if raw_age:
                age_bracket = self.AGE_MAP.get(raw_age)
                if not age_bracket and raw_age == "teens":
                    skip_reason = "Excluded age: teens"
            
            # If neither age nor gender is usable, and we haven't already set a skip reason, skip it
            if not gender and not age_bracket and not skip_reason:
                 skip_reason = "No usable age or gender"

            all_samples.append(CVSample(
                speaker_id=client_id,
                audio_path=audio_path,
                gender=gender,
                age_bracket=age_bracket,
                skip_reason=skip_reason,
                audio_bytes=audio_bytes,
            ))

        if limit and limit < len(all_samples):
            random.seed(seed)
            all_samples = random.sample(all_samples, limit)

        # Reporting
        total = len(all_samples)
        usable = sum(1 for s in all_samples if not s.skip_reason)
        skipped = total - usable
        
        skip_reasons = {}
        for s in all_samples:
            if s.skip_reason:
                skip_reasons[s.skip_reason] = skip_reasons.get(s.skip_reason, 0) + 1
                
        logger.info(f"Loaded {total} samples. Usable: {usable}, Skipped: {skipped}")
        for r, c in skip_reasons.items():
             logger.info(f"  Skipped reason '{r}': {c}")

        return all_samples
