import json
from pathlib import Path

from app.schemas.benchmark import BenchmarkProfile


_PROFILES_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "profiles"


class BenchmarkStore:
    """Loads benchmark profiles from JSON files and serves them up for queries."""

    def __init__(self, profiles_dir: Path | None = None):
        self._dir = profiles_dir or _PROFILES_DIR
        self._profiles: list[BenchmarkProfile] = []
        self._load()

    def _load(self) -> None:
        self._profiles = []
        if not self._dir.exists():
            return
        for fp in sorted(self._dir.rglob("*.json")):
            with open(fp) as f:
                data = json.load(f)
            # each file is a list of profiles for one model
            for entry in data:
                self._profiles.append(BenchmarkProfile(**entry))

    def reload(self) -> None:
        self._load()

    @property
    def all_profiles(self) -> list[BenchmarkProfile]:
        return list(self._profiles)

    def query(
        self,
        model_id: str | None = None,
        gpu_tier: str | None = None,
        backend: str | None = None,
        quantization: str | None = None,
    ) -> list[BenchmarkProfile]:
        results = self._profiles
        if model_id:
            results = [p for p in results if p.model_id == model_id]
        if gpu_tier:
            results = [p for p in results if p.gpu_tier == gpu_tier]
        if backend:
            results = [p for p in results if p.backend == backend]
        if quantization:
            results = [p for p in results if p.quantization == quantization]
        return results

    def get_best_for_model(self, model_id: str) -> list[BenchmarkProfile]:
        """All benchmark entries for a given model, sorted by tokens/sec descending."""
        profiles = self.query(model_id=model_id)
        return sorted(profiles, key=lambda p: p.tokens_per_sec, reverse=True)
