from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["models"] >= 3
    assert data["benchmarks"] > 0


def test_list_models():
    r = client.get("/api/models")
    assert r.status_code == 200
    data = r.json()
    assert "qwen2.5-7b" in data
    assert "llama3.1-8b" in data


def test_list_gpus():
    r = client.get("/api/gpus")
    assert r.status_code == 200
    data = r.json()
    assert "h100" in data
    assert data["h100"]["vram_gb"] == 80


def test_list_backends():
    r = client.get("/api/backends")
    assert r.status_code == 200
    data = r.json()
    assert "vllm" in data


def test_get_benchmarks():
    r = client.get("/api/benchmarks/qwen2.5-7b")
    assert r.status_code == 200
    data = r.json()
    assert len(data) >= 3


def test_get_benchmarks_404():
    r = client.get("/api/benchmarks/nonexistent")
    assert r.status_code == 404


def test_recommend_endpoint():
    r = client.post("/api/recommend", json={
        "model_id": "qwen2.5-7b",
        "workload_type": "chat",
        "optimization_priority": "cost",
        "constraints": {
            "max_p95_latency_ms": 250,
            "max_monthly_budget_usd": 1200,
        },
    })
    assert r.status_code == 200
    data = r.json()
    assert data["recommended"]["is_recommended"] is True
    assert len(data["alternatives"]) >= 1
    assert data["summary"]


def test_recommend_config_endpoint():
    r = client.post("/api/recommend/config", json={
        "model_id": "llama3.2-3b",
        "workload_type": "chat",
        "optimization_priority": "balanced",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["model_id"] == "llama3.2-3b"
    assert data["backend"] in ("vllm", "tensorrt-llm")
    assert data["gpu_count"] == 1
    assert data["max_model_len"] > 0


def test_recommend_bad_model():
    r = client.post("/api/recommend", json={
        "model_id": "fake-model",
        "workload_type": "chat",
        "optimization_priority": "cost",
    })
    assert r.status_code == 400
