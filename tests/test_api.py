from fastapi.testclient import TestClient

from psl_workflow.api import create_app


def test_api_health_endpoint(tmp_path) -> None:
    client = TestClient(create_app(data_dir=tmp_path))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
