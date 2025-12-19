from fastapi.testclient import TestClient
from server.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_index():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_agents_list():
    response = client.get("/agents")
    assert response.status_code == 200
    data = response.json()
    assert "agents" in data
    assert any(a.get("name") == "workforce" for a in data.get("agents") or [])


def test_create_office_files():
    for kind, ext in (("word", ".docx"), ("excel", ".xlsx"), ("powerpoint", ".pptx")):
        resp = client.post("/files/create", json={"kind": kind})
        assert resp.status_code == 200
        cd = resp.headers.get("content-disposition", "")
        assert ext in cd
        assert len(resp.content) > 0
