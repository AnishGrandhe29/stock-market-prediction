import sys
from fastapi.testclient import TestClient
sys.path.insert(0, '.')

from app.main import app

client = TestClient(app)

def test_generate_endpoint():
    print("Testing /latest endpoint...")
    response = client.get("/api/v1/predictions/latest")
    print(f"/latest Status: {response.status_code}")

if __name__ == "__main__":
    test_generate_endpoint()
