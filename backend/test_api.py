import requests

def test_api():
    try:
        response = requests.post("http://localhost:8000/api/v1/predictions", json={"symbol": "^NSEI"})
        print(f"Status: {response.status_code}")
        print(response.json())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
