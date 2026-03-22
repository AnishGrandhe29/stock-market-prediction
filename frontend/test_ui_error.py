import urllib.request
import traceback

def get_error():
    print("Fetching http://localhost:3000")
    try:
        # Requesting the page
        req = urllib.request.Request("http://localhost:3000")
        with urllib.request.urlopen(req) as response:
            html = response.read().decode('utf-8')
            print(f"Status Code: {response.status}")
            print(html[:1000])
    except urllib.error.HTTPError as e:
        print(f"HTTPError: {e.code}")
        html = e.read().decode('utf-8')
        print(html[:2000])
        # Try to find Next.js error
        if "Error:" in html:
            start = html.find("Error:")
            print("\nPOSSIBLE ERROR TRACE:\n")
            print(html[start:start+1000])
    except Exception as e:
        print(f"Failed to connect: {e}")

if __name__ == "__main__":
    get_error()
