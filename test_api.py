"""
Simple test script for Real-ESRGAN API
"""
import requests
import sys

def test_api(base_url="http://localhost:8000"):
    """Test the API endpoints"""

    # Test health endpoint
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✓ Health check: {response.json()}")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

    # Test models endpoint
    print("\nTesting /models endpoint...")
    try:
        response = requests.get(f"{base_url}/models")
        print(f"✓ Available models: {response.json()}")
    except Exception as e:
        print(f"✗ Models endpoint failed: {e}")
        return False

    # Test upscale endpoint (if image provided)
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nTesting /upscale endpoint with {image_path}...")
        try:
            with open(image_path, 'rb') as f:
                files = {'file': f}
                data = {
                    'model_name': 'RealESRGAN_x4plus',
                    'scale': 4.0,
                    'face_enhance': False
                }
                response = requests.post(f"{base_url}/upscale", files=files, data=data)
                if response.status_code == 200:
                    output_path = image_path.replace('.', '_upscaled.')
                    with open(output_path, 'wb') as out:
                        out.write(response.content)
                    print(f"✓ Upscale successful! Saved to {output_path}")
                else:
                    print(f"✗ Upscale failed: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"✗ Upscale test failed: {e}")
    else:
        print("\nSkipping upscale test (no image provided)")
        print("Usage: python test_api.py <image_path>")

    return True

if __name__ == "__main__":
    base_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8000"
    test_api(base_url)

