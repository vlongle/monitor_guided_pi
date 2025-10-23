#!/usr/bin/env python3
"""
Test script for Qwen VLM Service
Run this to test if the service is working correctly
"""

import requests
import base64
import json
from PIL import Image
import io
import numpy as np

def test_qwen_service():
    """Test the Qwen VLM service"""
    
    # service_url = "http://10.218.137.34:8766"
    service_url = "http://localhost:8766"
    
    # Test health endpoint
    print("🧪 Testing Qwen VLM Service...")
    print("=" * 40)
    
    try:
        # Health check
        print("1. Testing health endpoint...")
        response = requests.get(f"{service_url}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test status endpoint
    try:
        print("\n2. Testing status endpoint...")
        response = requests.get(f"{service_url}/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Status check passed: {data}")
        else:
            print(f"❌ Status check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Status check error: {e}")
        return False
    
    # Test image processing with a simple test image
    try:
        print("\n3. Testing image processing...")
        
        # Create a simple test image
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image[:] = (100, 150, 200)  # Blue-ish color
        
        # Convert to PIL and encode
        pil_image = Image.fromarray(test_image)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=95)
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        # Test single image processing
        payload = {
            "image": [img_str, img_str],
            "question": "Is there a blue object in this image?",
            # "camera_mode": "left_only"
            "camera_mode": "both"
        }
        
        response = requests.post(
            f"{service_url}/process_image",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Image processing test passed!")
            print(f"   Response: {data.get('response', 'N/A')}")
            print(f"   Inference time: {data.get('inference_time', 'N/A')}s")
        else:
            print(f"❌ Image processing test failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Image processing test error: {e}")
        return False
    
    print("\n🎉 All tests passed! Qwen VLM service is working correctly.")
    return True

if __name__ == "__main__":
    success = test_qwen_service()
    if not success:
        print("\n❌ Some tests failed. Check the service logs.")
        exit(1)
