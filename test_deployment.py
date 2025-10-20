#!/usr/bin/env python3
"""
Test script for deployed API
"""

import requests
import json
import sys
import time
from pathlib import Path

def test_api(base_url):
    """Test the deployed API endpoints"""
    
    print(f"🧪 Testing API at: {base_url}")
    print("=" * 50)
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test 2: Root endpoint
    print("\n2. Testing root endpoint...")
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            print("✅ Root endpoint works")
            print(f"   API Name: {response.json().get('name', 'Unknown')}")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
    
    # Test 3: OpenAPI docs
    print("\n3. Testing API documentation...")
    try:
        response = requests.get(f"{base_url}/docs", timeout=10)
        if response.status_code == 200:
            print("✅ API documentation accessible")
            print(f"   Docs URL: {base_url}/docs")
        else:
            print(f"❌ API docs failed: {response.status_code}")
    except Exception as e:
        print(f"❌ API docs failed: {e}")
    
    # Test 4: Detection endpoint with sample image (if available)
    print("\n4. Testing detection endpoint...")
    sample_images = list(Path("data/images").glob("*")) if Path("data/images").exists() else []
    
    if sample_images:
        try:
            image_path = sample_images[0]
            print(f"   Using sample image: {image_path.name}")
            
            with open(image_path, 'rb') as f:
                files = {'file': f}
                params = {
                    'process_labels': True,
                    'include_annotated_image': True,
                    'ocr_confidence': 0.5
                }
                
                start_time = time.time()
                response = requests.post(
                    f"{base_url}/detect", 
                    files=files, 
                    params=params, 
                    timeout=60
                )
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ Detection endpoint works")
                    print(f"   Processing time: {processing_time:.2f}s")
                    print(f"   Boxes detected: {result['summary']['total_boxes']}")
                    print(f"   Labels detected: {result['summary']['boxes_with_labels']}")
                    print(f"   Barcodes found: {result['summary']['barcodes_found']}")
                    print(f"   QR codes found: {result['summary']['qrcodes_found']}")
                    
                    if result.get('annotated_image'):
                        print("   ✅ Annotated image included")
                    else:
                        print("   ⚠️ No annotated image in response")
                        
                else:
                    print(f"❌ Detection failed: {response.status_code}")
                    print(f"   Error: {response.text}")
                    
        except Exception as e:
            print(f"❌ Detection test failed: {e}")
    else:
        print("   ⚠️ No sample images found in data/images/")
        print("   Skipping detection test")
    
    print("\n" + "=" * 50)
    print("🎉 API testing completed!")
    print(f"📚 Full API documentation: {base_url}/docs")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_deployment.py <api_url>")
        print("Example: python test_deployment.py https://your-service-url.run.app")
        sys.exit(1)
    
    api_url = sys.argv[1].rstrip('/')
    test_api(api_url)