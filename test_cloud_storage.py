#!/usr/bin/env python3
"""
Test script to verify Cloud Storage integration
This simulates what automation does - uploads data to Cloud Storage
"""

import os
from pathlib import Path

try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    print("ERROR: google-cloud-storage not installed")
    print("Install with: pip install google-cloud-storage")
    exit(1)

# Configuration
BUCKET_NAME = "aus-freight-logistics-data"
PROJECT_DIR = Path(__file__).parent
LOCAL_FILE = PROJECT_DIR / "data" / "exports_cleaned.csv"
GCS_BLOB_NAME = "exports_cleaned.csv"

def test_upload():
    """Test uploading file to Cloud Storage"""
    print("=" * 60)
    print("Testing Cloud Storage Upload")
    print("=" * 60)
    
    # Check if local file exists
    if not LOCAL_FILE.exists():
        print(f"❌ ERROR: Local file not found: {LOCAL_FILE}")
        print("Please run automation first to generate the data file.")
        return False
    
    print(f"✅ Found local file: {LOCAL_FILE}")
    file_size = LOCAL_FILE.stat().st_size / (1024 * 1024)  # Size in MB
    print(f"   File size: {file_size:.2f} MB")
    
    try:
        # Initialize Cloud Storage client
        print(f"\n📤 Uploading to Cloud Storage...")
        print(f"   Bucket: gs://{BUCKET_NAME}")
        print(f"   Blob: {GCS_BLOB_NAME}")
        
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(GCS_BLOB_NAME)
        
        # Upload file
        blob.upload_from_filename(str(LOCAL_FILE))
        
        print(f"✅ Upload successful!")
        print(f"   File uploaded: {GCS_BLOB_NAME}")
        
        # Verify upload
        if blob.exists():
            print(f"✅ Verification: File exists in Cloud Storage")
            print(f"   Cloud Storage URL: gs://{BUCKET_NAME}/{GCS_BLOB_NAME}")
            return True
        else:
            print(f"❌ ERROR: Upload verification failed")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: Upload failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def test_download():
    """Test downloading file from Cloud Storage"""
    print("\n" + "=" * 60)
    print("Testing Cloud Storage Download")
    print("=" * 60)
    
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(GCS_BLOB_NAME)
        
        if not blob.exists():
            print(f"❌ ERROR: File not found in Cloud Storage")
            print(f"   Please upload first using: python test_cloud_storage.py")
            return False
        
        print(f"✅ File found in Cloud Storage")
        print(f"   Bucket: gs://{BUCKET_NAME}")
        print(f"   Blob: {GCS_BLOB_NAME}")
        
        # Get file metadata
        blob.reload()
        size_mb = blob.size / (1024 * 1024)
        updated = blob.updated
        
        print(f"   Size: {size_mb:.2f} MB")
        print(f"   Last updated: {updated}")
        
        # Test download
        test_download_path = PROJECT_DIR / "data" / "test_download.csv"
        print(f"\n📥 Testing download...")
        blob.download_to_filename(str(test_download_path))
        
        if test_download_path.exists():
            print(f"✅ Download successful!")
            print(f"   Downloaded to: {test_download_path}")
            print(f"   Size: {test_download_path.stat().st_size / (1024 * 1024):.2f} MB")
            
            # Clean up test file
            test_download_path.unlink()
            print(f"   Test file cleaned up")
            return True
        else:
            print(f"❌ ERROR: Download verification failed")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: Download failed: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def check_bucket_contents():
    """List all files in the bucket"""
    print("\n" + "=" * 60)
    print("Cloud Storage Bucket Contents")
    print("=" * 60)
    
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blobs = list(bucket.list_blobs())
        
        if not blobs:
            print(f"⚠️  Bucket is empty: gs://{BUCKET_NAME}")
            print(f"   No files uploaded yet.")
            return False
        
        print(f"✅ Found {len(blobs)} file(s) in bucket:")
        for blob in blobs:
            size_mb = blob.size / (1024 * 1024)
            print(f"   - {blob.name} ({size_mb:.2f} MB, updated: {blob.updated})")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to list bucket contents: {str(e)}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "upload":
            success = test_upload()
            exit(0 if success else 1)
        elif command == "download":
            success = test_download()
            exit(0 if success else 1)
        elif command == "list":
            success = check_bucket_contents()
            exit(0 if success else 1)
        else:
            print(f"Unknown command: {command}")
            print("Usage: python test_cloud_storage.py [upload|download|list]")
            exit(1)
    else:
        # Run all tests
        print("Running full Cloud Storage integration test...\n")
        
        # Check bucket contents first
        check_bucket_contents()
        
        # Test upload
        upload_success = test_upload()
        
        # Test download
        download_success = test_download()
        
        # Final check
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"Upload test: {'✅ PASSED' if upload_success else '❌ FAILED'}")
        print(f"Download test: {'✅ PASSED' if download_success else '❌ FAILED'}")
        
        if upload_success and download_success:
            print("\n🎉 All tests passed! Cloud Storage integration is working.")
            print("\nNext steps:")
            print("1. Run automation - it will upload automatically")
            print("2. Refresh Cloud Run dashboard - it will download automatically")
        else:
            print("\n⚠️  Some tests failed. Please check the errors above.")
        
        exit(0 if (upload_success and download_success) else 1)

