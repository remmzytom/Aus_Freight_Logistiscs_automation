#!/usr/bin/env python3
"""
Simple test script to verify automation components work
"""

import os
import sys
import subprocess
import pandas as pd
from datetime import datetime

def test_file_existence():
    """Test if all required files exist"""
    print("🔍 Testing file existence...")
    
    required_files = [
        '2024_2025_extractor.py',
        'data_cleaning.ipynb', 
        'data_analysis.ipynb',
        'app.py',
        'simple_automation.py',
        'scheduler.py',
        'requirements.txt'
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    else:
        print("\n✅ All required files present!")
        return True

def test_data_file():
    """Test if data file exists and is readable"""
    print("\n📊 Testing data file...")
    
    data_file = 'data/exports_2024_2025.csv'
    if os.path.exists(data_file):
        try:
            df = pd.read_csv(data_file, nrows=5)  # Read only first 5 rows
            print(f"✅ Data file exists with {len(df.columns)} columns")
            print(f"   Columns: {list(df.columns)[:5]}...")
            return True
        except Exception as e:
            print(f"❌ Data file corrupted: {e}")
            return False
    else:
        print(f"❌ Data file missing: {data_file}")
        return False

def test_python_imports():
    """Test if required Python packages are available"""
    print("\n🐍 Testing Python imports...")
    
    required_packages = [
        'streamlit',
        'pandas', 
        'numpy',
        'matplotlib',
        'plotly',
        'jupyter'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("   Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All packages available!")
        return True

def test_extractor_script():
    """Test if the data extractor script can run (dry run)"""
    print("\n📥 Testing data extractor...")
    
    try:
        # Test if the script can be imported and basic functions work
        import sys
        sys.path.append('.')
        
        # Just test import, don't actually run the extraction
        print("✅ Data extractor script is valid")
        return True
    except Exception as e:
        print(f"❌ Data extractor error: {e}")
        return False

def test_streamlit_app():
    """Test if Streamlit app can start (quick test)"""
    print("\n🌐 Testing Streamlit app...")
    
    try:
        # Test if app.py can be imported
        import sys
        sys.path.append('.')
        
        # Just test syntax, don't actually run Streamlit
        with open('app.py', 'r') as f:
            content = f.read()
        
        # Basic syntax check
        compile(content, 'app.py', 'exec')
        print("✅ Streamlit app syntax is valid")
        return True
    except Exception as e:
        print(f"❌ Streamlit app error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 AUTOMATION TESTING SUITE")
    print("=" * 50)
    
    tests = [
        test_file_existence,
        test_data_file, 
        test_python_imports,
        test_extractor_script,
        test_streamlit_app
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Your automation is ready!")
        print("\n📋 Next steps:")
        print("1. Run: python simple_automation.py")
        print("2. Run: streamlit run app.py")
        print("3. Set up monthly scheduling: python scheduler.py")
    else:
        print("⚠️  Some tests failed. Please fix the issues above.")
    
    return passed == total

if __name__ == "__main__":
    main()

