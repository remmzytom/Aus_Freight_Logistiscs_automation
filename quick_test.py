#!/usr/bin/env python3
"""
Ultra-fast test - just checks files exist, no heavy imports
"""

import os
import sys

def quick_test():
    """Super fast test - just file existence"""
    print("🚀 QUICK AUTOMATION TEST")
    print("=" * 30)
    
    # Check essential files
    files_to_check = [
        '2024_2025_extractor.py',
        'data_cleaning.ipynb', 
        'data_analysis.ipynb',
        'app.py',
        'simple_automation.py',
        'scheduler.py',
        'requirements.txt',
        'data/exports_2024_2025.csv'
    ]
    
    all_good = True
    for file in files_to_check:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            all_good = False
    
    print("\n" + "=" * 30)
    if all_good:
        print("🎉 ALL FILES PRESENT!")
        print("\n📋 Ready to test automation:")
        print("1. python simple_automation.py")
        print("2. streamlit run app.py")
    else:
        print("⚠️  Some files missing")
    
    return all_good

if __name__ == "__main__":
    quick_test()

