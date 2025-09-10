#!/usr/bin/env python3
"""
Test script to verify refactored code compilation
This tests the Python components and validates R script structure
"""

import os
import subprocess
import sys

def test_python_compilation():
    """Test Python file compilation"""
    python_files = [
        'src/run_g_or_gxe_model.py',
        'src/run_e_model.py', 
        'src/preprocessing.py',
        'src/evaluate.py',
        'src/create_datasets.py',
        'src/create_individuals.py'
    ]
    
    print("=== Testing Python Compilation ===")
    all_passed = True
    
    for py_file in python_files:
        if os.path.exists(py_file):
            try:
                subprocess.run([sys.executable, '-m', 'py_compile', py_file], 
                             check=True, capture_output=True)
                print(f"✓ {py_file} - PASSED")
            except subprocess.CalledProcessError as e:
                print(f"✗ {py_file} - FAILED: {e}")
                all_passed = False
        else:
            print(f"⚠ {py_file} - NOT FOUND")
    
    return all_passed

def test_r_structure():
    """Test R file structure"""
    r_files = {
        'src/blues.R': ['library(sommer)', 'mmer('],
        'src/fa.R': ['library(sommer)', 'mmer('],
        'src/kinship.R': ['library(AGHmatrix)', 'Gmatrix('],
        'src/kronecker.R': ['library(data.table)', 'kronecker('],
        'src/comparisons.R': ['library(dplyr)']
    }
    
    print("\n=== Testing R Script Structure ===")
    all_passed = True
    
    for r_file, expected_patterns in r_files.items():
        if os.path.exists(r_file):
            with open(r_file, 'r') as f:
                content = f.read()
            
            missing_patterns = []
            for pattern in expected_patterns:
                if pattern not in content:
                    missing_patterns.append(pattern)
            
            if missing_patterns:
                print(f"✗ {r_file} - MISSING: {missing_patterns}")
                all_passed = False
            else:
                print(f"✓ {r_file} - STRUCTURE OK")
        else:
            print(f"⚠ {r_file} - NOT FOUND")
            all_passed = False
    
    return all_passed

def check_migration_completeness():
    """Check if migration is complete"""
    print("\n=== Migration Completeness Check ===")
    
    # Check if migration docs exist
    docs_exist = all(os.path.exists(f) for f in ['MIGRATION.md', 'README.md'])
    print(f"✓ Documentation: {'COMPLETE' if docs_exist else 'INCOMPLETE'}")
    
    # Check if environment.yml is updated (no asreml reference)
    with open('environment.yml', 'r') as f:
        env_content = f.read()
    
    has_asreml = 'asreml' in env_content.lower()
    print(f"✓ Environment: {'CLEAN' if not has_asreml else 'STILL HAS ASREML REFERENCES'}")
    
    # Check if main R files use sommer
    r_files = ['src/blues.R', 'src/fa.R']
    uses_sommer = True
    for r_file in r_files:
        if os.path.exists(r_file):
            with open(r_file, 'r') as f:
                content = f.read()
            if 'library(sommer)' not in content:
                uses_sommer = False
                break
    
    print(f"✓ R Migration: {'COMPLETE' if uses_sommer else 'INCOMPLETE'}")
    
    return docs_exist and not has_asreml and uses_sommer

def main():
    print("Maize GxE Prediction - Compilation Test")
    print("=" * 50)
    
    # Test Python compilation
    python_ok = test_python_compilation()
    
    # Test R structure  
    r_ok = test_r_structure()
    
    # Check migration completeness
    migration_ok = check_migration_completeness()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print(f"Python Compilation: {'✓ PASSED' if python_ok else '✗ FAILED'}")
    print(f"R Structure: {'✓ PASSED' if r_ok else '✗ FAILED'}")
    print(f"Migration: {'✓ COMPLETE' if migration_ok else '✗ INCOMPLETE'}")
    
    overall_status = python_ok and r_ok and migration_ok
    print(f"\nOverall Status: {'✓ READY FOR TRAINING' if overall_status else '✗ NEEDS FIXES'}")
    
    if overall_status:
        print("\n🎉 Refactoring successful! Code is ready for training with open-source dependencies.")
        print("📝 See MIGRATION.md for details on expected numerical differences.")
    else:
        print("\n⚠️  Some issues detected. Please review the output above.")
    
    return 0 if overall_status else 1

if __name__ == "__main__":
    sys.exit(main())