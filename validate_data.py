# validate_data.py
# CI/CD Step 1 — Validate training data before fine-tuning

import sys
import os

def validate():
    print("🔍 Starting data validation...")
    
    # Check training data exists
    training_samples = [
        "Email: Meeting at 3pm Friday. SUMMARY: Meeting Friday 3pm.",
        "Email: Submit expense reports by Thursday. SUMMARY: Submit expenses by Thursday.",
        "Email: Server maintenance Sunday 2am. SUMMARY: Server down Sunday 2am.",
        "Email: Complete security training by December 31st. SUMMARY: Security training due Dec 31.",
        "Email: Product launch confirmed for Q1. SUMMARY: Product launches Q1."
    ]
    
    # Check minimum samples
    min_samples = 3
    if len(training_samples) < min_samples:
        print(f"❌ Insufficient training data: {len(training_samples)} samples (minimum {min_samples})")
        sys.exit(1)
    
    # Check format
    for i, sample in enumerate(training_samples):
        if "Email:" not in sample or "SUMMARY:" not in sample:
            print(f"❌ Sample {i} has wrong format")
            sys.exit(1)
    
    # Check for empty samples
    for i, sample in enumerate(training_samples):
        if len(sample.strip()) == 0:
            print(f"❌ Sample {i} is empty")
            sys.exit(1)
    
    print(f"✅ Data validation passed — {len(training_samples)} samples validated")
    return True

if __name__ == "__main__":
    validate()