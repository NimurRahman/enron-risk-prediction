# check_folder_inventory.py
"""
Inventory all files in enron_modeling folder
Shows what you have, file sizes, types, locations
"""
import os
from pathlib import Path
import pandas as pd

print("="*80)
print("ENRON MODELING - FOLDER INVENTORY")
print("="*80)

BASE = Path(r"I:\enron_modeling")

# Collect all files
all_files = []

for root, dirs, files in os.walk(BASE):
    for file in files:
        filepath = os.path.join(root, file)
        
        # Get file info
        size = os.path.getsize(filepath)
        size_mb = size / (1024 * 1024)
        
        # Get relative path
        rel_path = os.path.relpath(filepath, BASE)
        
        # Get folder
        folder = os.path.dirname(rel_path)
        if folder == '':
            folder = 'root'
        
        # Get extension
        ext = os.path.splitext(file)[1].lower()
        
        all_files.append({
            'folder': folder,
            'filename': file,
            'extension': ext,
            'size_mb': size_mb,
            'path': rel_path
        })

# Create DataFrame
df = pd.DataFrame(all_files)

print(f"\nTOTAL FILES: {len(df)}")
print(f"TOTAL SIZE: {df['size_mb'].sum():.1f} MB")

# Summary by folder
print("\n" + "="*80)
print("SUMMARY BY FOLDER")
print("="*80)
folder_summary = df.groupby('folder').agg({
    'filename': 'count',
    'size_mb': 'sum'
}).rename(columns={'filename': 'count'})
folder_summary = folder_summary.sort_values('size_mb', ascending=False)
print(folder_summary.to_string())

# Summary by file type
print("\n" + "="*80)
print("SUMMARY BY FILE TYPE")
print("="*80)
type_summary = df.groupby('extension').agg({
    'filename': 'count',
    'size_mb': 'sum'
}).rename(columns={'filename': 'count'})
type_summary = type_summary.sort_values('size_mb', ascending=False)
print(type_summary.to_string())

# Large files (> 10 MB)
print("\n" + "="*80)
print("LARGE FILES (> 10 MB)")
print("="*80)
large_files = df[df['size_mb'] > 10].sort_values('size_mb', ascending=False)
print(large_files[['filename', 'folder', 'size_mb']].to_string(index=False))

# Files by type
print("\n" + "="*80)
print("FILES BY TYPE")
print("="*80)

for ext in ['.csv', '.parquet', '.pkl', '.png', '.py', '.md', '.txt']:
    files = df[df['extension'] == ext]
    if len(files) > 0:
        print(f"\n{ext.upper()} FILES ({len(files)}):")
        print(files[['filename', 'folder', 'size_mb']].to_string(index=False))

# Save to CSV
output_file = BASE / "FOLDER_INVENTORY.csv"
df.to_csv(output_file, index=False)
print(f"\n" + "="*80)
print(f"Full inventory saved to: {output_file}")
print("="*80)

# Critical files check
print("\n" + "="*80)
print("CRITICAL FILES CHECK")
print("="*80)

critical = {
    'predictions_test.csv': 'handoff_to_sumit',
    'predictions_test_enhanced.csv': 'outputs',
    'model_xgboost.pkl': 'models',
    'model_rf.pkl': 'models',
    'model_baseline_logreg.pkl': 'models',
    'shap_importance_xgboost.csv': 'outputs',
    'streamlit_sna_dashboard.py': 'root',
    'PROJECT_COMPLETION_SUMMARY.md': 'root'
}

for filename, expected_folder in critical.items():
    found = df[(df['filename'] == filename) & (df['folder'] == expected_folder)]
    if len(found) > 0:
        print(f"✅ {filename} - {found.iloc[0]['size_mb']:.1f} MB")
    else:
        print(f"❌ {filename} - MISSING!")

print("\n" + "="*80)
print("INVENTORY COMPLETE!")
print("="*80)