import pandas as pd
import os

# Thư mục chứa file parquet
input_dir = "datasets/data"
output_dir = "datasets/data"

# Danh sách file parquet cần convert
parquet_files = [
    "test_gen-00000-of-00001.parquet",
    "test_prefs-00000-of-00001.parquet",
    "test_sft-00000-of-00001.parquet",
    "train_gen-00000-of-00001.parquet",
    "train_prefs-00000-of-00001.parquet",
    "train_sft-00000-of-00001.parquet",
]

# Convert từng file
for file_name in parquet_files:
    input_path = os.path.join(input_dir, file_name)
    output_file_name = file_name.replace(".parquet", ".jsonl")
    output_path = os.path.join(output_dir, output_file_name)
    
    print(f"Converting {input_path} -> {output_path} ...")
    df = pd.read_parquet(input_path)
    df.to_json(output_path, orient="records", lines=True, force_ascii=False)
    print(f"Done: {output_path}")

print("All files converted successfully!")

