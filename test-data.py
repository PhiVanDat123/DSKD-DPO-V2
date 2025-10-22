import os
import json

# Thư mục input và output
input_dir = "/home/hungpv/projects/DSKD-DPO/datasets/data"
output_dir = "/home/hungpv/projects/DSKD-DPO/datasets/data100"

# Tạo thư mục output nếu chưa tồn tại
os.makedirs(output_dir, exist_ok=True)

# Danh sách file JSONL (chỉ tên file, không bao gồm path)
files = [
    "test_gen-00000-of-00001.jsonl",
    "test_prefs-00000-of-00001.jsonl",
    "test_sft-00000-of-00001.jsonl",
    "train_gen-00000-of-00001.jsonl",
    "train_prefs-00000-of-00001.jsonl",
    "train_sft-00000-of-00001.jsonl"
]

sample_size = 100

for file_name in files:
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name.replace(".jsonl", "_sample100.jsonl"))
    
    # Kiểm tra file tồn tại
    if not os.path.exists(input_path):
        print(f"File không tồn tại: {input_path}")
        continue

    # Đọc 100 mẫu
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            samples.append(json.loads(line))
    
    # Ghi ra file mới
    with open(output_path, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"{input_path} -> {output_path} ({len(samples)} samples)")

