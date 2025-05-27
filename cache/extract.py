import os
import random

# 设置随机种子（确保每次运行一致）
random.seed(42)

# 输入文件夹路径
input_folder = "./echo-data"  # 这里替换为你的文件夹路径
output_folder = "/data1/jiyifan/llm2vec-main/cache/echo-data-50k"       # 输出文件夹路径

# 抽样比例
sample_ratio = 0.33

# 创建输出文件夹（如果不存在）
os.makedirs(output_folder, exist_ok=True)

# 找到所有 .jsonl 文件
jsonl_files = [f for f in os.listdir(input_folder) if f.endswith(".jsonl") or f.endswith(".json")]

print(f"找到 {len(jsonl_files)} 个文件进行处理...")

for file_name in jsonl_files:
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name)  # 同名文件输出到新文件夹

    # 读取原始数据
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    total = len(lines)
    sample_count = int(total * sample_ratio)

    # 抽样索引
    sample_indices = set(random.sample(range(total), sample_count))

    print(f"文件 {file_name}: 总行数 {total}, 抽样 {sample_count}")

    # 写入抽样后的文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, line in enumerate(lines):
            if idx in sample_indices:
                f.write(line)

print("\n✅ 所有文件已抽样完成，输出文件存放在:", output_folder)
