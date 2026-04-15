import json
import glob
import os

# 指向你解压的 bc 根目录
base_dir = r"D:\prop"
# 自动寻找所有 case 文件夹下的 case.json
json_files = glob.glob(os.path.join(base_dir, "case*", "case.json"))

print(f"🔍 正在扫描边界条件参数... 发现 {len(json_files)} 个工况文件。")

# 用来存放所有参数的集合
all_params = {}

for file in json_files:
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # 提取 json 中的所有数值型参数
        for key, value in data.items():
            if isinstance(value, (int, float)):
                if key not in all_params:
                    all_params[key] = []
                all_params[key].append(value)

print("-" * 30)
print("📊 扫描结果 (即 Branch Net 的输入边界):")
for key, values in all_params.items():
    print(f"参数 [{key}]:")
    print(f"  -> branch_min = {min(values)}")
    print(f"  -> branch_max = {max(values)}")
print("-" * 30)