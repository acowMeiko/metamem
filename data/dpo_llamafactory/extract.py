import json
import os

def extract_json_samples(input_file, output_file, num_samples=128):
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        return

    try:
        # 尝试以 JSONL 格式读取（逐行读取，内存占用极低）
        samples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try:
                    samples.append(json.loads(line))
                    if len(samples) >= num_samples:
                        break
                except json.JSONDecodeError:
                    # 如果第一行就报错，说明可能不是 JSONL，尝试读取为标准 JSON 数组
                    if i == 0:
                        break 
                    else:
                        print(f"警告: 跳过第 {i+1} 行（解析错误）")

        # 如果 JSONL 没读到数据，尝试读取为标准 JSON 数组
        if not samples:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    samples = data[:num_samples]
                else:
                    print("错误: JSON 文件根节点不是列表格式")
                    return

        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            # 默认保存为带缩进的标准 JSON 数组，方便查看
            json.dump(samples, f, ensure_ascii=False, indent=4)
        
        print(f"成功！已提取 {len(samples)} 个样本到: {output_file}")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")

if __name__ == "__main__":
    # --- 你可以在这里修改文件名 ---
    input_path = "D:\\experiment\\code\\metanew2\\data\\dpo_llamafactory\\dpo_level_level1_llamafactory.json"   # 原始文件名
    output_path = "D:\\experiment\\code\\metanew2\\data\\dpo_llamafactory\\test.json" # 输出文件名
    
    extract_json_samples(input_path, output_path, 128)