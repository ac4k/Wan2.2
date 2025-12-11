#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
根据prompts.txt生成维度映射JSON文件
"""

import json
import os

# 读取prompts
prompts_file = './selected_prompts_50.txt'
vbench_info_file = '/home/qiuyao/commit/vbench/VBench/vbench/VBench_full_info.json'
output_file = './prompt_dimension_map.json'

# 读取prompts
prompts = []
with open(prompts_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            prompts.append(line)

print(f"读取了 {len(prompts)} 条prompts")

# 从VBench_full_info.json中查找维度
prompt_to_dimensions = {}
if os.path.exists(vbench_info_file):
    with open(vbench_info_file, 'r', encoding='utf-8') as f:
        vbench_data = json.load(f)
    
    for item in vbench_data:
        prompt_en = item.get('prompt_en', '').strip()
        dimensions = item.get('dimension', [])
        if prompt_en:
            prompt_to_dimensions[prompt_en] = dimensions

# 为每条prompt确定维度
result = {}
for prompt in prompts:
    # 从VBench_full_info.json中查找
    dimensions = prompt_to_dimensions.get(prompt, [])
    
    # 如果找不到，使用默认维度
    if not dimensions:
        dimensions = ['overall_consistency', 'aesthetic_quality', 'imaging_quality']
    
    result[prompt] = sorted(list(set(dimensions)))

# 保存结果
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print(f"维度映射已保存到: {output_file}")
print(f"\n示例（前5条）:")
for i, (prompt, dims) in enumerate(list(result.items())[:5]):
    print(f"  {prompt[:50]}... -> {dims}")

