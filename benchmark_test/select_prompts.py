#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从VBench的prompts中均匀挑选50条prompts，确保覆盖多个dimension
"""

import os
from pathlib import Path
from collections import defaultdict

# 定义要覆盖的dimension及其对应的prompt文件
# 优先选择custom_input支持的维度
DIMENSION_PROMPT_MAP = {
    # 官方明确支持的维度
    'subject_consistency': 'subject_consistency.txt',
    'background_consistency': 'scene.txt',  # background_consistency使用scene的prompts
    'motion_smoothness': 'subject_consistency.txt',  # 使用subject_consistency的prompts
    'dynamic_degree': 'subject_consistency.txt',  # 使用subject_consistency的prompts
    'aesthetic_quality': 'overall_consistency.txt',  # 使用overall_consistency的prompts
    'imaging_quality': 'overall_consistency.txt',  # 使用overall_consistency的prompts
    
    # 代码支持但文档未列出的维度
    'overall_consistency': 'overall_consistency.txt',
    'temporal_style': 'temporal_style.txt',
    'human_action': 'human_action.txt',
    'temporal_flickering': 'temporal_flickering.txt',
}

# 需要覆盖的主要dimension（去重后）
PRIMARY_DIMENSIONS = [
    'subject_consistency',
    'scene',  # background_consistency使用这个
    'overall_consistency',  # aesthetic_quality和imaging_quality使用这个
    'temporal_style',
    'human_action',
    'temporal_flickering',
]

def load_prompts_from_file(file_path):
    """从文件中加载prompts"""
    if not os.path.exists(file_path):
        print(f"警告: 文件不存在 {file_path}")
        return []
    
    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                prompts.append(line)
    return prompts

def select_prompts_uniformly(prompts, num_select):
    """均匀挑选prompts"""
    if len(prompts) <= num_select:
        return prompts
    
    # 计算步长，确保均匀分布
    step = len(prompts) / num_select
    selected = []
    for i in range(num_select):
        idx = int(i * step)
        selected.append(prompts[idx])
    
    return selected

def main():
    # 设置路径
    prompts_dir = Path('/home/qiuyao/commit/vbench/VBench/prompts/prompts_per_dimension')
    output_file = Path('/home/qiuyao/commit/Wan2.2/selected_prompts_50.txt')
    
    # 目标：50条prompts
    total_prompts = 50
    
    # 计算每个主要dimension应该挑选多少条
    num_dimensions = len(PRIMARY_DIMENSIONS)
    prompts_per_dimension = total_prompts // num_dimensions
    remainder = total_prompts % num_dimensions
    
    print(f"目标: 挑选 {total_prompts} 条prompts")
    print(f"覆盖 {num_dimensions} 个主要dimension")
    print(f"每个dimension约 {prompts_per_dimension} 条\n")
    
    selected_prompts = []
    dimension_stats = {}
    
    # 从每个dimension中挑选prompts
    for i, dimension in enumerate(PRIMARY_DIMENSIONS):
        # 最后一个dimension分配剩余的prompts
        num_to_select = prompts_per_dimension + (1 if i < remainder else 0)
        
        prompt_file = prompts_dir / f"{dimension}.txt"
        
        if not prompt_file.exists():
            print(f"警告: {prompt_file} 不存在，跳过")
            continue
        
        # 加载prompts
        all_prompts = load_prompts_from_file(prompt_file)
        
        if len(all_prompts) == 0:
            print(f"警告: {dimension} 没有prompts，跳过")
            continue
        
        # 均匀挑选
        selected = select_prompts_uniformly(all_prompts, num_to_select)
        
        selected_prompts.extend(selected)
        dimension_stats[dimension] = len(selected)
        
        print(f"✓ {dimension}: 从 {len(all_prompts)} 条中挑选了 {len(selected)} 条")
    
    # 如果总数不够，从最多的dimension中补充
    if len(selected_prompts) < total_prompts:
        needed = total_prompts - len(selected_prompts)
        print(f"\n需要补充 {needed} 条prompts")
        
        # 找到prompts最多的dimension
        max_dimension = max(dimension_stats.items(), key=lambda x: x[1])[0]
        prompt_file = prompts_dir / f"{max_dimension}.txt"
        all_prompts = load_prompts_from_file(prompt_file)
        
        # 从已选中的prompts中排除，然后补充
        selected_set = set(selected_prompts)
        available = [p for p in all_prompts if p not in selected_set]
        
        if len(available) >= needed:
            # 均匀挑选补充
            step = len(available) / needed
            for i in range(needed):
                idx = int(i * step)
                selected_prompts.append(available[idx])
            dimension_stats[max_dimension] += needed
            print(f"从 {max_dimension} 补充了 {needed} 条")
    
    # 去重（虽然理论上不应该有重复）
    selected_prompts = list(dict.fromkeys(selected_prompts))
    
    # 保存到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt in selected_prompts:
            f.write(prompt + '\n')
    
    print(f"\n{'='*60}")
    print(f"成功挑选 {len(selected_prompts)} 条prompts")
    print(f"输出文件: {output_file}")
    print(f"{'='*60}")
    print("\n各dimension分布:")
    for dim, count in sorted(dimension_stats.items()):
        print(f"  {dim}: {count} 条")
    
    # 显示前10条作为示例
    print(f"\n前10条prompts示例:")
    for i, prompt in enumerate(selected_prompts[:10], 1):
        print(f"  {i}. {prompt}")

if __name__ == "__main__":
    main()

