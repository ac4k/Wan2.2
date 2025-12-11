#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量评估脚本：根据prompt_dimension_map.json评估vbench_videos目录下的所有视频
"""

import os
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='批量评估VBench视频')
    parser.add_argument('--videos_dir', type=str, default='./vbench_videos',
                        help='视频目录')
    parser.add_argument('--prompts_file', type=str, default='./selected_prompts_50.txt',
                        help='Prompts文件')
    parser.add_argument('--dimension_map_file', type=str, default='./prompt_dimension_map.json',
                        help='维度映射文件')
    parser.add_argument('--vbench_dir', type=str, default='/home/qiuyao/commit/vbench/VBench',
                        help='VBench项目目录')
    parser.add_argument('--output_file', type=str, default='./vbench_evaluation_results.json',
                        help='输出结果JSON文件')
    parser.add_argument('--max_videos', type=int, default=None,
                        help='最大处理视频数量（None表示处理所有视频）')
    return parser.parse_args()


def load_prompts(prompts_file):
    """加载prompts列表"""
    prompts = []
    with open(prompts_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def load_dimension_map(dimension_map_file):
    """加载维度映射"""
    with open(dimension_map_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_video(video_path, prompt, dimensions, vbench_dir, output_dir):
    """评估单个视频的多个维度"""
    results = {}
    
    # 将路径转换为绝对路径
    video_path_abs = os.path.abspath(video_path)
    output_dir_abs = os.path.abspath(output_dir)
    
    vbench_eval_script = os.path.join(vbench_dir, 'evaluate.py')
    vbench_full_info = os.path.join(vbench_dir, 'vbench', 'VBench_full_info.json')
    
    if not os.path.exists(vbench_full_info):
        vbench_full_info = os.path.join(vbench_dir, 'VBench_full_info.json')
    
    vbench_full_info_abs = os.path.abspath(vbench_full_info) if os.path.exists(vbench_full_info) else None
    
    # 对每个维度进行评估
    for dimension in dimensions:
        print(f"  评估维度: {dimension}")
        
        # 为每个维度创建单独的输出目录
        dim_output_dir = os.path.join(output_dir_abs, dimension)
        os.makedirs(dim_output_dir, exist_ok=True)
        
        # 构建评估命令
        cmd = [
            'python3', vbench_eval_script,
            '--videos_path', video_path_abs,
            '--dimension', dimension,
            '--mode', 'custom_input',
            '--prompt', prompt,
            '--output_path', dim_output_dir
        ]
        
        if vbench_full_info_abs:
            cmd.extend(['--full_json_dir', vbench_full_info_abs])
        
        # 调试：打印命令（仅在详细模式下）
        if os.environ.get('VBENCH_DEBUG', '').lower() == 'true':
            print(f"      执行命令: {' '.join(cmd)}")
            print(f"      工作目录: {vbench_dir}")
        
        try:
            # 设置环境变量以避免分布式问题（单GPU评估）
            env = os.environ.copy()
            env.setdefault('MASTER_ADDR', 'localhost')
            env.setdefault('MASTER_PORT', '29500')
            env.setdefault('RANK', '0')
            env.setdefault('LOCAL_RANK', '0')
            env.setdefault('WORLD_SIZE', '1')
            
            # 切换到VBench目录执行
            result = subprocess.run(
                cmd,
                cwd=vbench_dir,
                capture_output=True,
                text=True,
                timeout=600,  # 10分钟超时
                env=env
            )
            
            if result.returncode == 0:
                # 查找评估结果文件
                result_files = list(Path(dim_output_dir).glob("*_eval_results.json"))
                if result_files:
                    latest_result = max(result_files, key=os.path.getmtime)
                    with open(latest_result, 'r', encoding='utf-8') as f:
                        eval_data = json.load(f)
                        if dimension in eval_data:
                            score_data = eval_data[dimension]
                            # VBench结果格式: [分数值, [详细结果列表]]
                            if isinstance(score_data, list) and len(score_data) > 0:
                                # 第一个元素是分数
                                score = score_data[0]
                                if isinstance(score, (int, float)):
                                    results[dimension] = float(score)
                                    print(f"    ✓ {dimension}: {results[dimension]:.4f}")
                                else:
                                    print(f"    ✗ {dimension}: 分数格式不正确")
                                    results[dimension] = None
                            elif isinstance(score_data, (int, float)):
                                results[dimension] = float(score_data)
                                print(f"    ✓ {dimension}: {results[dimension]:.4f}")
                            elif isinstance(score_data, dict):
                                # 尝试从字典中提取分数
                                if 'video_results' in score_data:
                                    video_results = score_data['video_results']
                                    if isinstance(video_results, list) and len(video_results) > 0:
                                        first = video_results[0]
                                        if isinstance(first, dict) and 'video_results' in first:
                                            results[dimension] = float(first['video_results'])
                                            print(f"    ✓ {dimension}: {results[dimension]:.4f}")
                                        else:
                                            results[dimension] = None
                                    else:
                                        results[dimension] = None
                                else:
                                    results[dimension] = None
                            else:
                                print(f"    ✗ {dimension}: 无法解析分数格式")
                                results[dimension] = None
                        else:
                            print(f"    ✗ {dimension}: 结果中未找到该维度")
                            results[dimension] = None
                else:
                    print(f"    ✗ {dimension}: 未找到结果文件")
                    results[dimension] = None
            else:
                print(f"    ✗ {dimension}: 评估失败")
                print(f"      返回码: {result.returncode}")
                if result.stderr:
                    print(f"      错误输出 (完整):")
                    print("      " + "\n      ".join(result.stderr.split('\n')))
                if result.stdout:
                    print(f"      标准输出:")
                    print("      " + "\n      ".join(result.stdout.split('\n')))
                results[dimension] = None
                
        except subprocess.TimeoutExpired:
            print(f"    ✗ {dimension}: 评估超时")
            results[dimension] = None
        except subprocess.TimeoutExpired as e:
            print(f"    ✗ {dimension}: 评估超时 (超过10分钟)")
            if hasattr(e, 'stdout') and e.stdout:
                print(f"      输出: {e.stdout[:500]}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"      错误: {e.stderr[:500]}")
            results[dimension] = None
        except Exception as e:
            print(f"    ✗ {dimension}: 评估异常 - {type(e).__name__}: {e}")
            import traceback
            print(f"      详细错误:")
            print("      " + "\n      ".join(traceback.format_exc().split('\n')[:10]))
            results[dimension] = None
    
    return results


def main():
    args = parse_args()
    
    print("=" * 60)
    print("批量评估VBench视频")
    print("=" * 60)
    print(f"视频目录: {args.videos_dir}")
    print(f"Prompts文件: {args.prompts_file}")
    print(f"维度映射文件: {args.dimension_map_file}")
    print(f"输出文件: {args.output_file}")
    print()
    
    # 加载数据
    print("加载数据...")
    prompts = load_prompts(args.prompts_file)
    dimension_map = load_dimension_map(args.dimension_map_file)
    
    print(f"加载了 {len(prompts)} 条prompts")
    print(f"加载了 {len(dimension_map)} 条维度映射")
    print()
    
    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(args.output_file), 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # 评估结果
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'total_videos': len(prompts),
        'results': []
    }
    
    # 处理每个视频
    max_videos = args.max_videos if args.max_videos else len(prompts)
    total_to_process = min(max_videos, len(prompts))
    
    for idx, prompt in enumerate(prompts[:total_to_process], start=1):
        video_filename = f"prompt_{idx}.mp4"
        video_path = os.path.join(args.videos_dir, video_filename)
        
        if not os.path.exists(video_path):
            print(f"[{idx}/{total_to_process}] 跳过: {video_filename} (文件不存在)")
            continue
        
        # 获取该prompt对应的维度
        dimensions = dimension_map.get(prompt, [])
        if not dimensions:
            print(f"[{idx}/{total_to_process}] 跳过: {video_filename} (未找到维度映射)")
            continue
        
        print(f"[{idx}/{total_to_process}] 评估: {video_filename}")
        print(f"  Prompt: {prompt[:60]}...")
        print(f"  维度: {', '.join(dimensions)}")
        
        # 评估视频
        video_results = evaluate_video(
            video_path, prompt, dimensions, 
            args.vbench_dir, output_dir
        )
        
        # 记录结果
        all_results['results'].append({
            'video_file': video_filename,
            'video_path': os.path.abspath(video_path),
            'prompt': prompt,
            'dimensions': dimensions,
            'scores': video_results
        })
        
        print()
    
    # 保存结果
    print("=" * 60)
    print("保存评估结果...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"结果已保存到: {args.output_file}")
    
    # 统计信息
    print()
    print("=" * 60)
    print("评估统计")
    print("=" * 60)
    
    total_evaluated = len(all_results['results'])
    dimension_stats = {}
    
    for result in all_results['results']:
        for dim, score in result['scores'].items():
            if dim not in dimension_stats:
                dimension_stats[dim] = {'total': 0, 'success': 0, 'failed': 0}
            dimension_stats[dim]['total'] += 1
            if score is not None:
                dimension_stats[dim]['success'] += 1
            else:
                dimension_stats[dim]['failed'] += 1
    
    print(f"总prompts数: {len(prompts)}")
    print(f"处理数量: {total_to_process}")
    print(f"已评估: {total_evaluated}")
    print()
    print("各维度统计:")
    for dim, stats in sorted(dimension_stats.items()):
        print(f"  {dim}: 成功 {stats['success']}/{stats['total']}")
    
    print()
    print("评估完成！")


if __name__ == "__main__":
    main()

