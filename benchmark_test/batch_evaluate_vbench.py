#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch evaluation script: Evaluate all videos in vbench_videos directory
according to prompt_dimension_map.json
"""

import os
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Batch evaluate VBench videos')
    parser.add_argument(
        '--videos_dir',
        type=str,
        default='./vbench_videos',
        help='Video directory'
    )
    parser.add_argument(
        '--prompts_file',
        type=str,
        default='./selected_prompts_50.txt',
        help='Prompts file'
    )
    parser.add_argument(
        '--dimension_map_file',
        type=str,
        default='./prompt_dimension_map.json',
        help='Dimension mapping file'
    )
    parser.add_argument(
        '--vbench_dir',
        type=str,
        default='/home/qiuyao/commit/vbench/VBench',
        help='VBench project directory'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='./vbench_evaluation_results.json',
        help='Output results JSON file'
    )
    parser.add_argument(
        '--max_videos',
        type=int,
        default=None,
        help='Maximum number of videos to process (None means process all)'
    )
    return parser.parse_args()


def load_prompts(prompts_file):
    """Load prompts list"""
    prompts = []
    with open(prompts_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(line)
    return prompts


def load_dimension_map(dimension_map_file):
    """Load dimension mapping"""
    with open(dimension_map_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_video(video_path, prompt, dimensions, vbench_dir, output_dir):
    """Evaluate multiple dimensions for a single video"""
    results = {}

    # Convert paths to absolute paths
    video_path_abs = os.path.abspath(video_path)
    output_dir_abs = os.path.abspath(output_dir)

    vbench_eval_script = os.path.join(vbench_dir, 'evaluate.py')
    vbench_full_info = os.path.join(vbench_dir, 'vbench', 'VBench_full_info.json')

    if not os.path.exists(vbench_full_info):
        vbench_full_info = os.path.join(vbench_dir, 'VBench_full_info.json')

    vbench_full_info_abs = (
        os.path.abspath(vbench_full_info)
        if os.path.exists(vbench_full_info)
        else None
    )

    # Evaluate each dimension
    for dimension in dimensions:
        print(f"  Evaluating dimension: {dimension}")

        # Create separate output directory for each dimension
        dim_output_dir = os.path.join(output_dir_abs, dimension)
        os.makedirs(dim_output_dir, exist_ok=True)

        # Build evaluation command
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

        # Debug: print command (only in verbose mode)
        if os.environ.get('VBENCH_DEBUG', '').lower() == 'true':
            print(f"      Executing command: {' '.join(cmd)}")
            print(f"      Working directory: {vbench_dir}")

        try:
            # Set environment variables to avoid distributed issues (single GPU evaluation)
            env = os.environ.copy()
            env.setdefault('MASTER_ADDR', 'localhost')
            env.setdefault('MASTER_PORT', '29500')
            env.setdefault('RANK', '0')
            env.setdefault('LOCAL_RANK', '0')
            env.setdefault('WORLD_SIZE', '1')

            # Switch to VBench directory to execute
            result = subprocess.run(
                cmd,
                cwd=vbench_dir,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                env=env
            )

            if result.returncode == 0:
                # Find evaluation result files
                result_files = list(Path(dim_output_dir).glob("*_eval_results.json"))
                if result_files:
                    latest_result = max(result_files, key=os.path.getmtime)
                    with open(latest_result, 'r', encoding='utf-8') as f:
                        eval_data = json.load(f)
                        if dimension in eval_data:
                            score_data = eval_data[dimension]
                            # VBench result format: [score_value, [detailed_result_list]]
                            if isinstance(score_data, list) and len(score_data) > 0:
                                # First element is the score
                                score = score_data[0]
                                if isinstance(score, (int, float)):
                                    results[dimension] = float(score)
                                    print(f"Success on {dimension}: {results[dimension]:.4f}")
                                else:
                                    print(f"Failed on {dimension}: Invalid score format")
                                    results[dimension] = None
                            elif isinstance(score_data, (int, float)):
                                results[dimension] = float(score_data)
                                print(f"Success on {dimension}: {results[dimension]:.4f}")
                            elif isinstance(score_data, dict):
                                # Try to extract score from dictionary
                                if 'video_results' in score_data:
                                    video_results = score_data['video_results']
                                    if isinstance(video_results, list) and len(video_results) > 0:
                                        first = video_results[0]
                                        if isinstance(first, dict) and 'video_results' in first:
                                            results[dimension] = float(first['video_results'])
                                            print(f"Success on {dimension}: {results[dimension]:.4f}")
                                        else:
                                            results[dimension] = None
                                    else:
                                        results[dimension] = None
                                else:
                                    results[dimension] = None
                            else:
                                print(f"Failed on {dimension}: Unable to parse score format")
                                results[dimension] = None
                        else:
                            print(f"Failed on {dimension}: Dimension not found in results")
                            results[dimension] = None
                else:
                    print(f"Failed on {dimension}: Result file not found")
                    results[dimension] = None
            else:
                print(f"Failed on {dimension}: Evaluation failed")
                print(f"      Return code: {result.returncode}")
                if result.stderr:
                    print(f"      Error output (full):")
                    print("      " + "\n      ".join(result.stderr.split('\n')))
                if result.stdout:
                    print(f"      Standard output:")
                    print("      " + "\n      ".join(result.stdout.split('\n')))
                results[dimension] = None

        except subprocess.TimeoutExpired as e:
            print(f"Failed on {dimension}: Evaluation timeout (exceeded 10 minutes)")
            if hasattr(e, 'stdout') and e.stdout:
                print(f"      Output: {e.stdout[:500]}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"      Error: {e.stderr[:500]}")
            results[dimension] = None
        except Exception as e:
            print(f"Failed on {dimension}: Evaluation exception - {type(e).__name__}: {e}")
            import traceback
            print(f"      Detailed error:")
            print("      " + "\n      ".join(traceback.format_exc().split('\n')[:10]))
            results[dimension] = None

    return results


def main():
    args = parse_args()

    print("=" * 60)
    print("Batch Evaluate VBench Videos")
    print("=" * 60)
    print(f"Video directory: {args.videos_dir}")
    print(f"Prompts file: {args.prompts_file}")
    print(f"Dimension mapping file: {args.dimension_map_file}")
    print(f"Output file: {args.output_file}")
    print()

    # Load data
    print("Loading data...")
    prompts = load_prompts(args.prompts_file)
    dimension_map = load_dimension_map(args.dimension_map_file)

    print(f"Loaded {len(prompts)} prompts")
    print(f"Loaded {len(dimension_map)} dimension mappings")
    print()

    # Create output directory
    output_dir = os.path.join(os.path.dirname(args.output_file), 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)

    # Evaluation results
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'total_videos': len(prompts),
        'results': []
    }

    # Process each video
    max_videos = args.max_videos if args.max_videos else len(prompts)
    total_to_process = min(max_videos, len(prompts))

    for idx, prompt in enumerate(prompts[:total_to_process], start=1):
        video_filename = f"prompt_{idx}.mp4"
        video_path = os.path.join(args.videos_dir, video_filename)

        if not os.path.exists(video_path):
            print(f"[{idx}/{total_to_process}] Skipped: {video_filename} (file not found)")
            continue

        # Get dimensions for this prompt
        dimensions = dimension_map.get(prompt, [])
        if not dimensions:
            print(f"[{idx}/{total_to_process}] Skipped: {video_filename} (dimension mapping not found)")
            continue

        print(f"[{idx}/{total_to_process}] Evaluating: {video_filename}")
        print(f"  Prompt: {prompt[:60]}...")
        print(f"  Dimensions: {', '.join(dimensions)}")

        # Evaluate video
        video_results = evaluate_video(
            video_path, prompt, dimensions,
            args.vbench_dir, output_dir
        )

        # Record results
        all_results['results'].append({
            'video_file': video_filename,
            'video_path': os.path.abspath(video_path),
            'prompt': prompt,
            'dimensions': dimensions,
            'scores': video_results
        })

        print()

    # Save results
    print("=" * 60)
    print("Saving evaluation results...")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {args.output_file}")

    # Statistics
    print()
    print("=" * 60)
    print("Evaluation Statistics")
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

    print(f"Total prompts: {len(prompts)}")
    print(f"Processed: {total_to_process}")
    print(f"Evaluated: {total_evaluated}")
    print()
    print("Dimension statistics:")
    for dim, stats in sorted(dimension_stats.items()):
        print(f"  {dim}: Success {stats['success']}/{stats['total']}")

    print()
    print("Evaluation completed!")


if __name__ == "__main__":
    main()

