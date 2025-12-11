#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uniformly select 50 prompts from VBench prompts, ensuring coverage of multiple dimensions
"""

import os
from pathlib import Path
from collections import defaultdict

# Define dimensions to cover and their corresponding prompt files
# Prioritize dimensions supported by custom_input
DIMENSION_PROMPT_MAP = {
    # Officially supported dimensions
    'subject_consistency': 'subject_consistency.txt',
    'background_consistency': 'scene.txt',  # background_consistency uses scene prompts
    'motion_smoothness': 'subject_consistency.txt',  # Uses subject_consistency prompts
    'dynamic_degree': 'subject_consistency.txt',  # Uses subject_consistency prompts
    'aesthetic_quality': 'overall_consistency.txt',  # Uses overall_consistency prompts
    'imaging_quality': 'overall_consistency.txt',  # Uses overall_consistency prompts

    # Code-supported but undocumented dimensions
    'overall_consistency': 'overall_consistency.txt',
    'temporal_style': 'temporal_style.txt',
    'human_action': 'human_action.txt',
    'temporal_flickering': 'temporal_flickering.txt',
}

# Primary dimensions to cover (deduplicated)
PRIMARY_DIMENSIONS = [
    'subject_consistency',
    'scene',  # background_consistency uses this
    'overall_consistency',  # aesthetic_quality and imaging_quality use this
    'temporal_style',
    'human_action',
    'temporal_flickering',
]


def load_prompts_from_file(file_path):
    """Load prompts from file"""
    if not os.path.exists(file_path):
        print(f"Warning: File does not exist {file_path}")
        return []

    prompts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                prompts.append(line)
    return prompts


def select_prompts_uniformly(prompts, num_select):
    """Uniformly select prompts"""
    if len(prompts) <= num_select:
        return prompts

    # Calculate step size to ensure uniform distribution
    step = len(prompts) / num_select
    selected = []
    for i in range(num_select):
        idx = int(i * step)
        selected.append(prompts[idx])

    return selected

def main():
    # Set paths
    prompts_dir = Path('/home/qiuyao/commit/vbench/VBench/prompts/prompts_per_dimension')
    output_file = Path('/home/qiuyao/commit/Wan2.2/selected_prompts_50.txt')

    # Target: 50 prompts
    total_prompts = 50

    # Calculate how many prompts to select from each primary dimension
    num_dimensions = len(PRIMARY_DIMENSIONS)
    prompts_per_dimension = total_prompts // num_dimensions
    remainder = total_prompts % num_dimensions

    print(f"Target: Select {total_prompts} prompts")
    print(f"Cover {num_dimensions} primary dimensions")
    print(f"Approximately {prompts_per_dimension} prompts per dimension\n")

    selected_prompts = []
    dimension_stats = {}

    # Select prompts from each dimension
    for i, dimension in enumerate(PRIMARY_DIMENSIONS):
        # Last dimension gets the remainder
        num_to_select = prompts_per_dimension + (1 if i < remainder else 0)

        prompt_file = prompts_dir / f"{dimension}.txt"

        if not prompt_file.exists():
            print(f"Warning: {prompt_file} does not exist, skipping")
            continue

        # Load prompts
        all_prompts = load_prompts_from_file(prompt_file)

        if len(all_prompts) == 0:
            print(f"Warning: {dimension} has no prompts, skipping")
            continue

        # Uniformly select
        selected = select_prompts_uniformly(all_prompts, num_to_select)

        selected_prompts.extend(selected)
        dimension_stats[dimension] = len(selected)

        print(f"✓ {dimension}: Selected {len(selected)} from {len(all_prompts)} prompts")

    # If total is insufficient, supplement from the dimension with most prompts
    if len(selected_prompts) < total_prompts:
        needed = total_prompts - len(selected_prompts)
        print(f"\nNeed to supplement {needed} prompts")

        # Find dimension with most prompts
        max_dimension = max(dimension_stats.items(), key=lambda x: x[1])[0]
        prompt_file = prompts_dir / f"{max_dimension}.txt"
        all_prompts = load_prompts_from_file(prompt_file)

        # Exclude already selected prompts, then supplement
        selected_set = set(selected_prompts)
        available = [p for p in all_prompts if p not in selected_set]

        if len(available) >= needed:
            # Uniformly select supplements
            step = len(available) / needed
            for i in range(needed):
                idx = int(i * step)
                selected_prompts.append(available[idx])
            dimension_stats[max_dimension] += needed
            print(f"Supplemented {needed} from {max_dimension}")

    # Deduplicate (though theoretically there should be no duplicates)
    selected_prompts = list(dict.fromkeys(selected_prompts))

    # Save to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt in selected_prompts:
            f.write(prompt + '\n')

    print(f"\n{'='*60}")
    print(f"Successfully selected {len(selected_prompts)} prompts")
    print(f"Output file: {output_file}")
    print(f"{'='*60}")
    print("\nDimension distribution:")
    for dim, count in sorted(dimension_stats.items()):
        print(f"  {dim}: {count} prompts")

    # Display first 10 as examples
    print(f"\nFirst 10 prompt examples:")
    for i, prompt in enumerate(selected_prompts[:10], 1):
        print(f"  {i}. {prompt}")

if __name__ == "__main__":
    main()

