#!/usr/bin/env python3
"""
Script to summarize accuracy results from result.txt files organized by domain and task.

Usage:
    python summarize_results.py <base_folder>
    
Example:
    python3 summarize_results.py qwen3_8b_thinking_results/pyautogui/screenshot/Qwen/Qwen3-VL-8B-Thinking
    python3 summarize_results.py evocua_results/pyautogui/screenshot/EvoCUA-S2
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def read_result_file(filepath: Path) -> float:
    """Read a result.txt file and return the score as a float."""
    try:
        with open(filepath, 'r') as f:
            content = f.read().strip()
            if content:
                return float(content)
            else:
                print(f"Warning: Empty file {filepath}")
                return 0.0
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return 0.0


def analyze_results(base_folder: str) -> Dict:
    """
    Analyze all result.txt files in the given base folder.
    
    Returns:
        Dictionary containing statistics by domain and overall
    """
    base_path = Path(base_folder)
    
    if not base_path.exists():
        print(f"Error: Path {base_folder} does not exist")
        sys.exit(1)
    
    # Dictionary to store results by domain
    domain_results = defaultdict(list)
    domain_task_ids = defaultdict(list)
    
    # Walk through all subdirectories
    for domain_dir in base_path.iterdir():
        if not domain_dir.is_dir():
            continue
        
        domain_name = domain_dir.name
        
        # Find all result.txt files in task subdirectories
        for task_dir in domain_dir.iterdir():
            if not task_dir.is_dir():
                continue
            
            result_file = task_dir / "result.txt"
            if result_file.exists():
                score = read_result_file(result_file)
                domain_results[domain_name].append(score)
                domain_task_ids[domain_name].append(task_dir.name)
    
    # Calculate statistics
    stats = {
        'base_folder': base_folder,
        'domains': {},
        'overall': {}
    }
    
    all_scores = []
    
    for domain, scores in domain_results.items():
        if not scores:
            continue
        
        total_tasks = len(scores)
        successful_tasks = sum(1 for s in scores if s > 0)
        total_score = sum(scores)
        avg_score = total_score / total_tasks if total_tasks > 0 else 0
        accuracy = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        stats['domains'][domain] = {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'failed_tasks': total_tasks - successful_tasks,
            'total_score': round(total_score, 2),
            'average_score': round(avg_score, 4),
            'accuracy_percent': round(accuracy, 2),
            'task_ids': domain_task_ids[domain]
        }
        
        all_scores.extend(scores)
    
    # Overall statistics
    if all_scores:
        total_tasks = len(all_scores)
        successful_tasks = sum(1 for s in all_scores if s > 0)
        total_score = sum(all_scores)
        avg_score = total_score / total_tasks if total_tasks > 0 else 0
        accuracy = (successful_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        stats['overall'] = {
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'failed_tasks': total_tasks - successful_tasks,
            'total_score': round(total_score, 2),
            'average_score': round(avg_score, 4),
            'accuracy_percent': round(accuracy, 2)
        }
    
    return stats


def print_summary(stats: Dict):
    """Print a formatted summary of the statistics."""
    print("=" * 80)
    print(f"RESULTS SUMMARY")
    print("=" * 80)
    print(f"Base Folder: {stats['base_folder']}")
    print()
    
    # Print overall statistics
    if stats['overall']:
        print("OVERALL STATISTICS")
        print("-" * 80)
        overall = stats['overall']
        print(f"  Total Tasks:       {overall['total_tasks']}")
        print(f"  Successful Tasks:  {overall['successful_tasks']}")
        print(f"  Failed Tasks:      {overall['failed_tasks']}")
        print(f"  Total Score:       {overall['total_score']}")
        print(f"  Average Score:     {overall['average_score']}")
        print(f"  Accuracy:          {overall['accuracy_percent']}%")
        print()
    
    # Print domain-specific statistics
    if stats['domains']:
        print("STATISTICS BY DOMAIN")
        print("-" * 80)
        
        # Sort domains by name
        for domain in sorted(stats['domains'].keys()):
            domain_stats = stats['domains'][domain]
            print(f"\n{domain.upper()}:")
            print(f"  Total Tasks:       {domain_stats['total_tasks']}")
            print(f"  Successful Tasks:  {domain_stats['successful_tasks']}")
            print(f"  Failed Tasks:      {domain_stats['failed_tasks']}")
            print(f"  Total Score:       {domain_stats['total_score']}")
            print(f"  Average Score:     {domain_stats['average_score']}")
            print(f"  Accuracy:          {domain_stats['accuracy_percent']}%")
        
        print()
        print("-" * 80)
        
        # Print comparison table
        print("\nDOMAIN COMPARISON TABLE")
        print("-" * 80)
        print(f"{'Domain':<20} {'Total':<10} {'Success':<10} {'Failed':<10} {'Accuracy':<12}")
        print("-" * 80)
        
        for domain in sorted(stats['domains'].keys()):
            domain_stats = stats['domains'][domain]
            print(f"{domain:<20} {domain_stats['total_tasks']:<10} "
                  f"{domain_stats['successful_tasks']:<10} "
                  f"{domain_stats['failed_tasks']:<10} "
                  f"{domain_stats['accuracy_percent']:.2f}%")
        
        print("-" * 80)
    
    print()


def save_detailed_report(stats: Dict, output_file: str = None):
    """Save a detailed JSON report."""
    if output_file is None:
        base_folder_name = Path(stats['base_folder']).name
        output_file = f"results_summary_{base_folder_name}.json"
    
    # Remove task_ids from the JSON output to keep it cleaner
    stats_copy = json.loads(json.dumps(stats))
    for domain in stats_copy.get('domains', {}).values():
        if 'task_ids' in domain:
            del domain['task_ids']
    
    with open(output_file, 'w') as f:
        json.dump(stats_copy, f, indent=2)
    
    print(f"Detailed report saved to: {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python summarize_results.py <base_folder>")
        print("\nExample:")
        print("  python summarize_results.py qwen3_8b_instruct_results/pyautogui/screenshot/Qwen/Qwen3-VL-8B-Instruct")
        sys.exit(1)
    
    base_folder = sys.argv[1]
    
    # Analyze results
    print("Analyzing results...")
    stats = analyze_results(base_folder)
    
    # Print summary
    print_summary(stats)
    
    # Save detailed report
    save_detailed_report(stats)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
