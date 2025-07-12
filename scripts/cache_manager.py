#!/usr/bin/env python3
"""
Cache Management Utility

This script provides utilities to manage preprocessing caches for the deep learning sound classification project.
It includes commands to view cache statistics, cleanup old caches, and optimize cache usage.

Usage:
    python scripts/cache_manager.py stats [--cache-dir CACHE_DIR]
    python scripts/cache_manager.py cleanup [--cache-dir CACHE_DIR] [--max-age DAYS]
    python scripts/cache_manager.py optimize [--cache-dir CACHE_DIR] [--max-size SIZE_GB]
    python scripts/cache_manager.py benchmark [--cache-dir CACHE_DIR] [--mode MODE]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from datasets.preprocessing import (  # type: ignore
        get_cache_usage_report,
        cleanup_cache_by_age,
        PreprocessingCache,
        PreprocessingConfig,
        create_preprocessor
    )
except ImportError:
    print("Error: Could not import preprocessing modules. Make sure you're running from the project root.")
    sys.exit(1)
from tqdm import tqdm


def format_size(size_bytes: float) -> str:
    """Format file size in human-readable format."""
    if size_bytes < 1024:
        return f"{size_bytes:.1f} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def print_cache_stats(cache_dir: Path) -> None:
    """Print comprehensive cache statistics."""
    print(f"\n🔍 Cache Statistics for: {cache_dir}")
    print("=" * 60)
    
    # Get usage report
    report = get_cache_usage_report(cache_dir)
    
    # Overall statistics
    print(f"📊 Overall Statistics:")
    print(f"  Total Size: {format_size(report['total_size_mb'] * 1024 * 1024)}")
    print(f"  Total Files: {report['total_files']:,}")
    print(f"  Cache Directory: {report['cache_directory']}")
    
    # File types
    if report['file_types']:
        print(f"\n📁 File Types:")
        for ext, count in sorted(report['file_types'].items()):
            print(f"  {ext if ext else 'no extension'}: {count:,} files")
    
    # Subdirectories
    if report['subdirectories']:
        print(f"\n📂 Subdirectories:")
        for subdir, info in sorted(report['subdirectories'].items(), 
                                  key=lambda x: x[1]['size_mb'], reverse=True):
            print(f"  {subdir}: {format_size(info['size_mb'] * 1024 * 1024)} ({info['files']} files)")
    
    # Largest files
    if report['largest_files']:
        print(f"\n🔝 Largest Files:")
        for file_info in report['largest_files'][:5]:
            print(f"  {file_info['path']}: {format_size(file_info['size_mb'] * 1024 * 1024)}")
    
    # Age information
    if report['oldest_files']:
        print(f"\n⏰ Age Information:")
        current_time = time.time()
        oldest_age = current_time - report['oldest_files'][0]['modified_time']
        newest_age = current_time - report['newest_files'][-1]['modified_time']
        
        print(f"  Oldest file: {format_time(oldest_age)} ago")
        print(f"  Newest file: {format_time(newest_age)} ago")


def cleanup_cache(cache_dir: Path, max_age_days: int = 30) -> None:
    """Clean up old cache files."""
    print(f"\n🧹 Cleaning up cache files older than {max_age_days} days...")
    
    # Get before stats
    before_report = get_cache_usage_report(cache_dir)
    before_size = before_report['total_size_mb']
    before_files = before_report['total_files']
    
    # Cleanup
    cleanup_summary = cleanup_cache_by_age(cache_dir, max_age_days)
    
    # Get after stats
    after_report = get_cache_usage_report(cache_dir)
    after_size = after_report['total_size_mb']
    after_files = after_report['total_files']
    
    # Print results
    print(f"✅ Cleanup Complete!")
    print(f"  Files removed: {cleanup_summary['files_removed']:,}")
    print(f"  Space freed: {format_size(cleanup_summary['space_freed_mb'] * 1024 * 1024)}")
    print(f"  Before: {before_files:,} files, {format_size(before_size * 1024 * 1024)}")
    print(f"  After: {after_files:,} files, {format_size(after_size * 1024 * 1024)}")
    
    if cleanup_summary['errors']:
        print(f"⚠️  Errors encountered:")
        for error in cleanup_summary['errors'][:5]:
            print(f"    {error}")


def optimize_cache(cache_dir: Path, max_size_gb: float = 5.0) -> None:
    """Optimize cache by enforcing size limits."""
    print(f"\n⚡ Optimizing cache (max size: {max_size_gb}GB)...")
    
    # Get current stats
    report = get_cache_usage_report(cache_dir)
    current_size_gb = report['total_size_mb'] / 1024
    
    print(f"Current cache size: {current_size_gb:.2f}GB")
    
    if current_size_gb <= max_size_gb:
        print("✅ Cache size is within limits, no optimization needed.")
        return
    
    # Create a cache manager to enforce size limits
    cache_manager = PreprocessingCache(cache_dir, max_size_gb)
    
    # Trigger cleanup
    cache_manager.cleanup_all_caches(max_age_days=30)
    
    # Get final stats
    final_report = get_cache_usage_report(cache_dir)
    final_size_gb = final_report['total_size_mb'] / 1024
    
    print(f"✅ Optimization complete!")
    print(f"  Before: {current_size_gb:.2f}GB ({report['total_files']} files)")
    print(f"  After: {final_size_gb:.2f}GB ({final_report['total_files']} files)")
    print(f"  Space saved: {format_size((current_size_gb - final_size_gb) * 1024 * 1024 * 1024)}")


def benchmark_cache(cache_dir: Path, mode: str = "envnet_v2") -> None:
    """Benchmark cache performance with different configurations."""
    print(f"\n🏃 Benchmarking cache performance for {mode}...")
    
    # Test configurations
    test_configs = [
        {"sample_rate": 44100, "n_mels": 128, "window_length": 1.5},
        {"sample_rate": 44100, "n_mels": 64, "window_length": 1.0},
        {"sample_rate": 22050, "n_mels": 128, "window_length": 2.0},
    ]
    
    results = []
    
    for i, config_dict in enumerate(test_configs):
        print(f"\n  Testing configuration {i+1}/{len(test_configs)}...")
        
        # Create preprocessor
        try:
            preprocessor = create_preprocessor(
                mode=mode,
                config_dict=config_dict,
                base_cache_dir=cache_dir,
                force_rebuild=False
            )
            
            # Get cache stats
            cache_stats = preprocessor.get_cache_stats()
            performance_stats = preprocessor.get_performance_stats()
            
            config_result = {
                'config': config_dict,
                'cache_stats': cache_stats.to_dict() if cache_stats else None,
                'performance_stats': performance_stats
            }
            
            results.append(config_result)
            
            if cache_stats:
                print(f"    Cache hit rate: {cache_stats.hit_rate():.1%}")
                print(f"    Cache size: {format_size(cache_stats.cache_size_mb * 1024 * 1024)}")
                print(f"    Avg load time: {cache_stats.avg_load_time_ms:.1f}ms")
                print(f"    Avg save time: {cache_stats.avg_save_time_ms:.1f}ms")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({'config': config_dict, 'error': str(e)})
    
    # Print summary
    print(f"\n📊 Benchmark Summary:")
    print("-" * 40)
    
    for i, result in enumerate(results):
        print(f"Configuration {i+1}:")
        print(f"  Config: {result['config']}")
        
        if 'error' in result:
            print(f"  Status: ❌ Failed - {result['error']}")
        else:
            cache_stats = result.get('cache_stats', {})
            if cache_stats:
                print(f"  Hit rate: {cache_stats.get('hit_rate', 0):.1%}")
                print(f"  Load time: {cache_stats.get('avg_load_time_ms', 0):.1f}ms")
                print(f"  Save time: {cache_stats.get('avg_save_time_ms', 0):.1f}ms")
            else:
                print(f"  Status: ✅ No cache stats available")
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Cache Management Utility for Deep Learning Sound Classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View cache statistics
  python scripts/cache_manager.py stats

  # Clean up files older than 7 days
  python scripts/cache_manager.py cleanup --max-age 7

  # Optimize cache to 2GB max size
  python scripts/cache_manager.py optimize --max-size 2.0

  # Benchmark AST preprocessing
  python scripts/cache_manager.py benchmark --mode ast
        """
    )
    
    parser.add_argument(
        "command",
        choices=["stats", "cleanup", "optimize", "benchmark"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/cache"),
        help="Cache directory path (default: data/cache)"
    )
    
    parser.add_argument(
        "--max-age",
        type=int,
        default=30,
        help="Maximum age in days for cleanup (default: 30)"
    )
    
    parser.add_argument(
        "--max-size",
        type=float,
        default=5.0,
        help="Maximum cache size in GB for optimization (default: 5.0)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["envnet_v2", "ast"],
        default="envnet_v2",
        help="Preprocessing mode for benchmark (default: envnet_v2)"
    )
    
    args = parser.parse_args()
    
    # Ensure cache directory exists
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Execute command
    try:
        if args.command == "stats":
            print_cache_stats(args.cache_dir)
        elif args.command == "cleanup":
            cleanup_cache(args.cache_dir, args.max_age)
        elif args.command == "optimize":
            optimize_cache(args.cache_dir, args.max_size)
        elif args.command == "benchmark":
            benchmark_cache(args.cache_dir, args.mode)
        
        print("\n✅ Operation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 