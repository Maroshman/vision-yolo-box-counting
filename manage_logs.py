#!/usr/bin/env python3
"""
API Log Management Utility

Provides commands to view, export, and manage API call logs.
"""

import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.api_logger import APILogger


def view_stats(args):
    """View API statistics."""
    logger = APILogger()
    stats = logger.get_stats(days=args.days)
    
    print(f"\n📊 API Statistics (Last {args.days} days)")
    print("=" * 60)
    print(f"Total Calls:          {stats['total_calls']}")
    print(f"Successful Calls:     {stats['successful_calls']}")
    print(f"Success Rate:         {stats['success_rate_percent']}%")
    print(f"Avg Processing Time:  {stats['avg_processing_time_ms']:.0f} ms")
    print(f"\nDetection Metrics:")
    print(f"  Total Boxes:        {stats['total_boxes_detected']}")
    print(f"  Barcodes Found:     {stats['barcodes_found']}")
    print(f"  QR Codes Found:     {stats['qrcodes_found']}")
    print(f"  OCR Used:           {stats['ocr_used']} times")
    print("=" * 60 + "\n")


def export_logs(args):
    """Export logs to CSV."""
    logger = APILogger()
    logger.export_to_csv(args.output, days=args.days)
    print(f"✅ Exported {args.days} days of logs to {args.output}")


def cleanup(args):
    """Clean up old log entries."""
    logger = APILogger()
    
    if not args.force:
        response = input(f"⚠️  This will delete entries older than {args.days} days. Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    deleted = logger.cleanup_old_entries(days=args.days)
    print(f"✅ Cleaned up {deleted} entries older than {args.days} days")


def main():
    parser = argparse.ArgumentParser(
        description="Manage API call logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View last 7 days of statistics
  python manage_logs.py stats --days 7
  
  # Export last 30 days to CSV
  python manage_logs.py export --days 30 --output logs.csv
  
  # Clean up entries older than 90 days
  python manage_logs.py cleanup --days 90
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='View API statistics')
    stats_parser.add_argument('--days', type=int, default=30, help='Number of days to include')
    stats_parser.set_defaults(func=view_stats)
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export logs to CSV')
    export_parser.add_argument('--days', type=int, default=30, help='Number of days to export')
    export_parser.add_argument('--output', type=str, default='api_logs.csv', help='Output CSV file path')
    export_parser.set_defaults(func=export_logs)
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Delete old log entries')
    cleanup_parser.add_argument('--days', type=int, default=90, help='Keep entries from last N days')
    cleanup_parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')
    cleanup_parser.set_defaults(func=cleanup)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    args.func(args)


if __name__ == '__main__':
    main()
