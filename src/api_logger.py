"""
API Call Logger

Tracks all detection API calls for continuous improvement.
Stores images and responses in SQLite database for analysis.
"""

import sqlite3
import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import logging
import shutil


class APILogger:
    """
    Log API calls to SQLite database with image storage.
    """
    
    def __init__(self, db_path: str = "logs/api_calls.db", images_dir: str = "logs/images"):
        """
        Initialize API logger.
        
        Args:
            db_path: Path to SQLite database file
            images_dir: Directory to store uploaded images
        """
        self.db_path = db_path
        self.images_dir = images_dir
        self.logger = logging.getLogger(__name__)
        
        # Create directories
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(images_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS api_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                image_path TEXT NOT NULL,
                image_hash TEXT NOT NULL,
                image_size_kb INTEGER,
                response_json TEXT NOT NULL,
                total_boxes INTEGER,
                boxes_with_labels INTEGER,
                orphan_labels INTEGER,
                barcodes_found INTEGER,
                qrcodes_found INTEGER,
                ocr_used INTEGER,
                processing_time_ms REAL,
                error TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes for common queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON api_calls(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_image_hash 
            ON api_calls(image_hash)
        """)
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"Database initialized at {self.db_path}")
    
    def _calculate_image_hash(self, image_data: bytes) -> str:
        """Calculate SHA256 hash of image data."""
        return hashlib.sha256(image_data).hexdigest()
    
    def _save_image(self, image_data: bytes, call_id: str) -> str:
        """
        Save image to disk.
        
        Args:
            image_data: Raw image bytes
            call_id: Unique call identifier
            
        Returns:
            Path to saved image
        """
        filename = f"{call_id}.jpg"
        filepath = os.path.join(self.images_dir, filename)
        
        with open(filepath, 'wb') as f:
            f.write(image_data)
        
        return filepath
    
    def log_api_call(
        self,
        image_data: bytes,
        response: Dict,
        processing_time_ms: float,
        error: Optional[str] = None
    ) -> int:
        """
        Log an API call to the database.
        
        Args:
            image_data: Original image bytes
            response: API response dictionary
            processing_time_ms: Processing time in milliseconds
            error: Error message if call failed
            
        Returns:
            Database row ID
        """
        timestamp = datetime.now().isoformat()
        call_id = f"{int(datetime.now().timestamp() * 1000)}"
        
        # Calculate image hash
        image_hash = self._calculate_image_hash(image_data)
        image_size_kb = len(image_data) / 1024
        
        # Save image
        image_path = self._save_image(image_data, call_id)
        
        # Extract summary stats from response
        summary = response.get('summary', {})
        total_boxes = summary.get('total_boxes', 0)
        boxes_with_labels = summary.get('boxes_with_labels', 0)
        orphan_labels = summary.get('orphan_labels', 0)
        barcodes_found = summary.get('barcodes_found', 0)
        qrcodes_found = summary.get('qrcodes_found', 0)
        ocr_used = summary.get('ocr_used', 0)
        
        # Insert into database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO api_calls (
                timestamp, image_path, image_hash, image_size_kb,
                response_json, total_boxes, boxes_with_labels,
                orphan_labels, barcodes_found, qrcodes_found,
                ocr_used, processing_time_ms, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            timestamp, image_path, image_hash, image_size_kb,
            json.dumps(response), total_boxes, boxes_with_labels,
            orphan_labels, barcodes_found, qrcodes_found,
            ocr_used, processing_time_ms, error
        ))
        
        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        self.logger.info(f"Logged API call {row_id}: {total_boxes} boxes, {processing_time_ms:.0f}ms")
        
        return row_id
    
    def get_stats(self, days: int = 30) -> Dict:
        """
        Get statistics for recent API calls.
        
        Args:
            days: Number of days to include
            
        Returns:
            Statistics dictionary
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get date threshold
        from datetime import timedelta
        threshold = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Total calls
        cursor.execute("""
            SELECT COUNT(*) FROM api_calls 
            WHERE timestamp >= ?
        """, (threshold,))
        total_calls = cursor.fetchone()[0]
        
        # Average processing time
        cursor.execute("""
            SELECT AVG(processing_time_ms) FROM api_calls 
            WHERE timestamp >= ? AND error IS NULL
        """, (threshold,))
        avg_time = cursor.fetchone()[0] or 0
        
        # Total boxes detected
        cursor.execute("""
            SELECT SUM(total_boxes) FROM api_calls 
            WHERE timestamp >= ?
        """, (threshold,))
        total_boxes = cursor.fetchone()[0] or 0
        
        # Success rate
        cursor.execute("""
            SELECT COUNT(*) FROM api_calls 
            WHERE timestamp >= ? AND error IS NULL
        """, (threshold,))
        successful_calls = cursor.fetchone()[0]
        success_rate = (successful_calls / total_calls * 100) if total_calls > 0 else 0
        
        # Detection method breakdown
        cursor.execute("""
            SELECT 
                SUM(barcodes_found) as barcodes,
                SUM(qrcodes_found) as qrcodes,
                SUM(ocr_used) as ocr
            FROM api_calls 
            WHERE timestamp >= ?
        """, (threshold,))
        methods = cursor.fetchone()
        
        conn.close()
        
        return {
            'period_days': days,
            'total_calls': total_calls,
            'successful_calls': successful_calls,
            'success_rate_percent': round(success_rate, 2),
            'avg_processing_time_ms': round(avg_time, 2),
            'total_boxes_detected': total_boxes,
            'barcodes_found': methods[0] or 0,
            'qrcodes_found': methods[1] or 0,
            'ocr_used': methods[2] or 0
        }
    
    def export_to_csv(self, output_path: str, days: int = 30):
        """
        Export API call logs to CSV.
        
        Args:
            output_path: Path to output CSV file
            days: Number of days to export
        """
        import csv
        from datetime import timedelta
        
        threshold = (datetime.now() - timedelta(days=days)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                id, timestamp, image_path, image_hash, image_size_kb,
                total_boxes, boxes_with_labels, orphan_labels,
                barcodes_found, qrcodes_found, ocr_used,
                processing_time_ms, error
            FROM api_calls
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """, (threshold,))
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'ID', 'Timestamp', 'Image Path', 'Image Hash', 'Size (KB)',
                'Boxes', 'Boxes w/ Labels', 'Orphan Labels',
                'Barcodes', 'QR Codes', 'OCR Used',
                'Processing Time (ms)', 'Error'
            ])
            writer.writerows(cursor.fetchall())
        
        conn.close()
        self.logger.info(f"Exported {days} days of logs to {output_path}")
    
    def cleanup_old_entries(self, days: int = 90):
        """
        Delete log entries older than specified days.
        
        Args:
            days: Keep entries from last N days, delete older
            
        Returns:
            Number of entries deleted
        """
        from datetime import timedelta
        
        threshold = (datetime.now() - timedelta(days=days)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get image paths to delete
        cursor.execute("""
            SELECT image_path FROM api_calls 
            WHERE timestamp < ?
        """, (threshold,))
        
        image_paths = [row[0] for row in cursor.fetchall()]
        
        # Delete database entries
        cursor.execute("""
            DELETE FROM api_calls 
            WHERE timestamp < ?
        """, (threshold,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        # Delete image files
        for image_path in image_paths:
            try:
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception as e:
                self.logger.warning(f"Could not delete image {image_path}: {e}")
        
        self.logger.info(f"Cleaned up {deleted_count} entries older than {days} days")
        
        return deleted_count
