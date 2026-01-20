"""
Database module for storing dataset metadata and search results
"""

import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from contextlib import contextmanager


class DatasetDatabase:
    """SQLite database for storing Hugging Face dataset metadata"""

    def __init__(self, db_path: str = "datasets.db"):
        """
        Initialize database connection

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.init_database()

    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_database(self):
        """Create database tables if they don't exist"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Main datasets table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT UNIQUE NOT NULL,
                    discovered_at TIMESTAMP NOT NULL,

                    -- Metadata fields
                    description TEXT,
                    readme TEXT,
                    downloads INTEGER DEFAULT 0,
                    likes INTEGER DEFAULT 0,
                    num_rows INTEGER,
                    download_size INTEGER,
                    dataset_size INTEGER,
                    created_at TEXT,
                    last_modified TEXT,
                    author TEXT,

                    -- Evaluation scores
                    heuristic_score REAL,
                    llm_evaluation TEXT,

                    -- Additional metadata stored as JSON
                    features TEXT,
                    splits TEXT,
                    structured_tags TEXT,
                    siblings TEXT
                )
            """)

            # Tags table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dataset_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    UNIQUE(dataset_id, tag),
                    FOREIGN KEY (dataset_id) REFERENCES datasets (dataset_id) ON DELETE CASCADE
                )
            """)

            print(f"✅ Database initialized: {self.db_path}")

    def insert_dataset(self, record: Dict[str, Any]) -> int:
        """
        Insert or update a dataset record

        Args:
            record: Dictionary containing dataset information

        Returns:
            Row ID of inserted/updated record
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Extract metadata
            metadata = record.get('metadata', {})

            # Prepare data
            dataset_id = record['dataset_id']
            discovered_at = record.get('discovered_at', datetime.now().isoformat())

            description = metadata.get('description', '')
            readme = metadata.get('readme', '')
            downloads = metadata.get('downloads', 0)
            likes = metadata.get('likes', 0)
            num_rows = metadata.get('num_rows', None)
            download_size = metadata.get('download_size', None)
            dataset_size = metadata.get('dataset_size', None)
            created_at = metadata.get('created_at', None)
            last_modified = metadata.get('last_modified', None)
            author = metadata.get('author', '')

            heuristic_score = record.get('heuristic_score', 0.0)
            llm_evaluation = record.get('llm_evaluation', '')

            # JSON fields
            features = json.dumps(metadata.get('features', {}))
            splits = json.dumps(metadata.get('splits', {}))
            structured_tags = json.dumps(metadata.get('structured_tags', {}))
            siblings = json.dumps(metadata.get('siblings', []))

            # Insert or replace dataset
            cursor.execute("""
                INSERT OR REPLACE INTO datasets (
                    dataset_id, discovered_at,
                    description, readme, downloads, likes, num_rows, download_size, dataset_size, created_at, last_modified, author,
                    heuristic_score, llm_evaluation,
                    features, splits, structured_tags, siblings
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_id, discovered_at,
                description, readme, downloads, likes, num_rows, download_size, dataset_size, created_at, last_modified, author,
                heuristic_score, llm_evaluation,
                features, splits, structured_tags, siblings
            ))

            row_id = cursor.lastrowid

            # Insert tags
            tags = metadata.get('tags', [])
            for tag in tags:
                cursor.execute("INSERT OR IGNORE INTO dataset_tags (dataset_id, tag) VALUES (?, ?)", (dataset_id, tag))

            print(f"✅ Inserted dataset: {dataset_id}")
            return row_id

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a dataset by ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,))
            row = cursor.fetchone()

            if not row:
                return None

            # Get tags
            cursor.execute("SELECT tag FROM dataset_tags WHERE dataset_id = ?", (dataset_id,))
            tags = [r['tag'] for r in cursor.fetchall()]

            return {
                'dataset_id': row['dataset_id'],
                'discovered_at': row['discovered_at'],
                'description': row['description'],
                'readme': row['readme'],
                'downloads': row['downloads'],
                'likes': row['likes'],
                'heuristic_score': row['heuristic_score'],
                'llm_evaluation': row['llm_evaluation'],
                'card_data': json.loads(row['card_data']),
                'siblings': json.loads(row['siblings']),
                'tags': tags
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Total datasets
            cursor.execute("SELECT COUNT(*) as count FROM datasets")
            total_datasets = cursor.fetchone()['count']

            # Average heuristic score
            cursor.execute("SELECT AVG(heuristic_score) as avg_score FROM datasets")
            avg_score = cursor.fetchone()['avg_score'] or 0.0

            # Total downloads
            cursor.execute("SELECT SUM(downloads) as total FROM datasets")
            total_downloads = cursor.fetchone()['total'] or 0

            # Top tags
            cursor.execute("SELECT tag, COUNT(*) as count FROM dataset_tags GROUP BY tag ORDER BY count DESC LIMIT 10")
            top_tags = [dict(r) for r in cursor.fetchall()]

            return {
                'total_datasets': total_datasets,
                'average_heuristic_score': avg_score,
                'total_downloads': total_downloads,
                'top_tags': top_tags
            }

    def export_to_json(self, output_file: str = "database_export.json"):
        """Export all datasets to JSON"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT dataset_id FROM datasets")
            dataset_ids = [row['dataset_id'] for row in cursor.fetchall()]

            datasets = []
            for dataset_id in dataset_ids:
                dataset = self.get_dataset(dataset_id)
                if dataset:
                    datasets.append(dataset)

            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(datasets, f, indent=2)

            print(f"✅ Exported {len(datasets)} datasets to {output_file}")
