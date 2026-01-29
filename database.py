"""
Database module for storing dataset metadata and search results
"""

import sqlite3
import json
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
import gspread
from google.oauth2.service_account import Credentials


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
                    siblings TEXT,

                    -- Scraped data
                    full_readme TEXT,
                    scraped_data TEXT,

                    -- Embedding (JSON array of floats)
                    embedding TEXT
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

    def _parse_num_rows(self, value) -> Optional[int]:
        """Parse num_rows string like '17.4k' into integer"""
        if not value:
            return None
        value = str(value).lower().replace(',', '').strip()
        try:
            if 'k' in value:
                return int(float(value.replace('k', '')) * 1000)
            elif 'm' in value:
                return int(float(value.replace('m', '')) * 1000000)
            return int(float(value))
        except:
            return None

    def insert_dataset(self, record: Dict[str, Any], embedding: Optional[List[float]] = None) -> int:
        """Insert or update a dataset record"""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            metadata = record.get('metadata', {})
            dataset_id = record['dataset_id']
            scraped = metadata.get('scraped_data', {}) or {}

            # Use scraped num_rows as fallback
            num_rows = metadata.get('num_rows') or self._parse_num_rows(scraped.get('num_rows'))

            # JSON fields
            features = json.dumps(metadata.get('features', {}))
            splits = json.dumps(metadata.get('splits', {}))
            structured_tags = json.dumps(metadata.get('structured_tags', {}))
            siblings = json.dumps(metadata.get('siblings', []))
            scraped_data = json.dumps(scraped) if scraped else None
            embedding_json = json.dumps(embedding) if embedding else None

            cursor.execute("""
                INSERT OR REPLACE INTO datasets (
                    dataset_id, discovered_at,
                    description, readme, downloads, likes, num_rows, download_size, dataset_size,
                    created_at, last_modified, author,
                    heuristic_score, llm_evaluation,
                    features, splits, structured_tags, siblings,
                    full_readme, scraped_data, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dataset_id,
                record.get('discovered_at'),
                metadata.get('description', ''),
                metadata.get('readme', ''),
                metadata.get('downloads', 0),
                metadata.get('likes', 0),
                num_rows,
                metadata.get('download_size'),
                metadata.get('dataset_size'),
                metadata.get('created_at'),
                metadata.get('last_modified'),
                metadata.get('author', ''),
                record.get('heuristic_score', 0.0),
                record.get('llm_evaluation', ''),
                features, splits, structured_tags, siblings,
                metadata.get('full_readme'),
                scraped_data,
                embedding_json
            ))

            row_id = cursor.lastrowid

            # Insert tags
            for tag in metadata.get('tags', []):
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

    def export_to_google_sheets(self, spreadsheet_id: str, credentials_file: str = "credentials.json"):
        """Export all datasets to Google Sheets

        Args:
            spreadsheet_id: The ID from the Google Sheets URL (between /d/ and /edit)
            credentials_file: Path to Google service account credentials JSON
        """
        # Authenticate with Google
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets',
            'https://www.googleapis.com/auth/drive'
        ]
        creds = Credentials.from_service_account_file(credentials_file, scopes=scopes)
        client = gspread.authorize(creds)

        # Open the spreadsheet
        spreadsheet = client.open_by_key(spreadsheet_id)

        # Get or create worksheet
        try:
            worksheet = spreadsheet.worksheet("Datasets")
            worksheet.clear()
        except gspread.WorksheetNotFound:
            worksheet = spreadsheet.add_worksheet(title="Datasets", rows=1000, cols=15)

        # Get data from database
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    dataset_id, author, description, downloads, likes, num_rows,
                    created_at, last_modified, heuristic_score, llm_evaluation,
                    discovered_at, scraped_data
                FROM datasets
                ORDER BY heuristic_score DESC, downloads DESC
            """)
            rows = cursor.fetchall()

        # Prepare data for sheets
        header = [
            'Dataset ID', 'URL', 'Author', 'Description', 'Downloads', 'Likes',
            'Num Rows', 'Created At', 'Last Modified', 'Heuristic Score',
            'Is Relevant', 'License', 'Citation', 'LLM Evaluation', 'Discovered At'
        ]

        data = [header]
        for row in rows:
            llm_eval = row['llm_evaluation'] or ''
            is_relevant = 'Yes' if 'DECISION:' in llm_eval and 'Yes' in llm_eval.split('DECISION:')[1].split('\n')[0] else 'No'
            scraped = json.loads(row['scraped_data']) if row['scraped_data'] else {}

            data.append([
                row['dataset_id'],
                f"https://huggingface.co/datasets/{row['dataset_id']}",
                row['author'] or '',
                (row['description'] or '')[:200],
                row['downloads'] or 0,
                row['likes'] or 0,
                row['num_rows'] or self._parse_num_rows(scraped.get('num_rows')) or '',
                row['created_at'] or '',
                row['last_modified'] or '',
                row['heuristic_score'] or 0,
                is_relevant,
                scraped.get('license', ''),
                (scraped.get('citation', '') or '')[:300],
                llm_eval[:500] if llm_eval else '',
                row['discovered_at'] or ''
            ])

        # Upload to Google Sheets
        worksheet.update(data, value_input_option='RAW')

        print(f"Exported {len(rows)} datasets to Google Sheets")
