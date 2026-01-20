"""
Natural language interface for querying the dataset database
"""

import anthropic
import os
from database import DatasetDatabase
from dotenv import load_dotenv

load_dotenv()


def chat_query(question: str, db_path: str = "datasets.db"):
    """Ask questions about the database in natural language"""

    db = DatasetDatabase(db_path)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    client = anthropic.Anthropic(api_key=api_key)

    # Get schema info
    schema = """
    Database Schema:

    Table: datasets
    - dataset_id (TEXT): Unique identifier like 'username/dataset-name'
    - discovered_at (TIMESTAMP): When dataset was found
    - description (TEXT): Dataset description
    - readme (TEXT): README content
    - downloads (INTEGER): Number of downloads
    - likes (INTEGER): Number of likes
    - num_rows (INTEGER): Total number of rows/examples in dataset
    - download_size (INTEGER): Download size in bytes
    - dataset_size (INTEGER): Dataset size in bytes
    - created_at (TEXT): When dataset was created on Hugging Face
    - last_modified (TEXT): When dataset was last updated
    - author (TEXT): Dataset author/creator
    - heuristic_score (REAL): Quality score
    - llm_evaluation (TEXT): Full LLM evaluation text (contains RELEVANCE, QUALITY, etc.)
    - features (TEXT): JSON object of dataset features/columns
    - splits (TEXT): JSON object of train/test/val splits with row counts
    - structured_tags (TEXT): JSON object of ALL structured tags (e.g., {"task_categories": "image-classification", "license": "apache-2.0", "modality": "image", "format": "imagefolder", "language": "en", "size_categories": "10K<n<100K", "doi": "10.57967/hf/5418", "library": "datasets", "region": "us"})

    Table: dataset_tags
    - dataset_id (TEXT): Links to datasets.dataset_id
    - tag (TEXT): Tag like 'deepfake', 'computer-vision', etc.

    Note: Use JOIN when filtering by tags. JSON fields use json_extract() for querying.
    """

    prompt = f"""{schema}

User question: {question}

Generate a SQLite query to answer this question. Return ONLY the SQL query, nothing else.
IMPORTANT:
- This is SQLite, NOT PostgreSQL
- Use LIKE for text search (e.g., WHERE description LIKE '%text%')
- No regex functions or :: casting
- Join dataset_tags when filtering by tags
- The llm_evaluation field is plain text - use LIKE to search it
"""

    # Get SQL from Claude
    message = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    sql = message.content[0].text.strip()
    # Clean markdown code blocks if present
    if sql.startswith("```"):
        sql = sql.split("```")[1]
        if sql.startswith("sql"):
            sql = sql[3:]
        sql = sql.strip()

    print(f"🔍 Generated SQL:\n{sql}\n")

    # Execute SQL
    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()

    # Format results
    if not results:
        print("No results found.")
        return []

    print(f"📊 Found {len(results)} results:\n")
    for row in results:
        print(dict(row))
        print("-" * 60)

    return [dict(row) for row in results]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python chat_database.py 'your question here'")
        print("Example: python chat_database.py 'show all datasets with more than 1000 downloads'")
        sys.exit(1)

    question = ' '.join(sys.argv[1:])
    chat_query(question)
