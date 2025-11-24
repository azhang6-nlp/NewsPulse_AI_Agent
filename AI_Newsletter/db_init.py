# AI_Newsletter/db_init.py
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "newsletter.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # user profiles
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        email TEXT PRIMARY KEY,
        profile_json TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # article index
    cur.execute("""
    CREATE TABLE IF NOT EXISTS article_index (
        user_email TEXT,
        article_id TEXT,
        url TEXT,
        published_at TEXT,
        title TEXT,
        summary TEXT,
        PRIMARY KEY (user_email, article_id)
    );
    """)

    conn.commit()
    conn.close()
    print("✅ SQLite initialized at:", DB_PATH)

if __name__ == "__main__":
    init_db()
