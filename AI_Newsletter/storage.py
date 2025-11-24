# AI_Newsletter/storage.py
import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List

DB_PATH = Path(__file__).parent / "newsletter.db"

def get_db():
    return sqlite3.connect(DB_PATH)

# ---------------- User Profiles ----------------

def load_user_profile(email: str) -> Dict[str, Any]:
    with get_db() as conn:
        cur = conn.execute(
            "SELECT profile_json FROM users WHERE email = ?",
            (email,),
        )
        row = cur.fetchone()
    if row:
        return json.loads(row[0])
    return {"email": email}

def save_user_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    email = profile.get("email")
    if not email:
        raise ValueError("Profile must contain 'email'")

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO users (email, profile_json)
            VALUES (?, ?)
            ON CONFLICT(email) DO UPDATE SET 
                profile_json = excluded.profile_json,
                updated_at = CURRENT_TIMESTAMP
            """,
            (email, json.dumps(profile)),
        )
    return {"ok": True}

# ---------------- Article Index ----------------

def get_seen_article_ids(user_email: str) -> List[str]:
    with get_db() as conn:
        cur = conn.execute(
            "SELECT article_id FROM article_index WHERE user_email = ?",
            (user_email,),
        )
        return [row[0] for row in cur.fetchall()]

def save_article_for_user(user_email: str, article: Dict[str, Any]) -> None:
    with get_db() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO article_index
            (user_email, article_id, url, published_at, title, summary)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user_email,
                article["id"],
                article["url"],
                article.get("published_at"),
                article.get("title"),
                article.get("summary", None),
            ),
        )
