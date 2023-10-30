import sqlite3


def create_database():
    conn = sqlite3.connect("summary_cache.db")
    cursor = conn.cursor()
    cursor.execute(
        """
                    CREATE TABLE IF NOT EXISTS summaries(
                        book_id TEXT PRIMARY KEY,
                        summary TEXT
                    )
                """
    )
    conn.commit()
    conn.close()


def get_summary_from_db(book_id):
    conn = sqlite3.connect("summary_cache.db")
    cursor = conn.cursor()
    cursor.execute("SELECT summary FROM summaries WHERE book_id=?", (book_id,))
    result = cursor.fetchone()
    summary = result[0] if result else None
    conn.close()
    return summary


def store_summary(book_id, summary):
    conn = sqlite3.connect("summary_cache.db")
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO summaries (book_id, summary) VALUES(?,?)",
        (book_id, summary),
    )
    conn.commit()
    conn.close()
