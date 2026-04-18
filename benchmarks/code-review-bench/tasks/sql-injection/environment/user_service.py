"""User service — handles user lookup and authentication."""

import sqlite3


def get_db():
    conn = sqlite3.connect("users.db")
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users "
        "(id INTEGER PRIMARY KEY, username TEXT, password TEXT, role TEXT)"
    )
    return conn


def find_user(username: str) -> dict | None:
    db = get_db()
    # BUG: SQL injection — user input directly interpolated into query
    cursor = db.execute(f"SELECT * FROM users WHERE username = '{username}'")
    row = cursor.fetchone()
    if row:
        return {"id": row[0], "username": row[1], "password": row[2], "role": row[3]}
    return None


def authenticate(username: str, password: str) -> bool:
    user = find_user(username)
    if user and user["password"] == password:
        return True
    return False


def create_user(username: str, password: str, role: str = "user") -> int:
    db = get_db()
    cursor = db.execute(
        "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
        (username, password, role),
    )
    db.commit()
    return cursor.lastrowid
