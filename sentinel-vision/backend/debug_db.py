import sqlite3
import os

db_path = 'sentinel.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    print("Tables:", cursor.fetchall())
    
    # Check if rules table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='rules'")
    if cursor.fetchone():
        print("Table 'rules' exists.")
        cursor.execute("PRAGMA table_info(rules)")
        print("Columns in 'rules':", cursor.fetchall())
    else:
        print("Table 'rules' does NOT exist.")
else:
    print(f"{db_path} not found.")
