import sqlite3
import os

db_path = 'sentinel.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Dropping old rules table...")
    cursor.execute("DROP TABLE IF EXISTS rules")
    
    print("Creating new rules table...")
    # Manually match models.py exactly
    cursor.execute("""
        CREATE TABLE rules (
            id VARCHAR PRIMARY KEY,
            name VARCHAR,
            category VARCHAR,
            target VARCHAR,
            cameras VARCHAR,
            confidenceThreshold FLOAT,
            alertSeverity VARCHAR,
            enabled INTEGER DEFAULT 1,
            description VARCHAR
        )
    """)
    
    conn.commit()
    print("Table 'rules' recreated successfully with correct schema.")
    
    cursor.execute("PRAGMA table_info(rules)")
    print("New columns in 'rules':", cursor.fetchall())
    conn.close()
else:
    print("sentinel.db not found.")
