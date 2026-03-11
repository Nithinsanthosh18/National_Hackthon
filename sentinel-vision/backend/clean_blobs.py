import sqlite3
import os

db_path = "sentinel.db"
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    print("Checking for binary data in tables...")
    
    for table in ["events", "cameras", "watchlist"]:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        
        for col in columns:
            cursor.execute(f"SELECT id FROM {table} WHERE typeof({col}) = 'blob'")
            blobs = cursor.fetchall()
            if blobs:
                print(f"Found BLOB in {table}.{col} at IDs: {blobs}")
                # Convert blob to text or clear it
                cursor.execute(f"UPDATE {table} SET {col} = '' WHERE typeof({col}) = 'blob'")
    
    conn.commit()
    conn.close()
    print("Database cleanup complete.")
else:
    print("Database not found.")
