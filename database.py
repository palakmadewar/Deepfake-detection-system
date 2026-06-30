import sqlite3
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), 'truth_shield.db')

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            filename TEXT NOT NULL,
            media_type TEXT NOT NULL,
            result TEXT NOT NULL,
            confidence REAL NOT NULL,
            details TEXT
        )
    ''')
    conn.commit()
    conn.close()

def add_record(filename, media_type, result, confidence, details):
    conn = get_db_connection()
    cursor = conn.cursor()
    details_str = json.dumps(details) if details else None
    cursor.execute('''
        INSERT INTO history (filename, media_type, result, confidence, details)
        VALUES (?, ?, ?, ?, ?)
    ''', (filename, media_type, result, confidence, details_str))
    record_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return record_id

def get_records(media_type=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    if media_type and media_type != 'all':
        cursor.execute('''
            SELECT id, timestamp, filename, media_type, result, confidence, details
            FROM history
            WHERE media_type = ?
            ORDER BY timestamp DESC
        ''', (media_type,))
    else:
        cursor.execute('''
            SELECT id, timestamp, filename, media_type, result, confidence, details
            FROM history
            ORDER BY timestamp DESC
        ''')
    rows = cursor.fetchall()
    conn.close()
    
    records = []
    for r in rows:
        record = dict(r)
        if record['details']:
            try:
                record['details'] = json.loads(record['details'])
            except Exception:
                pass
        records.append(record)
    return records

def get_record_by_id(record_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, timestamp, filename, media_type, result, confidence, details
        FROM history
        WHERE id = ?
    ''', (record_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        record = dict(row)
        if record['details']:
            try:
                record['details'] = json.loads(record['details'])
            except Exception:
                pass
        return record
    return None
