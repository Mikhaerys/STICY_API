import sqlite3
from datetime import datetime

# Initialize the SQLite database and create tables if they do not exist


def init_db():
    conn = sqlite3.connect('waste_detection.db')
    c = conn.cursor()
    # Table for waste detection records
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            waste_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            detection_date TIMESTAMP NOT NULL
        )
    ''')

    # Table for waste bins status
    c.execute('''
        CREATE TABLE IF NOT EXISTS waste_bins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bin_type TEXT NOT NULL,
            is_full BOOLEAN NOT NULL,
            last_updated TIMESTAMP NOT NULL
        )
    ''')

    # Initialize waste bins if they don't exist
    c.execute('SELECT COUNT(*) FROM waste_bins')
    if c.fetchone()[0] == 0:
        bins = [
            ('Plastic', False),
            ('Paper', False),
            ('Medical', False)
        ]
        c.executemany('''
            INSERT INTO waste_bins (bin_type, is_full, last_updated)
            VALUES (?, ?, ?)
        ''', [(bin_type, is_full, datetime.now()) for bin_type, is_full in bins])

    conn.commit()
    conn.close()

# Save a new waste detection record to the database


def save_detection(waste_type: str, confidence: float):
    conn = sqlite3.connect('waste_detection.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO detections (waste_type, confidence, detection_date)
        VALUES (?, ?, ?)
    ''', (waste_type, confidence, datetime.now()))
    conn.commit()
    conn.close()

# Retrieve all waste detection records, ordered by most recent


def get_all_detections():
    conn = sqlite3.connect('waste_detection.db')
    c = conn.cursor()
    c.execute('SELECT * FROM detections ORDER BY detection_date DESC')
    detections = c.fetchall()
    conn.close()
    return detections

# Retrieve the most recent waste detection record


def get_last_detection():
    conn = sqlite3.connect('waste_detection.db')
    c = conn.cursor()
    c.execute('SELECT * FROM detections ORDER BY detection_date DESC LIMIT 1')
    detection = c.fetchone()
    conn.close()
    return detection

# Update the status of a specific waste bin


def update_bin_status(bin_type: str, is_full: bool):
    conn = sqlite3.connect('waste_detection.db')
    c = conn.cursor()
    c.execute('''
        UPDATE waste_bins 
        SET is_full = ?, last_updated = ?
        WHERE bin_type = ?
    ''', (is_full, datetime.now(), bin_type))
    conn.commit()
    conn.close()

# Retrieve the status of all waste bins


def get_bins_status():
    conn = sqlite3.connect('waste_detection.db')
    c = conn.cursor()
    c.execute('SELECT * FROM waste_bins')
    bins = c.fetchall()
    conn.close()
    return bins
