import sqlite3
from datetime import datetime


def init_db():
    conn = sqlite3.connect('waste_detection.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            waste_type TEXT NOT NULL,
            confidence REAL NOT NULL,
            detection_date TIMESTAMP NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


def save_detection(waste_type: str, confidence: float):
    conn = sqlite3.connect('waste_detection.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO detections (waste_type, confidence, detection_date)
        VALUES (?, ?, ?)
    ''', (waste_type, confidence, datetime.now()))
    conn.commit()
    conn.close()


def get_all_detections():
    conn = sqlite3.connect('waste_detection.db')
    c = conn.cursor()
    c.execute('SELECT * FROM detections ORDER BY detection_date DESC')
    detections = c.fetchall()
    conn.close()
    return detections
