# app/database.py
import sqlite3
import os
import json
from datetime import datetime

DB_PATH = "app.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Users table - ADD role column
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS users
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       username
                       TEXT
                       UNIQUE
                       NOT
                       NULL,
                       password_hash
                       TEXT
                       NOT
                       NULL,
                       email
                       TEXT
                       UNIQUE,
                       role
                       TEXT
                       DEFAULT
                       'user', -- Add role column: 'user', 'dermatologist', 'admin'
                       created_at
                       DATETIME
                       DEFAULT
                       CURRENT_TIMESTAMP
                   )
                   ''')

    # Predictions table (stores analysis results)
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS predictions
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       user_id
                       INTEGER,
                       image_filename
                       TEXT
                       NOT
                       NULL,
                       predicted_class
                       TEXT
                       NOT
                       NULL,
                       confidence
                       REAL
                       NOT
                       NULL,
                       all_probabilities
                       TEXT
                       NOT
                       NULL, -- Stored as JSON string
                       recommendation
                       TEXT
                       NOT
                       NULL,
                       needs_appointment
                       BOOLEAN
                       NOT
                       NULL,
                       timestamp
                       DATETIME
                       DEFAULT
                       CURRENT_TIMESTAMP,
                       FOREIGN
                       KEY
                   (
                       user_id
                   ) REFERENCES users
                   (
                       id
                   )
                       )
                   ''')

    # Appointments table (stores booking information) - ADD notes_from_dermatologist column
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS appointments
                   (
                       id
                       INTEGER
                       PRIMARY
                       KEY
                       AUTOINCREMENT,
                       user_id
                       INTEGER,   -- Patient/user who booked
                       prediction_id
                       INTEGER,   -- Related analysis
                       scheduled_date
                       DATETIME
                       NOT
                       NULL,
                       status
                       TEXT
                       DEFAULT
                       'pending', -- pending, confirmed, cancelled
                       notes
                       TEXT,
                       notes_from_dermatologist
                       TEXT
                       DEFAULT
                       '',        -- NEW COLUMN: Notes from dermatologist
                       created_at
                       DATETIME
                       DEFAULT
                       CURRENT_TIMESTAMP,
                       FOREIGN
                       KEY
                   (
                       user_id
                   ) REFERENCES users
                   (
                       id
                   ),
                       FOREIGN KEY
                   (
                       prediction_id
                   ) REFERENCES predictions
                   (
                       id
                   )
                       )
                   ''')

    conn.commit()
    conn.close()
    print(f"Database {DB_PATH} initialized.")


def get_user_by_username(username):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, password_hash, role FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    return user


def create_user(username, password_hash, email=None, role='user'):  # Add role parameter
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO users (username, password_hash, email, role) VALUES (?, ?, ?, ?)',
                       (username, password_hash, email, role))  # Include role in insert
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False  # Username or email already exists


def save_prediction_result(user_id, image_filename, result):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
                   INSERT INTO predictions (user_id, image_filename, predicted_class, confidence, all_probabilities,
                                            recommendation, needs_appointment)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ''', (
                       user_id, image_filename, result['predicted_class'], result['confidence'],
                       json.dumps(result['all_predictions']), result['recommendation'], result['needs_appointment']
                   ))
    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return prediction_id  # Return ID for potential appointment booking


def book_appointment(user_id, prediction_id, scheduled_datetime, notes=""):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
                   INSERT INTO appointments (user_id, prediction_id, scheduled_date, notes, status)
                   VALUES (?, ?, ?, ?, ?)
                   ''', (user_id, prediction_id, scheduled_datetime, notes, 'pending'))
    conn.commit()
    conn.close()


def get_user_appointments(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
                   SELECT a.id,
                          a.scheduled_date,
                          a.status,
                          a.notes,
                          a.notes_from_dermatologist,
                          p.predicted_class,
                          p.confidence,
                          p.timestamp as prediction_time
                   FROM appointments a
                            LEFT JOIN predictions p ON a.prediction_id = p.id
                   WHERE a.user_id = ?
                   ORDER BY a.scheduled_date ASC
                   ''', (user_id,))
    appointments = cursor.fetchall()
    conn.close()
    return appointments


# --- NEW FUNCTION: Get ALL appointments for Dermatologist/Admin ---
def get_all_appointments():
    """
    Retrieve all appointments for Dermatologist or Admin view.
    Joins appointments with predictions and users to get patient info.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
                   SELECT a.id,
                          u.username  as patient_username,
                          a.scheduled_date,
                          a.status,
                          a.notes,
                          a.notes_from_dermatologist,
                          p.predicted_class,
                          p.confidence,
                          p.timestamp as prediction_time
                   FROM appointments a
                            JOIN users u ON a.user_id = u.id
                            LEFT JOIN predictions p ON a.prediction_id = p.id
                   ORDER BY a.scheduled_date ASC
                   ''')
    appointments = cursor.fetchall()
    conn.close()
    return appointments


# --- NEW FUNCTION: Update appointment status and notes (for Dermatologist/Admin) ---
def update_appointment_status_and_notes(appointment_id, new_status, dermatologist_notes):
    """
    Update an appointment's status and add notes from the dermatologist.
    """
    if new_status not in ['pending', 'confirmed', 'cancelled']:  # Add more statuses if needed
        print(f"Invalid status: {new_status}")
        return False

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('UPDATE appointments SET status = ?, notes_from_dermatologist = ? WHERE id = ?',
                       (new_status, dermatologist_notes, appointment_id))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Error updating appointment: {e}")
        conn.close()
        return False


# --- NEW FUNCTION: Get ALL users for Admin ---
def get_all_users():
    """
    Retrieve all users for Admin management.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
                   SELECT id, username, email, role, created_at
                   FROM users
                   ORDER BY created_at DESC
                   ''')
    users = cursor.fetchall()
    conn.close()
    return users


# --- NEW FUNCTION: Update user role for Admin ---
def update_user_role(user_id, new_role):
    """
    Update a user's role (for Admin).
    """
    if new_role not in ['user', 'dermatologist', 'admin']:
        print(f"Invalid role: {new_role}")
        return False

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('UPDATE users SET role = ? WHERE id = ?', (new_role, user_id))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Error updating user role: {e}")
        conn.close()
        return False


# --- NEW FUNCTION: Delete user for Admin ---
def delete_user(user_id):
    """
    Delete a user (for Admin).
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        # Deleting user might violate foreign key constraints in predictions/appointments.
        # Consider soft deletion or cascading deletes if needed.
        # For now, let's just try deleting the user.
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"Error deleting user: {e}")
        conn.close()
        return False


def get_user_predictions(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
                   SELECT id, image_filename, predicted_class, confidence, recommendation, needs_appointment, timestamp
                   FROM predictions
                   WHERE user_id = ?
                   ORDER BY timestamp DESC
                   ''', (user_id,))
    predictions = cursor.fetchall()
    conn.close()
    return predictions


if __name__ == "__main__":
    init_db()