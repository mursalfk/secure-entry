import psycopg2
import base64
from datetime import datetime
from config import DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

def connect_db():
    """Establish connection to PostgreSQL database."""
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

def init_db():
    """Initialize PostgreSQL database with a users table."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100) NOT NULL,
            username VARCHAR(100) UNIQUE NOT NULL,
            image BYTEA NOT NULL,
            last_entered TIMESTAMP DEFAULT NULL
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

def get_user_name_by_id(user_id):
    """Fetch the username from the database given an ID."""
    conn = connect_db()
    cursor = conn.cursor()
    
    # Ensure user_id is an integer (avoid numpy.int64 issue)
    user_id = int(user_id)
    print(f"User ID: {user_id}")

    cursor.execute("SELECT name FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    
    cursor.close()
    conn.close()

    return user[0] if user else None  # Return name if found, else None


def add_user(name, username, image_data):
    """Add a new user to the database."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name, username, image) VALUES (%s, %s, %s)", 
                   (name, username, psycopg2.Binary(image_data)))
    conn.commit()
    cursor.close()
    conn.close()

def get_all_users():
    """Retrieve all users from the database and convert image to Base64."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, username, image, last_entered FROM users")
    users = cursor.fetchall()
    cursor.close()
    conn.close()

    # Convert image to Base64 for HTML rendering
    users_processed = []
    for user in users:
        id, name, username, image_binary, last_entered = user
        image_b64 = base64.b64encode(image_binary).decode('utf-8')
        users_processed.append((id, name, username, image_b64, last_entered))

    return users_processed

def delete_user(username):
    """Delete a user from the database by username."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE username = %s", (username,))
    conn.commit()
    cursor.close()
    conn.close()

def get_user_by_username(username):
    """Retrieve a user by username."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, username FROM users WHERE username = %s", (username,))
    user = cursor.fetchone()
    conn.close()
    return user

def update_last_entered(username):
    """Update the last entered timestamp when a user logs in."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("UPDATE users SET last_entered = %s WHERE username = %s", (datetime.now(), username))
    conn.commit()
    cursor.close()
    conn.close()

# Initialize database on first run
init_db()
