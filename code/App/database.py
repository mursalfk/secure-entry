import psycopg2
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
            image BYTEA NOT NULL
        )
    """)
    conn.commit()
    cursor.close()
    conn.close()

def add_user(name, image_data):
    """Add a new user to the database."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (name, image) VALUES (%s, %s)", (name, psycopg2.Binary(image_data)))
    conn.commit()
    cursor.close()
    conn.close()

def get_all_users():
    """Retrieve all users from the database."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, image FROM users")
    users = cursor.fetchall()
    cursor.close()
    conn.close()

    # DEBUGGING: Print users to check
    print("Fetched Users:", users)

    return users

def delete_user(user_id):
    """Delete a user from the database."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
    conn.commit()
    cursor.close()
    conn.close()

def get_user_by_id(user_id):
    """Retrieve a user by ID."""
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    conn.close()
    return user

# Initialize database on first run
init_db()
