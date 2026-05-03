import sqlite3

def query_session(session_id):
    try:
        conn = sqlite3.connect("app/database.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM detection_sessions WHERE id = ?", (session_id,))
        session = cursor.fetchone()
        conn.close()
        if session:
            print("Session found:", session)
        else:
            print("Session not found.")
    except Exception as e:
        print("Error querying session:", e)

if __name__ == "__main__":
    query_session(7)