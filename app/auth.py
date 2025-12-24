import hashlib

# Simple user database (In production, use a secure database)
USERS = {
    "doctor": {
        "password_hash": hashlib.sha256("password123".encode()).hexdigest(),
        "role": "doctor",
        "name": "Dr. Smith"
    },
    "nurse": {
        "password_hash": hashlib.sha256("nurse123".encode()).hexdigest(),
        "role": "nurse",
        "name": "Nurse Joy"
    },
    "patient": {
        "password_hash": hashlib.sha256("patient123".encode()).hexdigest(),
        "role": "patient",
        "name": "John Doe"
    }
}

def authenticate(username, password):
    """
    Verify credentials.
    Returns user dict if valid, None otherwise.
    """
    if username in USERS:
        input_hash = hashlib.sha256(password.encode()).hexdigest()
        if input_hash == USERS[username]["password_hash"]:
            return USERS[username]
    return None
