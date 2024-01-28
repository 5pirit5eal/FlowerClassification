CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    image_url TEXT NOT NULL UNIQUE,
    predicted_class TEXT NOT NULL,
    correct_prediction BOOLEAN NOT NULL 
);