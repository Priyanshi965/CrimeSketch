import sqlite3
import random
import os

# Forensic Data Samples
CITIES = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Kolkata", "Hyderabad", "Chennai", "Ahmedabad", "Gurugram"]
CRIMES = [
    "Cyber Fraud", "Identity Theft", "Burglary", "Money Laundering", 
    "Assault", "Grand Theft Auto", "Narcotics Distribution", "Forgery",
    "Armed Robbery", "Extortion", "Public Disturbance", "Traffic Violation"
]
RISKS = ["low", "medium", "high"]

# Moral Citizen Profile (Low risk, minor or no crime)
MORAL_CRIMES = ["Traffic Violation", "Public Disturbance", "None"]

def seed_database(db_path):
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT id, name FROM suspects")
    suspects = cursor.fetchall()

    print(f"Found {len(suspects)} suspects. Seeding forensic metadata...")

    for suspect_id, name in suspects:
        risk = random.choice(RISKS)
        city = random.choice(CITIES)
        
        if risk == "low":
            crime = random.choice(MORAL_CRIMES)
        elif risk == "medium":
            crime = random.choice(CRIMES[:6]) # Less violent
        else:
            crime = random.choice(CRIMES[4:10]) # High danger

        cursor.execute('''
            UPDATE suspects 
            SET city = ?, crime_type = ?, risk_level = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (city, crime, risk, suspect_id))

    conn.commit()
    conn.close()
    print("Database seeding complete!")

if __name__ == "__main__":
    # Standard path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(current_dir, "crimesketch.db")
    seed_database(db_path)
