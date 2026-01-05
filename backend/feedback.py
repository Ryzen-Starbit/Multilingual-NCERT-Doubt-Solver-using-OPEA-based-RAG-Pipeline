import json
import os
from datetime import datetime
FEEDBACK_FILE = "backend/feedback_store.json"
if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump([], f)
def save_feedback(question, rating, comment=None):
    with open(FEEDBACK_FILE, "r") as f:
        data = json.load(f)
    data.append({
        "question": question,
        "rating": rating,
        "comment": comment,
        "timestamp": datetime.utcnow().isoformat()
    })
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(data, f, indent=2)
