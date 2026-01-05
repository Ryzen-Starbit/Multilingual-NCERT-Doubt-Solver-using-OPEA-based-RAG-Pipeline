import json
import os
from datetime import datetime
MEMORY_FILE = "backend/chat_memory_store.json"
if not os.path.exists(MEMORY_FILE):
    with open(MEMORY_FILE, "w") as f:
        json.dump({}, f)
def load_memory():
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)
def save_memory(memory):
    with open(MEMORY_FILE, "w") as f:
        json.dump(memory, f, indent=2)
def add_message(session_id, role, content):
    memory = load_memory()
    if session_id not in memory:
        memory[session_id] = []
    memory[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    })
    save_memory(memory)
def get_conversation(session_id, limit=6):
    memory = load_memory()
    return memory.get(session_id, [])[-limit:]
