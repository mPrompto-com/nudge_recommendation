import json

def load_qa_data(file_path: str) -> list:
    """Loads Q&A data from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading Q&A data: {e}")
        return []