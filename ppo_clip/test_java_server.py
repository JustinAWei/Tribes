import requests
import json
from pprint import pprint

def send_new_game_request():
    url = "http://localhost:8080/newGame"
    headers = {
        "Content-Type": "application/json"
    }
    # If your server expects any specific JSON payload, you can define it here
    payload = {}  # Adjust this if your server requires specific data

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise an error for bad responses
        print("Response from server:")
        # Pretty print the JSON response
        pprint(response.json())
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    send_new_game_request()