# client_example.py - Example client usage
import requests
import json

# MLServer endpoint
BASE_URL = (
    "http://localhost:8080/v2/models/reformed-baptist-1689-bible-expert/versions/v0.1.0"
)
MAX_TOKENS = 2**9


def text_completion_example():
    """Example of simple text completion"""

    payload = {
        "inputs": [
            {
                "name": "prompt",
                "shape": [1],
                "datatype": "BYTES",
                "data": ["Explain Hebrews 11:1 in plain language.\n\nAnswer:"],
            },
            {
                "name": "parameters",
                "shape": [1],
                "datatype": "BYTES",
                "data": [json.dumps({"max_tokens": MAX_TOKENS, "temperature": 0.7})],
            },
        ]
    }

    response = requests.post(f"{BASE_URL}/infer", json=payload)

    if response.status_code == 200:
        result = response.json()
        # Decode the response
        generated_text = result["outputs"][0]["data"][0]
        print("Generated text:", generated_text)
    else:
        print(f"Error: {response.status_code}, {response.text}")


def chat_completion_example():
    """Example of chat completion"""

    messages = [
        #        {"role": "system", "content": "You are a concise biblical scholar."},
        {"role": "user", "content": "What does the Bible say about selflessness?"},
    ]

    payload = {
        "inputs": [
            {
                "name": "messages",
                "shape": [1],
                "datatype": "BYTES",
                "data": [json.dumps(messages)],
            },
            {
                "name": "parameters",
                "shape": [1],
                "datatype": "BYTES",
                "data": [json.dumps({"max_tokens": MAX_TOKENS, "temperature": 0.7})],
            },
        ]
    }

    response = requests.post(f"{BASE_URL}/infer", json=payload)

    if response.status_code == 200:
        result = response.json()
        generated_text = result["outputs"][0]["data"][0]
        print("Chat response:", generated_text)
    else:
        print(f"Error: {response.status_code}, {response.text}")


if __name__ == "__main__":
    print("Testing text completion...")
    text_completion_example()

    print("\nTesting chat completion...")
    chat_completion_example()
