import os
import requests
from dotenv import load_dotenv

# Load API key from .env
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("")

# DeepSeek API endpoint
DEEPSEEK_API_URL = ""


# Initialize API client
def deepseek_api_call(prompt):
    url = ""

    payload = {
        "model": "Pro/deepseek-ai/DeepSeek-V3", # #Pro/deepseek-ai/DeepSeek-V3
        "max_tokens": 8000,
        "temperature": 0,
        "messages": [{
                "role": "user",
                "content": prompt,
            }]
    }

    headers = {
        "Authorization": "Bearer ",
        "Content-Type": "application/json"
    }

    response = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=payload,
            timeout=900
        )

    # Attempt JSON parsing with validation
    try:
        json_data = response.json()
        #print(json_data)
    except requests.exceptions.JSONDecodeError:
        raise ValueError("Invalid JSON response")

    # Validate response structure
    required_keys = ["choices", "usage"]
    for key in required_keys:
        if key not in json_data:
            raise KeyError(f"Missing key in response: {key}")

    return json_data["choices"][0]["message"]["content"]


#print(deepseek_api_call("Who are you?"))