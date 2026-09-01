import os
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env
load_dotenv()

# Get API key
api_key = os.getenv("OPENAI_API_KEY")

# Create OpenAI client
client = OpenAI(api_key=api_key)

# Get message from user
user_message = input("You: ")

# Send message to the AI model
response = client.responses.create(
    model="gpt-3.5-turbo",
    input=user_message
)

# Print AI response
print("AI:", response.output_text)