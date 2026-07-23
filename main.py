import anthropic
from dotenv import load_dotenv
import json

load_dotenv()
client = anthropic.Anthropic()

def ask_claude(conversation):
    response = client.messages.create(
        model = "claude-haiku-4-5-20251001",
        max_tokens = 1024,
        messages = conversation
    )
    print("_________________________")
    print(response.content)
    print("_________________________")
    conversation.append({"role": "assistant", "content": response.content[0].text}) # appending claude's response for conversation history

command = ""
conversation = []
try:
    with open("conversation.json", "r") as f:
        conversation = list(json.load(f))
except:
    conversation = []

try:
    while command != "quit":
        command = input()
        # just to get rid of annoying things for Quit QUIT quit yknow im sayin?
        command = command.lower()

        if command != "quit":
            conversation.append({"role": "user", "content": command}) # appending user's question to keep track of conversation
            ask_claude(conversation)
            print(conversation[-1]["content"]) # claude's response
            print("***************************")
finally:
    with open("conversation.json", "w") as f:
        json.dump(conversation, f)