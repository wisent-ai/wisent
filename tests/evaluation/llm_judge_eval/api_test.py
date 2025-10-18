import os
from anthropic import Anthropic

client = Anthropic()  # odczyta ANTHROPIC_API_KEY z env
msg = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=512,
    messages=[{"role": "user", "content": "Napisz haiku o jesieni."}],
)
print(msg.content[0].text)
