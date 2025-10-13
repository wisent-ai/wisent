# Example of using WisentModel for inference with streaming

# First we need to import the WisentModel class

from wisent_guard.core.models.wisent_model import WisentModel

MODEL_NAME = "Llama-3.2-1B-Instruct/"  # Example model name

wisent = WisentModel(
    model_name=MODEL_NAME,  
    device="cuda"  
)

# Now we need to create a chat-style prompt
prompt = [[{"role": "user", "content": "What is wisent?"}]]

# We can now use the model to generate a response with streaming
response_stream = wisent.generate_stream(
    inputs=prompt,
    max_new_tokens=512,
    temperature=0.01, 
)
print("Prompt:", prompt[0][0]["content"])
print("Response (streaming): ", end="", flush=True)
for chunk in response_stream:
    print(chunk, end="", flush=True)
