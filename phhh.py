from huggingface_hub import InferenceClient

client = InferenceClient(model="mistralai/Mistral-7B-Instruct-v0.3", token="YOUR_HF_TOKEN")

response = client.text_generation("Tell me a short story about a dragon.", max_new_tokens=100)
print(response)
