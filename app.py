# Import necessary libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st

# Load the model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"  # You can replace this with your fine-tuned model

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Move the model to evaluation mode and to GPU if available
model.eval()
if torch.cuda.is_available():
    model = model.to('cuda')

# Function to generate a response from the chatbot
def generate_response(prompt):
    # Encode the prompt and prepare for generation
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Move tensors to GPU if available
    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    # Generate response with controlled parameters
    outputs = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],  # Include the attention mask
        max_length=100,
        do_sample=True,        # Enable sampling for varied output
        temperature=0.5,      # Lower temperature for more deterministic output
        top_p=0.9,            # Nucleus sampling
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode the response and strip whitespace
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.strip()
    
    # Post-processing: Check for repetitive phrases
    if "A cable is a type of cable" in response:
        return "Could you please specify which type of cable you're referring to? We have various options like coaxial, fiber optic, and HDMI."

    return response

# Streamlit application
st.title("Cable Industry Chatbot")
context = "You are a helpful assistant in the cable industry. Please provide detailed and accurate answers."

# User input for chatbot
user_input = st.text_input("Ask me anything about cables:")

if user_input:
    full_prompt = f"{context}\nUser: {user_input}\nAssistant:"
    response = generate_response(full_prompt)
    st.write(response)
