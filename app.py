import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set page title
st.title("Cable Industry Customer Service Chatbot")

# Cache model and tokenizer loading for better performance
@st.cache(allow_output_mutation=True)
def load_model():
    model_name = "gpt2"  # Use a smaller model for faster performance
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model = model.half()  # Use half-precision for faster inference
    return model, tokenizer

model, tokenizer = load_model()

# Function to generate responses using the loaded model
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids,
        max_length=100,  # Reduced max_length for faster response
        num_beams=3,     # Using beam search instead of sampling
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Streamlit UI for Chatbot Interaction
st.subheader("Chat with the Cable Industry Assistant")
user_input = st.text_input("Ask a question about the cable industry:")

if st.button("Generate Response"):
    if user_input:
        with st.spinner('Generating response...'):
            response = generate_response(user_input)
            st.write(f"**Response:** {response}")
    else:
        st.write("Please enter a question.")
