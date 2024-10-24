import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

# Set page title
st.title("Cable Industry Customer Service Chatbot")

# Load GPT-Neo model and tokenizer (you can replace with a smaller model)
model_name = "EleutherAI/gpt-neo-1.3B"  # Choose a smaller model like GPT-Neo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate responses using the loaded model
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs.input_ids, 
        max_length=200, 
        do_sample=True, 
        temperature=0.7, 
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Function for fine-tuning the model using a Hugging Face dataset
def fine_tune_model(dataset_name, output_dir='./fine-tuned-model'):
    dataset = load_dataset(dataset_name)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        num_train_epochs=1,
        save_steps=500,
        save_total_limit=2
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
    )
    
    trainer.train()
    return f"Model fine-tuned and saved to {output_dir}"

# Streamlit UI for Chatbot Interaction
st.subheader("Chat with the Cable Industry Assistant")
user_input = st.text_input("Ask a question about the cable industry:")

if st.button("Generate Response"):
    if user_input:
        response = generate_response(user_input)
        st.write(f"**Response:** {response}")
    else:
        st.write("Please enter a question.")

# Streamlit UI for Fine-Tuning the Model
st.subheader("Fine-Tune the Chatbot")
dataset_input = st.text_input("Enter the Hugging Face Dataset Name (e.g., 'cable_data')")

if st.button("Fine-tune Model"):
    if dataset_input:
        with st.spinner("Fine-tuning..."):
            result = fine_tune_model(dataset_input)
            st.success(result)
    else:
        st.write("Please enter a valid dataset name.")

