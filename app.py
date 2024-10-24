# Function to generate a response from the chatbot
def generate_response(prompt):
    # Ensure the prompt is a string
    if not isinstance(prompt, str):
        return "Invalid input. Please ask a valid question."

    # Encode the prompt and prepare for generation
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding='longest',  # Pad to the longest sequence in the batch
        truncation=True,
        max_length=512      # Set a maximum length for the input
    )
    
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
