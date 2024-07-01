# Hugging-Face-Chatbot
# Chatbot Project

## Overview

This repository contains a chatbot project that utilizes the Hugging Face API to implement various models for conversational AI. The chatbot integrates with the Hugging Face endpoints for the "mistralai/Mistral-7B-Instruct-v0.2" and "mistralai/Mistral-7B-Instruct-v0.3" models. Additionally, it demonstrates the usage of the GPT-2 model locally via the Hugging Face pipeline.

### Features

- **Hugging Face API Integration:**
  - Utilizes the "mistralai/Mistral-7B-Instruct-v0.2" and "mistralai/Mistral-7B-Instruct-v0.3" models for remote inference.
  - Provides endpoints to seamlessly call and interact with these models.

- **Local Model Deployment:**
  - Implements the GPT-2 model locally using the Hugging Face pipeline.
  - Offers a fallback mechanism for offline usage or in scenarios where remote endpoints are inaccessible.

### Technical Details

- **Hugging Face API:**
  - Integrated with the Hugging Face API to call the "Mistral-7B-Instruct" models.
  - Used endpoints for model inference, ensuring robust and scalable chatbot interactions.

- **Local GPT-2 Model:**
  - Deployed the GPT-2 model locally using the Hugging Face pipeline.
  - Ensured smooth local inference for consistent chatbot performance.

### Results

- **Enhanced Conversational AI:**
  - Achieved high-quality responses leveraging state-of-the-art models.
  - Demonstrated the flexibility of using both remote and local models.

### Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/chatbot-project.git
   ```

2. **Install dependencies:**
   ```bash
   cd chatbot-project
   pip install -r requirements.txt
   ```

3. **Set up Hugging Face API Token:**
   - Obtain your Hugging Face API token from [Hugging Face](https://huggingface.co/).
   - Set the API token as an environment variable:
     ```bash
     export HUGGINGFACE_API_KEY='your_api_key'
     ```

4. **Run the application:**
   ```bash
   python main.py
   ```

### Usage

- **Calling Remote Models:**
  - Configure the endpoints for "mistralai/Mistral-7B-Instruct-v0.2" and "mistralai/Mistral-7B-Instruct-v0.3".
  - Example API call:
    ```python
    response = call_huggingface_model(endpoint="mistralai/Mistral-7B-Instruct-v0.2", input_text="Hello, how are you?")
    print(response)
    ```

- **Using Local GPT-2 Model:**
  - Ensure the local environment is set up for model inference.
  - Example usage:
    ```python
    from transformers import pipeline

    generator = pipeline('text-generation', model='gpt-2')
    response = generator("Hello, how are you?", max_length=50)
    print(response)
    ```

---

Thank you for exploring the Chatbot Project! We hope it serves as a valuable resource for building advanced conversational AI applications.
