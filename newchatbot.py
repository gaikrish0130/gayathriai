import ollama

def simple_ollama_chatbot():
    """
    Creates a simple chatbot that interacts with an Ollama model.
    """
    print("Welcome to the Ollama Chatbot! Type 'exit' to quit.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        try:
            # Send the prompt to the Ollama model
            response = ollama.chat(model='llama3', messages=[
                {'role': 'user', 'content': user_input},
            ])
            
            # Print the model's response
            print(f"Chatbot: {response['message']['content']}")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Ensure Ollama is running and the 'llama3' model is available.")

if __name__ == "__main__":
    simple_ollama_chatbot()