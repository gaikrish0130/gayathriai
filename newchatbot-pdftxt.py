import ollama

from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def simple_ollama_chatbot():
    """
    Creates a simple chatbot that interacts with an Ollama model.
    """
    print("Welcome to the Ollama Chatbot! Type 'exit' to quit.")

    " Load documents (if needed) "
    pdf_loader = PyPDFLoader("Form26AS.pdf")
    pdf_docs = pdf_loader.load()

    " Load a webpage "
    #web_loader = WebBaseLoader("https://sysarcinfomatix.com")
    #web_docs = web_loader.load()

     # Load a text file
    text_loader = TextLoader("test.txt")
    text_docs = text_loader.load()

    " Split documents into chunks "
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_docs = pdf_docs + text_docs #+ web_docs # Combine all loaded documents
    chunks = text_splitter.split_documents(all_docs)

    " Create embeddings for the chunks "
    from langchain_ollama import OllamaEmbeddings
    #embeddings = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = OllamaEmbeddings(model="llama3")

    " Create a vector store from the chunks "
    from langchain_community.vectorstores import Chroma
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    " Initialize the Ollama chat model "
    #from langchain_community.chat_models import ChatOllama
    llm = ChatOllama(model="llama3")

    " Create a retrieval-based QA chain "
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate

    #user_input = input("You: ")
    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question based on the provided context.
    Context: {context}
    Question: {input}
    """)

    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    user_input=input("You: ");
    response = retrieval_chain.invoke({"input": user_input}) 
    #"What is the main topic of the document?"})
    print(f"chatbot resonse: {response["answer"]}")
    exit

    while True:
        
        user_input=input("You: ");
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        try:
            # Send the prompt to the Ollama model
            response = retrieval_chain.invoke({"input": user_input}) 
            print(response["answer"])
            
            # Print the model's response
            #print(f"Chatbot: {response['message']['content']}")

        except Exception as e:
            print(f"An error occurred: {e}")
            print("Ensure Ollama is running and the 'llama3' model is available.")

if __name__ == "__main__":
    simple_ollama_chatbot()