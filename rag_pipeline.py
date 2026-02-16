from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


def setup_rag_pipeline():
    # Load the source document
    source_path = "data/Python QA.txt"
    loader = TextLoader(source_path, encoding="utf-8")
    raw_docs = loader.load()

    # Split the text into smaller chunks for embedding
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(raw_docs)

    # Generate embeddings using a sentence transformer model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Store embeddings in an in-memory Chroma vector store
    vectordb = Chroma.from_documents(chunks, embedding_model)

    # Load the LLM (using Ollama's local gemma2:2b model)
    llm_model = OllamaLLM(model="gemma2:2b")

    # Define a custom prompt template for QA
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "Answer the question using only the context below.\n"
            "If the answer isn't in the context, respond with: I don't know.\n"
            "Context:{context}\n"
            "Question:{question}\n"
            "Answer:"
        )
    )
    
    # Build the RetrievalQA chain
    chain = RetrievalQA.from_chain_type(
        llm=llm_model,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        chain_type_kwargs={"prompt": custom_prompt}
    )

    return chain


def main():
    qa_chain = setup_rag_pipeline()
    print("RAG pipeline ready! Ask questions (type 'exit' to quit):")

    while True:
        query = input("\nYour question: ")
        if query.lower() == "exit":
            print("Session ended.")
            break

        result = qa_chain.invoke({"query": query})["result"]
        print(f"Answer: {result}")


if __name__ == "__main__":
    main()
