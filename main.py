import os

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

from langchain_text_splitters import CharacterTextSplitter

def build_vector_store(speech_path: str, persist_dir: str = "chroma_speech"):
    # 1. Load document
    loader = TextLoader(speech_path, encoding="utf-8")
    documents = loader.load()

    # 2. Split into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
    )
    docs = text_splitter.split_documents(documents)

    # 3. Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Build or reload Chroma store
    if os.path.exists(persist_dir):
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        vectordb.persist()

    return vectordb

def qa_query(llm, retriever, query):
    docs = retriever.invoke(query)
    context = "\n".join(doc.page_content for doc in docs)
    prompt = f"Given the following context, answer the question:\n\n{context}\n\nQuestion: {query}\nAnswer:"
    answer = llm.invoke(prompt)
    return answer

def main():
    speech_path = "speech.txt"
    if not os.path.exists(speech_path):
        raise FileNotFoundError(
            f"{speech_path} not found. Place the file in the project root."
        )

    print("Building / loading vector store...")
    vectordb = build_vector_store(speech_path)

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    llm = Ollama(model="mistral")

    print("AmbedkarGPT CLI ready. Type your question (or 'exit' to quit).")
    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        answer = qa_query(llm, retriever, query)
        print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()
