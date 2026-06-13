import os
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

VECTOR_DB_PATH = "vector_store"

# Local embedding model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


def create_vector_store(text):
    """
    Split document into chunks and create FAISS vector store.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_text(text)

    embeddings = get_embeddings()

    vector_store = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings
    )

    vector_store.save_local(VECTOR_DB_PATH)

    return len(chunks)


def load_vector_store():
    embeddings = get_embeddings()

    return FAISS.load_local(
        VECTOR_DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def retrieve_context(question, k=5):
    vector_store = load_vector_store()

    docs = vector_store.similarity_search(
        question,
        k=k
    )

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    sources = [
        doc.page_content[:500]
        for doc in docs
    ]

    return context, sources


def ask_question(question):
    """
    RAG Chat
    """

    context, sources = retrieve_context(question)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.1
    )

    prompt = f"""
You are SignSafe AI.

You help users understand agreements and contracts.

Use ONLY the provided agreement context.

If the answer is not present in the agreement,
say:

"I could not find that information in the uploaded agreement."

Agreement Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content, sources