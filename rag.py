from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
VECTORSTORE_DIR = Path(__file__).parent / "resources" / "vectorstore"
COLLECTION_NAME = "research_tool"

llm = None
vector_store = None



def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )


def process_pdf(pdf_paths):
    print("Initializing components")
    initialize_components()

    vector_store.reset_collection()

    print("Loading data")
    all_docs = []
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        all_docs.extend(loader.load())

    print("Splitting text")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=200,
    )
    docs = text_splitter.split_documents(all_docs)

    print(f"Adding {len(docs)} chunks to vector DB")
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)


def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector DB is not initialized")

    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
    result = chain.invoke({"question": query}, return_only_outputs=True)
    sources = result.get('sources', '')

    return result['answer'], sources



if __name__ == "__main__":
    pdfs = [

    ]

    process_pdf(pdfs)
    answer, sources = generate_answer(
            ""
    )
    print(f"Answers: {answer}")
    print(f"Sources: {sources}")
