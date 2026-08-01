from dotenv import load_dotenv
import os
from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_hugging_face_embeddings,
)
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

print("Loading PDF...")
extracted_data = load_pdf_file(data="data/")
print(f"PDF pages loaded: {len(extracted_data)}")

filter_data = filter_to_minimal_docs(extracted_data)

print("Splitting text...")
text_chunks = text_split(filter_data)
print(f"Text chunks created: {len(text_chunks)}")

print("Loading embeddings...")
embeddings = download_hugging_face_embeddings()
print("Embeddings loaded.")

print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

if not pc.has_index(index_name):
    print("Creating Pinecone index...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

print("Uploading documents...")

try:
    PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=embeddings,
    )

    print("Documents uploaded successfully!")

    stats = index.describe_index_stats()
    print("Index Stats:")
    print(stats)

except Exception as e:
    print("ERROR OCCURRED:")
    print(type(e).__name__)
    print(e)