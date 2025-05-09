import chromadb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# setting the environment

# Use os.path.join to create the DATA_PATH, ensuring it's correct
DATA_PATH = r"C:\Users\neela\Desktop\Miscellaneous\coding\Pydantic_agent\data"
# Use a different path for the ChromaDB persistence directory
CHROMA_PATH = r"C:\Users\neela\Desktop\Miscellaneous\coding\Pydantic_agent\db"  


chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)


collection = chroma_client.get_or_create_collection(name="TechSupportFAQs")

# Ensure the DATA_PATH is a directory containing the PDF
loader = PyPDFDirectoryLoader(os.path.dirname(DATA_PATH), glob="**/Blender_(software).pdf")

raw_documents = loader.load()

# Check if raw_documents is empty and print a message if it is
if not raw_documents:
    print(f"No documents found at path: {DATA_PATH}. Please check the path and ensure the PDF file exists.")



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(raw_documents)



documents = []
metadata = []
ids = []

i = 0

for chunk in chunks:
    documents.append(chunk.page_content)
    ids.append("ID"+str(i))
    metadata.append(chunk.metadata)

    i += 1

# adding to chromadb


collection.upsert(
    documents=documents,
    metadatas=metadata,
    ids=ids
)


def search_RAG(query):
    results = collection.query(
        query_texts=[query],
        n_results=3,
    )
    return results