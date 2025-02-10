from src.helper import repo_ingestion, load_repo, text_split, load_embedding
from langchain.vectorstores import Chroma
import os


url = "https://github.com/ZikGitHub/Medical-Chatbot-GenerativeAI.git"
repo_ingestion(repo_url=url)

documents = load_repo(repo_path="data/")
text = text_split(documents)
embeddings = load_embedding()

# storing vector in chromadb
vectordb = Chroma.from_documents(documents=text, embedding=embeddings, persist_directory="db")
vectordb.persist()