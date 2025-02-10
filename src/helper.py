import os
from git import Repo
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import Language
from langchain_huggingface import HuggingFaceEmbeddings

def repo_ingestion(repo_url):
    os.mkdir("data")
    repo_path = "repo/"
    Repo.clone_from(repo_url, to_path=repo_path)



def load_repo(repo_path):
    loader = GenericLoader.from_filesystem(repo_path,
                                           glob="**/*",
                                           suffixes=[".py"],
                                           parser=LanguageParser(language="python", parser_threshold=500))
    
    documents = loader.load()
    return documents



# creating text chunks
def text_split(documents):
    documents_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, 
                                                                      chunk_size=500, 
                                                                      chunk_overlap=20)
    
    texts = documents_splitter.split_documents(documents)
    return texts


# loading embeddings model
def load_embedding():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings