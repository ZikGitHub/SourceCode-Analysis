from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from src.helper import load_embedding

import os

from flask import Flask, render_template, request, jsonify


app = Flask(__name__)

embeddings = load_embedding()
persist_directory = "db"

# Now we can load the persisted database from disk
vectordb = Chroma(persist_directory=persist_directory,
                   embedding_function=embeddings)

