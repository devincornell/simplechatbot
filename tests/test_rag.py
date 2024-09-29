from __future__ import annotations
import typing

import dataclasses
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import getpass
import os
from langchain_chroma import Chroma


import langchain_core.documents


import sys
sys.path.append('..')
import simplechatbot.v4 as simplechatbot
from simplechatbot.v4.tools.rag import RAG

if __name__ == '__main__':
    
    try:
        nvidia_api_key = os.environ["NVIDIA_API_KEY"]
    except KeyError:
        raise KeyError("NVIDIA_API_KEY needs to be set!")

    # create all the embeddings and stuff
    rag = RAG.from_web_pages(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        nvidia_api_key=nvidia_api_key,
    )

    print(f'search results for "How do I implement Self-Reflection?"')
    for doc in rag.search("How do I implement Self-Reflection?"):
        print(doc.page_content)

    #docs = vectorstore.similarity_search(input_message)
    #parsed_docs = "\n\n".join([doc.page_content for doc in docs])
    #return f"question:{input_message}\n\ncontext{parsed_docs}"
