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

if typing.TYPE_CHECKING:
    import langchain_core.documents

@dataclasses.dataclass
class RAG:
    splitter: RecursiveCharacterTextSplitter
    splits: list[langchain_core.documents.Document]
    vectorstore: Chroma

    @classmethod
    def from_web_pages(cls,
        web_paths: tuple[str],
        nvidia_api_key: str,
    ) -> typing.Self:
        # Load, chunk and index the contents of the blog.
        loader = WebBaseLoader(
            web_paths=web_paths,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()

        return cls.from_docs(
            docs = docs, 
            nvidia_api_key=nvidia_api_key,
        )

    @classmethod
    def from_docs(cls, 
        docs: list[langchain_core.documents.Document],
        nvidia_api_key: str,
    ) -> typing.Self:
        '''Create a new vectorstore for working with docs.'''
        # now they just use a text splitter to make embeddings and chunk up the doc

        # should parameterize a bunch of this
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=splits, 
            embedding=NVIDIAEmbeddings(
                model="NV-Embed-QA", 
                api_key=nvidia_api_key
            )
        )
        
        # return the new RAG object
        return cls(
            splitter=text_splitter,
            splits=splits,
            vectorstore=vectorstore
        )
    
    def search(self, input_message: str) -> list[langchain_core.documents.Document]:
        '''Search for a query in the vectorstore.'''
        return self.vectorstore.similarity_search(input_message)
