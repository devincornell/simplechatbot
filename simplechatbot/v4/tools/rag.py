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
import langchain_core.tools
import langchain_core.retrievers

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
    
    def as_tool(self, name: str, description: str) -> langchain_core.tools.BaseTool:
        '''Return this RAG object as a tool based on name and description.'''
        return langchain_core.tools.create_retriever_tool(
            retriever=RAGRetriever(self.vectorstore),
            description=description,
            name=name,
        )
    
    def search(self, input_message: str) -> list[langchain_core.documents.Document]:
        '''Search for a query in the vectorstore.'''
        return self.vectorstore.similarity_search(input_message)


# could be a dataclass but avoiding that in case it interacts with BaseRetriever
class RAGRetriever(langchain_core.retrievers.BaseRetriever):
    '''Implementation of BaseRetriever to use for .create_retriever_tool().
    Description:
        See documentation for BaseRetriever here:
        https://python.langchain.com/v0.2/api_reference/core/retrievers/langchain_core.retrievers.BaseRetriever.html
    '''
    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

    def _get_relevant_documents(self, query: str) -> list[langchain_core.documents.Document]:
            """All BaseRetriever subclasses must implement this method."""
            self.vectorstore.similarity_search(query)

