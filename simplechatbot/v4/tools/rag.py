import dataclasses
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import getpass
import os
from langchain_chroma import Chroma

@dataclasses.dataclass
class RagPrompt:

    def apply_rag(self, input_message: str) -> str:

        if not os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
            print("Valid NVIDIA_API_KEY needed")
            return 

        # Load, chunk and index the contents of the blog.
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()

        # Create splitter, split documents into 1000 character chunks with 200 character overlap, create vectorstore 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=NVIDIAEmbeddings(model="NV-Embed-QA"))

        docs = vectorstore.similarity_search(input_message)

        parsed_docs = "\n\n".join([doc.page_content for doc in docs])

        return f"question:{input_message}\n\ncontext{parsed_docs}"
