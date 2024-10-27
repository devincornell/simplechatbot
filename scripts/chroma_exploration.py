'''
Here I was exploring the tutorial on setting up RAG.
https://python.langchain.com/docs/tutorials/qa_chat_history/

It is also similar to the beginning of this article:
https://python.langchain.com/docs/tutorials/rag/

I found this article which provided some other context that I drew into this.
https://python.langchain.com/docs/integrations/vectorstores/chroma/

'''

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import getpass
import os
from langchain_chroma import Chroma


import sys
sys.path.append('..')
import simplechatbot.v4 as simplechatbot


if __name__ == '__main__':
    if True:
        # this just downloads the github page, parses it, and returns an iterable of  
        # langchain_core.documents.Document objects
        from langchain_community.document_loaders import WebBaseLoader
        import bs4

        # you can see the markdown doc here: 
        # https://lilianweng.github.io/posts/2023-06-23-agent/

        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("post-content", "post-title", "post-header")
                )
            ),
        )
        docs = loader.load()
    else:
        # alternatively, we can instantiate those objects manually
        # in this tutorial I saw that we can actually add our own docs
        # https://python.langchain.com/docs/integrations/vectorstores/chroma/
        from langchain_core.documents import Document

        document_1 = Document(
            page_content="I had chocolate chip pancakes and scrambled eggs for breakfast this morning.",
            metadata={"source": "tweet"},
            id=1,
        )

        document_2 = Document(
            page_content="The weather forecast for tomorrow is cloudy and overcast, with a high of 62 degrees.",
            metadata={"source": "news"},
            id=2,
        )

        document_3 = Document(
            page_content="Building an exciting new project with LangChain - come check it out!",
            metadata={"source": "tweet"},
            id=3,
        )
        docs = [document_1, document_2, document_3]

    # reguardless of the method used to instantiate the docs, we can print them out
    for doc in docs:
        print(doc)
    

    # now they just use a text splitter to make embeddings and chunk up the doc
    keychain = simplechatbot.util.APIKeyChain.from_json_file('keys.json')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=NVIDIAEmbeddings(model="NV-Embed-QA", api_key=keychain["nvidia"]))


    # now we just have a query from a user.
    input_message = "How do I implement Self-Reflection?"

    docs = vectorstore.similarity_search(input_message)

    parsed_docs = "\n\n".join([doc.page_content for doc in docs])

    response = f"question:{input_message}\n\ncontext{parsed_docs}"
    print(response)
