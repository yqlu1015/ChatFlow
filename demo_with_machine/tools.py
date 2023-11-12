from typing import Optional

from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatLiteLLM
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.tools import StructuredTool

GPT_MODEL = "gpt-3.5-turbo-0613"


class Tools(object):
    """ A collection of tools used by llm.

    """

    @staticmethod
    def check_order():
        print("check_order successfully called!")


tool = Tools()


def setup_knowledge_base(knowledge_file: str = "ad_words.txt",
                         collection_name: str = "product-knowledge-base"):
    """
    We assume that the product catalog is simply a text string.
    """
    # load product catalog
    with open(knowledge_file, "r") as f:
        knowledge_file = f.read()

    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    texts = text_splitter.split_text(knowledge_file)

    llm = OpenAI(temperature=0)
    # llm = ChatLiteLLM(temperature=0, model_name=GPT_MODEL)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(
        texts, embeddings, collection_name=collection_name
    )

    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )
    return knowledge_base


def get_tools(knowledge_base: Optional[RetrievalQA] = None) -> list:
    """Construct a list of tools from functions.

    """
    tools = []
    if knowledge_base:
        tools.append(
            Tool(
                name="ProductSearch",
                func=knowledge_base.run,
                description="useful for when you need to answer questions about product information",
            )
        )
    return tools
