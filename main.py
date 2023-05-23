from pprint import pprint

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Supplying a persist_directory will store the embeddings on disk
persist_directory = "db"

embeddings = OpenAIEmbeddings()


def create_get_index():
    loader = DirectoryLoader("./data", glob="**/*.txt")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = loader.load()
    documents = text_splitter.split_documents(docs)
    if len(documents) == 0:
        raise ValueError("No documents")
    return Chroma.from_documents(
        documents=documents, embedding=embeddings, persist_directory=persist_directory
    )


def get_index():
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)


vectordb = get_index()

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=vectordb.as_retriever()
)

result = qa("var kan jag äta röd curry i stockholm")
pprint(result)
