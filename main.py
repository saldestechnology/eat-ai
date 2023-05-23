from pprint import pprint
import os
from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
import pinecone

index_name = "eat-index"
embeddings = OpenAIEmbeddings()


loader = DirectoryLoader("./data", glob="**/*.txt")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = loader.load()

# initialize pinecone
pinecone.init(
    api_key=os.environ["PINECONE_API_KEY"],  # find at app.pinecone.io
    environment=os.environ["PINECONE_ENV"],  # next to api key in console
)


def create_get_index():
    documents = text_splitter.split_documents(docs)
    if len(documents) == 0:
        raise ValueError("No documents")

    return Pinecone.from_documents(docs, embeddings, index_name=index_name)


def get_index():
    return Pinecone.from_documents(docs, embeddings, index_name=index_name)


index = get_index()
# query = "var kan jag äta en saffransrätt som kostar max 80"
query = "vart kan jag äta indiskt i stockholm"
docs = index.similarity_search(query)
pprint(docs)

llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
chain = load_qa_chain(llm, chain_type="stuff")

result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
pprint(result)
