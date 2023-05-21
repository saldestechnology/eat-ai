from pprint import pprint

from langchain import ElasticVectorSearch, OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

elasticsearch_url = "http://localhost:9200"
index_name = "menus"
embeddings = OpenAIEmbeddings()


def create_get_index():
    loader = DirectoryLoader("./data", glob="**/*.txt")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = loader.load()
    documents = text_splitter.split_documents(docs)
    if len(documents) == 0:
        raise ValueError("No documents")
    return ElasticVectorSearch.from_documents(documents, embeddings, elasticsearch_url=elasticsearch_url,
                                              index_name=index_name)


def get_index():
    return ElasticVectorSearch(elasticsearch_url=elasticsearch_url, index_name=index_name, embedding=embeddings)


index = get_index()
query = "var kan jag äta saffran rätt"
docs = index.similarity_search(query)
pprint(docs)

llm = OpenAI(temperature=0)
chain = load_qa_chain(llm, chain_type="stuff")

result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
pprint(result)

