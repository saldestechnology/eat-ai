import os
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator

loaders = []
for root, _, files in os.walk("./data", topdown=False):
    for name in files:
        file_path = os.path.join(root, name)
        loaders.append(TextLoader(file_path))

index = VectorstoreIndexCreator().from_loaders(loaders=loaders)

query = "var kan jag äta saffran rätt med ris som kostar mindre än 80"
res = index.query_with_sources(query)

print(res)