from langchain_milvus import Milvus

class VectorStoreManager:
    def __init__(self, embedding, uri: str = "./milvus_demo.db", drop_old: bool = True):
        self.embedding = embedding
        self.uri = uri
        self.drop_old = drop_old
        self.vectorstore = None

    def build_vectorstore(self, documents, index_params: dict = None):
        if index_params is None:
            index_params = {
                "metric_type": "COSINE",
                "index_type": "FLAT",
                "params": {},
            }
        self.vectorstore = Milvus.from_documents(
            documents=documents,
            embedding=self.embedding,
            connection_args={"uri": self.uri},
            drop_old=self.drop_old,
            index_params=index_params,
        )
        return self.vectorstore

    def as_retriever(self):
        if self.vectorstore:
            return self.vectorstore.as_retriever()
        else:
            raise ValueError("Vectorstore has not been built yet.")
