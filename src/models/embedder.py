from langchain_community.embeddings import SentenceTransformerEmbeddings

class EmbeddingModel:
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.embeddings = SentenceTransformerEmbeddings(model_name=model_name)
