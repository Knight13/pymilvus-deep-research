from langchain_text_splitters import RecursiveCharacterTextSplitter

class DocumentSplitter:
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split(self, text: str):
        return self.splitter.create_documents([text])
