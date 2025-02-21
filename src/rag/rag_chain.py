from langchain_core.prompts import PromptTemplate

class RetrievalAugmentedGenerationChain:
    def __init__(self, llm, retriever):
        self.llm = llm
        self.retriever = retriever
        self.prompt_template = """
        You are an AI assistant and provide answers using factual and statistical information when possible.
        Use the following context to provide a concise answer to the question.
        If you don't know the answer, just say so.

        $context$
        {context}
        $/context$

        $question$
        {question}
        $/question$
        """
        self.prompt = PromptTemplate(template=self.prompt_template, input_variables=["question"])

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_answer(self, question: str):
        # Retrieve context based on the question.
        docs = self.retriever.get_relevant_documents(question)
        context = self.format_docs(docs)
        chain_input = {"context": context, "question": question}
        # Generate an answer using the provided LLM.
        answer = self.llm(chain_input)
        return answer
