from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class RetrievalAugmentedGenerationChain:
    def __init__(self, llm, retriever, docs):
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
        self.docs = docs
        self.rag_chain = (
        {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
        | self.prompt
        | self.llm
        | StrOutputParser()
        )

    def format_docs(self):
        return "\n\n".join(doc.page_content for doc in self.docs)

    def generate_answer(self, question: str):
        # Generate an answer using the rag_chain.
        answer = self.rag_chain.invoke(question).split('</think>')[-1].strip()
        return answer
