from src.models.language_model import LanguageReasoningModel
from src.models.embedder import EmbeddingModel
from src.question.processor import QuestionProcessor
from src.search.wiki_fetcher import WikipediaFetcher
from src.search.document_splitter import DocumentSplitter
from src.search.vector_store import VectorStoreManager
from src.rag.rag_chain import RetrievalAugmentedGenerationChain
from src.reporter.report_generator import ReportGenerator
from langchain_huggingface import HuggingFacePipeline as Pipeline
from langchain_huggingface import ChatHuggingFace as Chat
import transformers
from tqdm import tqdm
import pickle
from dynaconf import Dynaconf

def main(cfg: object):
    reasoning_model = LanguageReasoningModel(model_name=cfg.model_name)
    
    # Load the embedding model.
    embedding_model = EmbeddingModel()
    
    # Initialize the question processor.
    question_processor = QuestionProcessor(reasoning_model)
    
    # Get sub-questions.
    sub_questions = question_processor.extract_subquestions(cfg.query)
    
    # Break each sub-question into further sub-sub-questions.
    breakdown = {}
    for q in sub_questions:
        sub_subs = question_processor.extract_subsubquestions(q, cfg.topic)
        breakdown[q] = sub_subs
    
    # Fetch Wikipedia page and split the text.
    wiki_fetcher = WikipediaFetcher()
    page = wiki_fetcher.get_page(cfg.page_title)
    doc_splitter = DocumentSplitter(cfg.doc_chunk_size, cfg.doc_chunk_overlap)
    docs = doc_splitter.split(page.text)
    
    # Build the vector store.
    vector_store_manager = VectorStoreManager(embedding=embedding_model.embeddings)
    vectorstore = vector_store_manager.build_vectorstore(docs)
    
    # Set up a HuggingFace pipeline based LLM (you can swap this with another model as needed).
    hf_pipeline = transformers.pipeline(
         model=reasoning_model.model,
         tokenizer=reasoning_model.tokenizer,
         task="text-generation",
         return_full_text=False,
         max_new_tokens=4048,
    )
    llm = Pipeline(pipeline=hf_pipeline)
    chat = Chat(llm=llm)
    
    # Set up the retrieval-augmented generation chain.
    rag_chain = RetrievalAugmentedGenerationChain(llm=llm, retriever=vectorstore.as_retriever())
    
    # Generate answers for each (sub-)question.
    answers = {}
    total_questions = sum(1 for subs in breakdown.values() if subs) + len(breakdown)
    pbar = tqdm(total=total_questions)
    for key, sub_q_list in breakdown.items():
        if not sub_q_list:
            answers[key] = rag_chain.generate_answer(key)
            pbar.update(1)
        else:
            for sub_q in sub_q_list:
                answers[sub_q] = rag_chain.generate_answer(sub_q)
                pbar.update(1)
    pbar.close()
    
    # Synthesize a markdown report.
    report_generator = ReportGenerator(cfg.topic, breakdown, answers)
    markdown_report = report_generator.generate_markdown()
    
    # Save the report and answers.
    with open(cfg.report_file_path, 'w') as f:
        f.write(markdown_report)
    with open(cfg.answers_file_path, 'wb') as f:
        pickle.dump(answers, f)

config_path = "./configs/main.yaml"
if __name__ == "__main__":

    cfg = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=[config_path],
    )
    main()
