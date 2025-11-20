import os
import json
from typing import List, Dict
from tqdm import tqdm

import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_text_splitters import CharacterTextSplitter

from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Chunk configs
CHUNK_CONFIGS = {
    "small": {"chunk_size": 250, "chunk_overlap": 50, "persist_dir": "chroma_small"},
    #"medium": {"chunk_size": 550, "chunk_overlap": 80, "persist_dir": "chroma_medium"},
    #"large": {"chunk_size": 900, "chunk_overlap": 120, "persist_dir": "chroma_large"},
}

CORPUS_DIR = "corpus"
TESTDATA_PATH = "/Users/jananip/AmbedkarGPT-Intern-Task/test_subset.json"
RESULTS_PATH = "testresults.json"

def load_corpus() -> List:
    docs = []
    for fname in sorted(os.listdir(CORPUS_DIR)):
        if fname.endswith(".txt"):
            loader = TextLoader(os.path.join(CORPUS_DIR, fname), encoding="utf-8")
            docs.extend(loader.load())
    return docs

def build_vectordb(chunk_size: int, chunk_overlap: int, persist_dir: str):
    docs = load_corpus()
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(persist_dir):
        vectordb = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        vectordb = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        vectordb.persist()
    return vectordb, embeddings

def compute_retrieval_metrics(
    retrieved_docs,
    relevant_sources: List[str],
    k: int = 5
) -> Dict[str, float]:
    import os
    retrieved_sources = []
    for d in retrieved_docs[:k]:
        src = os.path.basename(d.metadata.get("source", ""))
        retrieved_sources.append(src)

    hit = any(src in relevant_sources for src in retrieved_sources)
    hit_rate = 1.0 if hit else 0.0

    reciprocal_rank = 0.0
    for idx, src in enumerate(retrieved_sources, start=1):
        if src in relevant_sources:
            reciprocal_rank = 1.0 / idx
            break

    if len(retrieved_sources) == 0:
        precision_at_k = 0.0
    else:
        correct = sum(1 for src in retrieved_sources if src in relevant_sources)
        precision_at_k = correct / min(k, len(retrieved_sources))

    return {
        "hit_rate": hit_rate,
        "mrr": reciprocal_rank,
        "precision_at_k": precision_at_k
    }

def compute_text_metrics(pred: str, ref: str, embeddings_model) -> Dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(ref, pred)
    rouge_l = rouge_scores["rougeL"].fmeasure

    emb_ref = embeddings_model.embed_query(ref)
    emb_pred = embeddings_model.embed_query(pred)
    cos_sim = cosine_similarity([emb_ref], [emb_pred])[0][0]

    smoothie = SmoothingFunction().method4
    bleu = sentence_bleu(
        [ref.split()],
        pred.split(),
        smoothing_function=smoothie
    )

    answer_relevance = cos_sim
    faithfulness = (cos_sim + rouge_l) / 2.0
    factuality = faithfulness

    return {
        "rouge_l": float(rouge_l),
        "cosine_similarity": float(cos_sim),
        "bleu": float(bleu),
        "answer_relevance": float(answer_relevance),
        "faithfulness": float(faithfulness),
        "factuality": float(factuality),
    }

def load_test_questions() -> List[Dict]:
    with open(TESTDATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def generate_llm_answer(llm, retriever, question):
    docs = retriever.invoke(question)
    context = "\n".join(doc.page_content for doc in docs)
    prompt = f"Given the following context, answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    answer = llm.invoke(prompt)
    return answer, docs

def run_evaluation():
    questions = load_test_questions()
    models = {}
    for name, cfg in CHUNK_CONFIGS.items():
        print(f"Building vector store for {name} chunks...")
        vectordb, embeddings = build_vectordb(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            persist_dir=cfg["persist_dir"]
        )
        llm = Ollama(model="mistral")
        retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2})

        models[name] = {
            "vectordb": vectordb,
            "embeddings": embeddings,
            "llm": llm,
            "retriever": retriever
        }

    all_results = []

    for q in tqdm(questions, desc="Evaluating questions"):
        qid = q["id"]
        question_text = q["question"]
        groundtruth = q["groundtruth"]
        answerable = q["answerable"]
        relevant_sources = q.get("sourcedocuments", [])

        q_result = {
            "id": qid,
            "question": question_text,
            "answerable": answerable,
            "results_by_chunk": {}
        }

        for name, bundle in models.items():
            retriever = bundle["retriever"]
            llm = bundle["llm"]
            embeddings = bundle["embeddings"]

            # Retrieve docs only
            retrieved_docs = retriever.invoke(question_text)

            if answerable and relevant_sources:
                retrieval_metrics = compute_retrieval_metrics(
                    retrieved_docs,
                    relevant_sources,
                    k=5
                )
            else:
                retrieval_metrics = {
                    "hit_rate": None,
                    "mrr": None,
                    "precision_at_k": None
                }

            # Get model answer
            pred_answer, _ = generate_llm_answer(llm, retriever, question_text)

            text_metrics = compute_text_metrics(pred_answer, groundtruth, embeddings)

            q_result["results_by_chunk"][name] = {
                "predicted_answer": pred_answer,
                "retrieval_metrics": retrieval_metrics,
                "answer_metrics": text_metrics,
            }

        all_results.append(q_result)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {RESULTS_PATH}")

if __name__ == "__main__":
    run_evaluation()
