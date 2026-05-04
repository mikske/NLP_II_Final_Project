#%%
#импортируем все, что может понадобиться
#LangChain для построения пайплайна
#ChromaDB как векторная база данных
#BM25 для реализации гибридного поиск

import json
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rank_bm25 import BM25Okapi

DATA_PATH = Path("data/bank_docs.json")
PERSIST_DIR = "chroma_bank"
EMBED_MODEL = "intfloat/multilingual-e5-large"

#очищаем текст
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[•●▪]", "", text)
    return text.strip()


#загружаем
def load_documents() -> List[Document]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Не найден файл {DATA_PATH}. Проверь, что запускаешь скрипт из корня проекта, "
            f"где есть папка data/bank_docs.json"
        )

    rows = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    docs = []

    for row in rows:
        docs.append(
            Document(
                page_content=clean_text(row["text"]),
                metadata={
                    "doc_id": row["doc_id"],
                    "title": row["title"],
                    "product_type": row["product_type"],
                },
            )
        )

    return docs


#я большой любитель кастомного чанкинга, моя ВКР на 90% из этого состоит не судите строго
#чанкинг по фиксированному размеру
def chunk_by_size(docs: List[Document]) -> List[Document]:
    splitter = CharacterTextSplitter(separator=" ", chunk_size=500, chunk_overlap=80)
    return splitter.split_documents(docs)


#чанкинг по предложениям
def chunk_by_sentences(docs: List[Document]) -> List[Document]:
    chunks = []

    for doc in docs:
        sentences = re.split(r"(?<=[.!?])\s+", doc.page_content)
        buffer = []

        for sent in sentences:
            buffer.append(sent)

            if len(" ".join(buffer)) >= 350:
                chunks.append(
                    Document(
                        page_content=" ".join(buffer),
                        metadata=doc.metadata.copy(),
                    )
                )
                buffer = []

        if buffer:
            chunks.append(
                Document(
                    page_content=" ".join(buffer),
                    metadata=doc.metadata.copy(),
                )
            )

    return chunks


#рекурсивный чанкинг делит по крупным логическим границам, потом по предложениям и словам
def chunk_recursive(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    )
    return splitter.split_documents(docs)


#добавляем каждому чанку id
def add_chunk_ids(chunks: List[Document], strategy: str) -> List[Document]:
    for i, chunk in enumerate(chunks):
        chunk.metadata = chunk.metadata.copy()
        chunk.metadata["chunk_id"] = f"{strategy}_{i:03d}"
        chunk.metadata["chunk_strategy"] = strategy
    return chunks


# создаем модель эмбеддингов, искать будет по косиносному сходству
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


#собираем векторную базу ChromaDB
#recreate=True нужен, чтобы при повторном запуске Colab не ругался на старую коллекцию
def build_vectorstore(chunks: List[Document], persist_dir: str = PERSIST_DIR, recreate: bool = True) -> Chroma:
    if recreate and Path(persist_dir).exists():
        shutil.rmtree(persist_dir)

    embeddings = get_embeddings()

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="bank_products",
    )


#создаем несколько вариантов ретривера
def build_retrievers(vectorstore: Chroma, chunks: List[Document]):
    similarity = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 4

    hybrid = EnsembleRetriever(
        retrievers=[similarity, bm25],
        weights=[0.65, 0.35],
    )

    compressor = EmbeddingsFilter(
        embeddings=get_embeddings(),
        similarity_threshold=0.55,
    )

    compressed = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=hybrid,
    )

    return {
        "similarity": similarity,
        "hybrid": hybrid,
        "compressed": compressed,
    }


#пример фильтрации по метаданным
def filter_by_product(vectorstore: Chroma, question: str, product_type: str):
    return vectorstore.similarity_search(
        question,
        k=4,
        filter={"product_type": product_type},
    )


#тест
TEST_SET = [
    ("Какая минимальная сумма потребительского кредита?", "credit_cash_001"),
    ("Можно ли досрочно погасить кредит без штрафа?", "credit_cash_001"),
    ("Какие требования к заемщику по кредиту наличными?", "credit_cash_001"),
    ("Можно ли оформить кредит без справки о доходах?", "credit_cash_001"),
    ("Какая ставка по семейной ипотеке?", "mortgage_family_001"),
    ("Какой первоначальный взнос по ипотеке Дом Семьи?", "mortgage_family_001"),
    ("Можно ли использовать материнский капитал?", "mortgage_family_001"),
    ("Сколько созаемщиков можно привлечь по ипотеке?", "mortgage_family_001"),
    ("Какая ставка по вкладу на 6 месяцев?", "deposit_smart_001"),
    ("Можно ли пополнять вклад Смарт Доход?", "deposit_smart_001"),
    ("Есть ли частичное снятие по вкладу?", "deposit_smart_001"),
    ("Что будет при досрочном расторжении вклада?", "deposit_smart_001"),
    ("Когда обслуживание дебетовой карты бесплатное?", "debit_card_001"),
    ("Какой кэшбэк по карте Комфорт?", "debit_card_001"),
    ("Сколько стоит СМС по карте?", "debit_card_001"),
    ("Какой лимит бесплатных переводов по номеру телефона?", "debit_card_001"),
    ("Сколько стоит обслуживание тарифа Бизнес Старт?", "small_business_001"),
    ("Сколько бесплатных платежей юрлицам включено?", "small_business_001"),
    ("Какие документы нужны ООО для открытия счета?", "small_business_001"),
    ("Можно ли открыть расчетный счет дистанционно?", "small_business_001"),
]


#оцениваем retrieval quality
#Hit Rate@k показывает, попал ли нужный документ в топ-k
#MRR учитывает позицию правильного документа: чем выше он в выдаче, тем лучше
def evaluate_retriever(retriever, test_set=TEST_SET, k: int = 4) -> Dict[str, float]:
    hits = []
    reciprocal_ranks = []

    for question, gold_doc_id in test_set:
        docs = retriever.invoke(question)[:k]
        doc_ids = [d.metadata.get("doc_id") for d in docs]

        hit = int(gold_doc_id in doc_ids)
        hits.append(hit)

        if hit:
            rank = doc_ids.index(gold_doc_id) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    return {
        "hit_rate@k": float(np.mean(hits)),
        "mrr": float(np.mean(reciprocal_ranks)),
    }


# сравниваем стратегии чанкинга
def compare_chunking() -> pd.DataFrame:
    docs = load_documents()

    strategies = {
        "size": chunk_by_size,
        "sentences": chunk_by_sentences,
        "recursive": chunk_recursive,
    }

    rows = []

    for name, func in strategies.items():
        chunks = add_chunk_ids(func(docs), name)
        vs = build_vectorstore(chunks, persist_dir=f"{PERSIST_DIR}_{name}", recreate=True)
        retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        metrics = evaluate_retriever(retriever)

        rows.append(
            {
                "strategy": name,
                "chunks": len(chunks),
                **metrics,
            }
        )

    return pd.DataFrame(rows)

def main():
    docs = load_documents()
    chunks = add_chunk_ids(chunk_recursive(docs), "recursive")
    vectorstore = build_vectorstore(chunks, persist_dir=PERSIST_DIR, recreate=True)
    retrievers = build_retrievers(vectorstore, chunks)

    print("Chunking comparison:")
    print(compare_chunking())

    print("\nRetriever comparison:")
    rows = []

    for name, retriever in retrievers.items():
        start = time.perf_counter()
        metrics = evaluate_retriever(retriever)
        elapsed = time.perf_counter() - start

        rows.append(
            {
                "retriever": name,
                **metrics,
                "eval_time_sec": elapsed,
            }
        )

    print(pd.DataFrame(rows))

    question = "Какая ставка по вкладу на 6 месяцев?"
    print("\nFiltered search example:")

    for d in filter_by_product(vectorstore, question, "deposit"):
        print(d.metadata, d.page_content[:250])


if __name__ == "__main__":
    main()
