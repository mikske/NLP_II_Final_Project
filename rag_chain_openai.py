#дисклеймер: у меня есть апи ключ опенаи
import os
from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from rag_mvp import load_documents, chunk_recursive, add_chunk_ids, build_vectorstore, build_retrievers


SYSTEM_PROMPT = """
Ты банковский консультант. Отвечай только на основе найденного контекста.
Если в контексте нет ответа, скажи: «В базе знаний нет достаточной информации».
Не выдумывай условия. Указывай источники в конце ответа в формате: Источники: title / chunk_id.
Ответ должен быть кратким, точным и понятным клиенту.
"""

#достаем апи
def setup_openai_key() -> None:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get("OPENAI_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY не найден."
        )

    os.environ["OPENAI_API_KEY"] = api_key


#форматируем: название документа + chunk_id
def format_docs(docs):
    if not docs:
        return "Релевантный контекст не найден."

    return "\n\n".join(
        f"Источник: {d.metadata.get('title')} / {d.metadata.get('chunk_id')}\n{d.page_content}"
        for d in docs
    )


#собираем полный RAG pipeline:
def build_rag_chain():
    setup_openai_key()

    docs = load_documents()
    chunks = add_chunk_ids(chunk_recursive(docs), "recursive")
    vectorstore = build_vectorstore(chunks)

    #берем compressed retriever, потому что он использует гибридный поиск
    retriever = build_retrievers(vectorstore, chunks)["compressed"]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "Контекст:\n{context}\n\nВопрос: {question}"),
        ]
    )

    #низкая температура потому что нам важна точность
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=500,
    )

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda _: [],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


if __name__ == "__main__":
    chain = build_rag_chain()
    print(chain.invoke("Можно ли пополнять вклад Смарт Доход?"))

