# RAG-система банковского консультанта

## Описание проекта

В рамках проекта была реализована MVP-версия RAG-системы банковского консультанта без использования OpenAI API. Система построена на основе LangChain, FAISS и локальной instruction-модели Qwen/Qwen2.5-0.5B-Instruct.

Основная задача проекта — реализовать полный retrieval-augmented generation pipeline:

* подготовка базы знаний;
* чанкинг документов;
* создание эмбеддингов;
* retrieval;
* генерация ответов на основе найденного контекста;
* оценка качества RAG-системы.

Проект выполнялся в Google Colab.

Ссылка: https://colab.research.google.com/drive/12ci_5kRfDV_lTIZG4Uc97bzZJ2F2qtCw#scrollTo=CDfujEuYo3Of

---

## Используемые технологии

### Основные библиотеки

* Python
* LangChain
* FAISS
* sentence-transformers
* transformers
* Hugging Face
* pandas
* numpy

### Модель эмбеддингов

В качестве embedding-модели использовалась:

```python
intfloat/multilingual-e5-base
```

### LLM

В качестве локальной instruction-модели использовалась:

```python
Qwen2.5-0.5B-Instruct
```

Модель применялась для генерации ответов по retrieved context.

---

# Этап 1. Подготовка данных и создание базы знаний

## Синтетическая база знаний

Была создана небольшая синтетическая база банковских документов.

В базу вошли:

* потребительский кредит;
* ипотечный кредит;
* депозитный продукт;
* требования к заёмщикам;
* FAQ.

Документы содержат:

* процентные ставки;
* суммы;
* сроки;
* требования;
* условия оформления;
* ответы на частые вопросы.

После создания документы были:

* очищены от лишних символов;
* приведены к единому формату;
* дополнены metadata.

---

## Чанкинг

Были протестированы три стратегии чанкинга:

1. С ограничением по объему
2. По предложениям
3. И рекурсивный
Для разбиения документов использовался LangChain.

Наиболее стабильные результаты retrieval показал recursive chunking.

---

## Эмбеддинги и vector store

Для всех чанков были построены embedding-векторы.

В качестве vector database использовался:

```python
FAISS
```

FAISS применялся для similarity search и retrieval.

---

# Этап 2. Система retrieval

## Реализованные retrieval-подходы

Были реализованы:

* similarity search;
* MMR retrieval;
* BM25 retrieval;
* hybrid retrieval.

### Similarity search

Основной retrieval-механизм построен на cosine similarity embedding-векторов.

### MMR

MMR retrieval использовался для повышения разнообразия retrieved chunks.

### BM25

BM25 применялся как keyword-based retrieval.

### Hybrid retrieval

Hybrid retrieval объединяет BM25 и semantic retrieval.

---

## Оценка retrieval

Для оценки retrieval был создан тестовый набор из 20 вопросов.

Использовались метрики:

* HitRate@3
* MRR@5

### Результаты

| Retriever  | HitRate@3 | MRR@5 |
| ---------- | --------- | ----- |
| similarity | 1.00      | 0.925 |
| mmr        | 1.00      | 0.925 |
| bm25       | 0.85      | 0.693 |
| hybrid     | 0.85      | 0.693 |

Лучшие результаты показал semantic retrieval.

---

# Этап 3. Интеграция с LLM

## RAG pipeline

Была реализована полноценная RAG-цепочка:

1. пользователь задаёт вопрос;
2. retriever извлекает релевантные чанки;
3. retrieved chunks объединяются в context;
4. LLM генерирует ответ;
5. система возвращает ответ и источники.

---

## Дополнительные механизмы

Дополнительно были реализованы:

* история диалога;
* self-query retrieval;
* multi-query retrieval;
* reranking;
* metadata filtering;
* проверка groundedness;
* обработка отсутствия релевантной информации.

---

## Пример ответа системы

Вопрос:

```text
Какая максимальная сумма ипотечного кредита?
```

Ответ:

```text
Максимальная сумма ипотечного кредита составляет 30 000 000 рублей.

Источник: Ипотечный кредит (mortgage_001)
```

---

# Этап 4. Анализ и оптимизация

## Оценка качества RAG

Использовались следующие метрики:

* context relevancy;
* answer relevancy;
* faithfulness.

### Итоговые результаты

| Метрика              | Значение |
| -------------------- | -------- |
| context_relevancy    | 1.000    |
| answer_relevancy     | 0.453    |
| faithfulness_numeric | 0.917    |

Низкое значение answer relevancy связано с использованием простой overlap-метрики без лемматизации.

---

## Оптимизация производительности

Были реализованы:

* retrieval caching;
* уменьшение количества retrieved chunks;
* измерение времени ответа системы.

Кеширование retrieval-результатов позволило ускорить повторные запросы.

---

# Основные выводы

В рамках проекта была успешно реализована RAG-система банковского консультанта.

Наиболее эффективным компонентом системы оказался semantic retrieval на основе embedding-модели intfloat/multilingual-e5-base.

Система показывает хорошие результаты retrieval даже на небольшом synthetic dataset.

Основные ограничения текущей версии:

* маленькая база знаний;
* простые эвристические метрики;
* отсутствие полноценного reranker;
* отсутствие production-инфраструктуры.
---

# Запуск проекта

Проект запускался в Google Colab.

Основные зависимости:

```python
pip install langchain
pip install langchain-community
pip install sentence-transformers
pip install faiss-cpu
pip install transformers
pip install rank_bm25
```

Для работы модели требуется GPU, я использовала L4.
