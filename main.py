import csv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

csv_file_path = "data.csv"
loader = CSVLoader(file_path=csv_file_path)
docs = loader.load()

local_llm = "llama3.2"
local_embedding = "nomic-embed-text"

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OllamaEmbeddings(model=local_embedding)
)

# Retrieve more candidates than you need — re-ranker will cut these down
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

model = ChatOllama(model=local_llm, temperature=0)


# --- Query Parser (fixed from before) ---
query_parser_prompt = ChatPromptTemplate.from_template("""
You are a search query optimizer for an emoji database.

Given a user's question, extract the key concepts as a plain text search query
that will be used to search emoji names, descriptions, and tags.

Rules:
- Output ONLY plain text words, NO emoji characters
- Focus on emotions, synonyms, and descriptive terms
- Keep it concise (under 10 words)

Example:
User question: "What emoji should I use when I'm really happy?"
Optimized search query: happy joyful excited cheerful face smile

User question: {question}
Optimized search query:
""")

query_parser_chain = query_parser_prompt | model | StrOutputParser()


# --- Re-ranker ---
rerank_prompt = ChatPromptTemplate.from_template("""
You are evaluating how relevant an emoji is to a user's query.

User query: {query}

Emoji information:
{document}

Score the relevance of this emoji to the query on a scale of 0 to 10.
- 10 means the emoji is a perfect match for the query
- 0 means the emoji is completely irrelevant

Reply with ONLY a single integer between 0 and 10. No explanation.
Score:
""")

rerank_chain = rerank_prompt | model | StrOutputParser()



def rerank_documents(query: str, documents: list[Document], top_n: int = 3):
    pairs = [[query, doc.page_content] for doc in documents]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(scores, documents), reverse=True)
    return [doc for _, doc in ranked[:top_n]]

# --- Full RAG Chain with Query Parsing + Re-ranking ---
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def parse_and_log_query(question: str) -> str:
    parsed = query_parser_chain.invoke({"question": question})
    print(f"  Original query : {question}")
    print(f"  Parsed query   : {parsed}\n")
    return parsed


rag_chain_with_reranking = (
    RunnableMap({
        "parsed_query": RunnableLambda(parse_and_log_query),
        "original_question": RunnablePassthrough()
    })
    | RunnableMap({
        # Retrieve a wider candidate set
        "context": lambda x: rerank_documents(
            query=x["original_question"],
            documents=retriever.invoke(x["parsed_query"]),
            top_n=3
        ),
        "question": lambda x: x["original_question"]
    })
    | RunnableMap({
        "answer": (
            RunnableLambda(lambda x: prompt.invoke({
                "context": format_docs(x["context"]),
                "question": x["question"]
            }))
            | model
            | StrOutputParser()
        ),
        "source_documents": lambda x: x["context"]
    })
)


# --- Run ---
question = "What emojis do I use to express feeling sad, depressed and anxious?"
print(f"Question: {question}\n")
print("🔍 Query Parsing:")

result = rag_chain_with_reranking.invoke(question)

print(f"\nAnswer: {result['answer']}\n")
print("🔎 SOURCE ROWS USED:")
for doc in result["source_documents"]:
    print(doc.page_content)
