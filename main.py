import csv
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser

csv_file_path = "data.csv"

# source_column is useful to point to the 'main' field, but optional.
loader = CSVLoader(file_path=csv_file_path)
docs = loader.load()

# --- INSPECTION: See how LangChain formatted a row ---
print("\n--- Example of a Loaded Document (One Row) ---")
print(docs[1].page_content) 
print("----------------------------------------------\n")

# Note: We usually don't need to split text for CSVs if rows are small,
# but if fields contain massive text, you might still use a splitter.
# Here, we pass 'docs' directly to the vector store.
local_llm = "llama3.2"
local_embedding = "nomic-embed-text"

vectorstore = Chroma.from_documents(
    documents=docs, 
    embedding=OllamaEmbeddings(model=local_embedding)
)
retriever = vectorstore.as_retriever(k=5)

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOllama(model=local_llm, temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain_with_sources = (
    RunnableMap({
        "context": retriever,
        "question": RunnablePassthrough()
    })
    | RunnableMap({
        "context_string": lambda x: format_docs(x["context"]),
        "answer": (
            prompt.partial(question=RunnablePassthrough())
            | model 
            | StrOutputParser()
        ),
        "source_documents": lambda x: x["context"]
    })
)

question = "What emojis do I use to express feeling sad, depressed and anxious?"
print(f"Question: {question}\n")

result = rag_chain_with_sources.invoke(question)

print(f"Answer: {result['answer']}\n")

print("🔎 SOURCE ROW USED:")
for doc in result['source_documents']:
    print(doc.page_content)