from sentence_transformers import SentenceTransformer
from transformers import pipeline
import chromadb

# Load embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load vector store
client = chromadb.Client(
    chromadb.config.Settings(persist_directory="vector_store")
)
collection = client.get_collection("complaints")

# Load LLM
llm = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct",
    max_new_tokens=300
)

PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust.
Answer the question ONLY using the context below.
If the answer is not in the context, say you do not have enough information.

Context:
{context}

Question:
{question}

Answer:
"""

def rag_answer(question, k=5):
    query_embedding = embed_model.encode([question])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=k
    )

    context = "\n\n".join(results["documents"][0])

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    response = llm(prompt)[0]["generated_text"]
    return response, results["documents"][0]
