from sklearn.model_selection import train_test_split
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb


sample_df, _ = train_test_split(
    df,
    train_size=12000,
    stratify=df['product'],
    random_state=42
)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = []
for _, row in sample_df.iterrows():
    chunks = splitter.split_text(row['cleaned_narrative'])
    for i, chunk in enumerate(chunks):
        docs.append({
            "text": chunk,
            "metadata": {
                "complaint_id": row['complaint_id'],
                "product": row['product'],
                "chunk_index": i
            }
        })

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.create_collection(name="complaints")

texts = [d["text"] for d in docs]
metadatas = [d["metadata"] for d in docs]
embeddings = model.encode(texts, show_progress_bar=True)

collection.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=[str(i) for i in range(len(texts))]
)
client.persist()
