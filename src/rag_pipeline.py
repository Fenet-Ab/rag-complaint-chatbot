import pandas as pd
from sklearn.model_selection import train_test_split
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm

# Load cleaned data
df = pd.read_csv("data/processed/filtered_complaints.csv")

# Stratified sampling
sample_df, _ = train_test_split(
    df,
    train_size=12000,
    stratify=df["product"],
    random_state=42
)

print("Sample size:", sample_df.shape)

# Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

documents = []
metadatas = []

for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
    chunks = splitter.split_text(row["cleaned_narrative"])
    for i, chunk in enumerate(chunks):
        documents.append(chunk)
        metadatas.append({
            "complaint_id": row["complaint_id"],
            "product": row["product"],
            "chunk_index": i
        })

# Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents, show_progress_bar=True)

# Vector store
client = chromadb.Client(
    chromadb.config.Settings(persist_directory="vector_store")
)

collection = client.get_or_create_collection(name="complaints")

collection.add(
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
    ids=[str(i) for i in range(len(documents))]
)

client.persist()
print("Vector store saved to vector_store/")
