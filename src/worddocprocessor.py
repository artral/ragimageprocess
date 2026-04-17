from annotated_types import doc
import chromadb
from docx import Document
from sentence_transformers import SentenceTransformer
import uuid
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from torch import embedding


class WordDocumentProcessor:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection(name="word_documents",metadata={"hnsw:space": "cosine"}  )
    
    def process_document(self, file_path):
        doc = Document(file_path)
        chunks = []
        full_text = "\n\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])

        # Split the full text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100
            )
        chunk_texts = text_splitter.split_text(full_text)

        for ind,text in enumerate(chunk_texts):
            print(f"Processing paragraph {ind}: '{text[:30]}...'")  # Debug print
            if text.strip():
                chunks.append({
                "id": str(uuid.uuid4()),
                "text": text,
                "metadata": {
                    "source": file_path,
                    "paragraph_index": ind,
                    "style": "Normal"  # Placeholder - you might want to extract this from the original paragraph
                }
            })

        return chunks

    def embed_and_store(self, file_path):
        chunks = self.process_document(file_path)
        texts     = [c["text"]     for c in chunks]
        ids       = [c["id"]       for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        
        embeddings = self.model.encode(texts).tolist()
        text_list = [chunk["text"] for chunk in chunks]
        embeddings = self.model.encode(text_list,normalize_embeddings=True).tolist()
        
        # Store in ChromaDB
        doc_id = str(uuid.uuid4())
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )
        print(f"Document {file_path} processed and stored with ID: {doc_id}")
    
    def query(self, query_text, top_k=5):
        query_embedding = self.model.encode(query_text).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]


        )
        print(f"\n🔍 Top {top_k} results for: '{query_text}'\n" + "─" * 50)
        for i, (doc, meta, dist) in enumerate(zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
         )):
            matches = []
            for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
            ):
                matches.append({
                "text": doc,           # ← matching text
                "score": 1 - dist,
                "source": meta["source"],
                "style": meta["style"]
            })


            print(f"[{i+1}] Score: {1 - dist:.4f}")   # cosine → higher = better
            print(f"     Source : {meta['source']}")
            print(f"     Style  : {meta['style']}")
            print(f"doc value: {doc}")  # debug - see what doc actually is
            print(f"doc type: {type(doc)}")

            print()

        return results