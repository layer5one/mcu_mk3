# /home/taylo/mcp-agent/memory_manager.py
import chromadb
from chromadb.utils import embedding_functions
import logging
import time
import os

logger = logging.getLogger("mcp_server")

class LocalMemoryManager:
    def __init__(self, device_id: str, persist_directory: str = "./chroma_db"):
        self.device_id = device_id
        self.persist_directory = persist_directory
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=f"{self.device_id}_memory",
                embedding_function=self.embedding_function
            )
            logger.info(f"ChromaDB collection '{self.device_id}_memory' loaded/created.")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.client = None
            self.collection = None

    def add_memory(self, text: str, metadata: dict = None) -> bool:
        if not self.collection:
            return False

        prefixed_text = f"{self.device_id}: {text}"
        doc_id = str(hash(prefixed_text))

        full_metadata = metadata or {}
        full_metadata['device_id'] = self.device_id
        full_metadata['timestamp'] = int(time.time())
        full_metadata['synced'] = 0

        try:
            self.collection.add(
                documents=[prefixed_text],
                metadatas=[full_metadata],
                ids=[doc_id]
            )
            logger.info(f"Added memory with ID {doc_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to add memory: {e}")
            return False

    def query_memory(self, query_text: str, n_results: int = 5) -> list:
        if not self.collection:
            return []
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            return results.get('documents', [])
        except Exception as e:
            logger.error(f"Failed to query memory: {e}")
            return []

    def get_unsynced_memories(self):
        if not self.collection:
            return [], {}

        results = self.collection.get(where={"synced": 0})
        return results.get('ids', []), results

    def mark_memories_as_synced(self, ids_to_update: list):
        if not self.collection or not ids_to_update:
            return

        try:
            existing_records = self.collection.get(ids=ids_to_update)

            new_metadatas = []
            for meta in existing_records['metadatas']:
                meta['synced'] = 1
                new_metadatas.append(meta)

            self.collection.update(
                ids=ids_to_update,
                metadatas=new_metadatas
            )
            logger.info(f"Marked {len(ids_to_update)} memories as synced.")
        except Exception as e:
            logger.error(f"Failed to mark memories as synced: {e}")

    def delete_old_memories(self, days_old: int = 7):
        if not self.collection:
            return
        threshold = int(time.time()) - (days_old * 86400)
        self.collection.delete(where={"timestamp": {"$lt": threshold}})
        logger.info(f"Pruned memories older than {days_old} days.")

    def embed_knowledge(self, file_path: str, chunk_size: int = 500):
        if not self.collection or not os.path.exists(file_path):
            return False
        try:
            with open(file_path, 'r') as f:
                text = f.read()
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            ids = [str(hash(chunk)) for chunk in chunks]
            metadatas = [{"type": "knowledge", "source": file_path, "timestamp": int(time.time()), "synced": 0} for _ in chunks]
            self.collection.add(documents=chunks, metadatas=metadatas, ids=ids)
            logger.info(f"Embedded knowledge from {file_path} into {len(chunks)} chunks.")
            return True
        except Exception as e:
            logger.error(f"Failed to embed knowledge: {e}")
            return False

    def query_knowledge(self, query_text: str, n_results: int = 3) -> list:
        return self.query_memory(query_text, n_results)
