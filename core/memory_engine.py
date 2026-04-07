import json
import os
from datetime import datetime
from uuid import uuid4

try:
    import chromadb
    from chromadb.utils import embedding_functions
except Exception:
    chromadb = None
    embedding_functions = None


class SentiaMemory:
    COLLECTION_NAME = "sentia_long_term_memory_local_v1"
    LEGACY_COLLECTION_NAMES = ("sentia_long_term_memory",)

    def __init__(self, base_dir):
        print("[System] 正在记忆数据库...")

        self.db_path = os.path.join(base_dir, "models", "memory_db")
        self.log_path = os.path.join(self.db_path, "memory_log.jsonl")
        self.local_embedding_model_dir = os.path.join(base_dir, "models", "memory_embedding")
        self.hf_home = os.path.join(self.db_path, "hf_home")
        self.use_vector_memory = False
        self.client = None
        self.collection = None
        self.embedding_fn = None
        self.is_ready = True

        os.makedirs(self.db_path, exist_ok=True)
        os.makedirs(self.hf_home, exist_ok=True)
        os.environ.setdefault("HF_HOME", self.hf_home)
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        if not os.path.exists(self.log_path):
            open(self.log_path, "a", encoding="utf-8").close()

        if chromadb is None or embedding_functions is None:
            print("[Memory] ChromaDB 或向量依赖不可用，切换到轻量记忆模式。")
            print(f"[Memory] 轻量记忆已就绪，当前已存储 {self._lightweight_count()} 条记录。")
            return

        if not os.path.isdir(self.local_embedding_model_dir):
            print("[Memory] 未检测到本地嵌入模型，跳过联网加载，使用轻量记忆模式。")
            print(f"[Memory] 如需向量检索，可将模型放到: {self.local_embedding_model_dir}")
            print(f"[Memory] 轻量记忆已就绪，当前已存储 {self._lightweight_count()} 条记录。")
            return

        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.local_embedding_model_dir,
                cache_folder=self.db_path,
                local_files_only=True,
            )
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=self.embedding_fn
            )
            migrated = self._migrate_legacy_collections_if_needed()
            count = self.collection.count()
            self.use_vector_memory = True
            if migrated:
                print(f"[Memory] 已迁移 {migrated} 条旧记忆到本地模型索引。")
            print(f"[Memory] 向量记忆已就绪！当前已存储 {count} 条长期记忆节点。")
        except Exception as e:
            print(f"[Warning] 向量记忆初始化失败，已降级到轻量记忆模式: {e}")
            print(f"[Memory] 轻量记忆已就绪，当前已存储 {self._lightweight_count()} 条记录。")

    def _migrate_legacy_collections_if_needed(self):
        if self.collection is None or self.client is None:
            return 0
        if self.collection.count() > 0:
            return 0

        migrated = 0
        existing_names = {collection.name for collection in self.client.list_collections()}
        for legacy_name in self.LEGACY_COLLECTION_NAMES:
            if legacy_name == self.COLLECTION_NAME or legacy_name not in existing_names:
                continue

            legacy_collection = self.client.get_collection(legacy_name)
            legacy_count = legacy_collection.count()
            if legacy_count == 0:
                continue

            data = legacy_collection.get(limit=legacy_count)
            ids = data.get("ids", [])
            documents = data.get("documents", [])
            metadatas = data.get("metadatas", [])
            if not ids or not documents:
                continue

            self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            migrated += len(ids)

        return migrated

    def _get_timestamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    def _lightweight_count(self):
        return len(self._load_lightweight_memories())

    def _load_lightweight_memories(self):
        if not os.path.exists(self.log_path):
            return []

        entries = []
        with open(self.log_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def _write_lightweight_memory(self, record):
        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _score_lightweight_memory(self, query_text, record):
        query = "".join(query_text.lower().split())
        document = "".join(record.get("document", "").lower().split())
        if not query or not document:
            return 0

        overlap = len(set(query) & set(document))
        contains_bonus = 5 if query in document else 0
        importance_bonus = int(record.get("importance", 1))
        return overlap + contains_bonus + importance_bonus

    def write_memory(self, event_description, emotion_tag="Neutral", importance=1):
        if not self.is_ready:
            return

        memory_id = f"mem_{uuid4().hex}"
        timestamp = self._get_timestamp()
        memory_text = f"[{timestamp}] [情绪:{emotion_tag}] {event_description}"

        if not self.use_vector_memory:
            record = {
                "id": memory_id,
                "timestamp": timestamp,
                "emotion": emotion_tag,
                "importance": importance,
                "document": memory_text,
            }
            try:
                self._write_lightweight_memory(record)
                if importance >= 4:
                    print(f"[Memory] 核心事件已永久归档: {memory_text}")
            except Exception as e:
                print(f"[Warning] 轻量记忆写入失败: {e}")
            return

        try:
            self.collection.add(
                documents=[memory_text],
                metadatas=[{"timestamp": timestamp, "emotion": emotion_tag, "importance": importance}],
                ids=[memory_id]
            )
            if importance >= 4:
                print(f"[Memory] 核心事件已永久归档: {memory_text}")
        except Exception as e:
            print(f"[Warning] 记忆写入失败: {e}")

    def recall_memory(self, query_text, n_results=2):
        if not self.is_ready:
            return ""

        if not self.use_vector_memory:
            records = self._load_lightweight_memories()
            if not records:
                return ""

            ranked_records = sorted(
                records,
                key=lambda record: (
                    self._score_lightweight_memory(query_text, record),
                    int(record.get("importance", 1)),
                    record.get("timestamp", ""),
                ),
                reverse=True,
            )
            top_records = [record for record in ranked_records if self._score_lightweight_memory(query_text, record) > 0][:n_results]
            if not top_records:
                return ""
            documents = [record["document"] for record in top_records]
            return "[检索到的长期历史记忆]：\n" + "\n".join(documents)

        if self.collection.count() == 0:
            return ""

        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=min(n_results, self.collection.count())
            )
            documents = results.get("documents", [[]])[0]
            if not documents:
                return ""
            return "[检索到的长期历史记忆]：\n" + "\n".join(documents)
        except Exception as e:
            print(f"[Warning] 记忆检索失败: {e}")
            return ""

    def get_last_shutdown_memory(self):
        if not self.is_ready:
            return ""

        if not self.use_vector_memory:
            records = self._load_lightweight_memories()
            for record in reversed(records):
                if int(record.get("importance", 1)) == 5:
                    last_event = record.get("document", "")
                    return f"[极其重要的历史情绪残留]：在你上次运行程序时，发生了这件事 -> {last_event}。你在说第一句话时，必须极其强烈地带入这个情绪进行回复！"
            return ""

        if self.collection.count() == 0:
            return ""

        try:
            results = self.collection.get(
                where={"importance": 5},
                limit=1
            )
            if results and results.get("documents"):
                last_event = results["documents"][-1]
                return f"[极其重要的历史情绪残留]：在你上次运行程序时，发生了这件事 -> {last_event}。你在说第一句话时，必须极其强烈地带入这个情绪进行回复！"
            return ""
        except Exception:
            return ""
