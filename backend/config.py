from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    chroma_persist_dir: str = "./chroma_data"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    default_top_k: int = 5
    max_document_size_kb: int = 500

    model_config = {"env_prefix": "RAG_"}


settings = Settings()
