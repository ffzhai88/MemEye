from benchmark import embeddings
from benchmark.embeddings import MultimodalEmbedder, TextEmbedder
from benchmark.retrieval import clear_retriever_cache


def test_text_embedder_reuses_shared_backend_without_reloading():
    model = object()
    key = ("sentence_transformers", "test-model")
    embeddings._SHARED_EMBEDDING_BACKENDS[key] = {"model": model}
    wrapper = TextEmbedder("test-model")
    wrapper._load()
    assert wrapper._model is model
    embeddings.clear_embedding_model_cache()


def test_multimodal_embedder_reuses_shared_backend_without_reloading():
    model, processor = object(), object()
    key = ("siglip", "test-siglip")
    embeddings._SHARED_EMBEDDING_BACKENDS[key] = {
        "model": model, "processor": processor, "device": "cuda"
    }
    wrapper = MultimodalEmbedder("test-siglip")
    wrapper._load()
    assert wrapper._model is model
    assert wrapper._processor is processor
    embeddings.clear_embedding_model_cache()


def test_retriever_cleanup_can_retain_or_release_shared_models():
    key = ("test", "model")
    embeddings._SHARED_EMBEDDING_BACKENDS[key] = {"model": object()}
    clear_retriever_cache(keep_embedding_models=True)
    assert key in embeddings._SHARED_EMBEDDING_BACKENDS
    clear_retriever_cache()
    assert not embeddings._SHARED_EMBEDDING_BACKENDS
