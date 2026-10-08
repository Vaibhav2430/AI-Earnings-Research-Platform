"""Local, offline embedding inference. Only the download command uses the network."""
import math
import os
from pathlib import Path

class IndexError(Exception):
    """An actionable indexing or retrieval error."""


def validate_text(text):
    if not isinstance(text, str) or not text.strip():
        raise IndexError('Embedding input must be nonempty text.')
    # Bound resource use. Long passages are embedded using token windows below.
    if len(text.encode('utf-8')) > 8000:
        raise IndexError('Input exceeds the conservative 8,000-byte limit. Rechunk with a smaller --max-chars value.')


def unit_vector(vector, dimensions):
    if not isinstance(vector, list) or len(vector) != dimensions:
        raise IndexError('Embedding vector has an unexpected dimension.')
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in vector):
        raise IndexError('Embedding vector contains invalid values.')
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0:
        raise IndexError('Embedding vector has an invalid norm.')
    return [x / norm for x in vector]


MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
MODEL_REVISION = '1110a243fdf4706b3f48f1d95db1a4f5529b4d41'


def token_windows(token_ids, size):
    if size < 1:
        raise IndexError('Invalid model context length.')
    return [token_ids[i:i + size] for i in range(0, len(token_ids), size)]


class LocalEmbedder:
    provider = 'local-sentence-transformers'
    # Include pooling strategy and revision so incompatible cached vectors never mix.
    model = f'{MODEL_ID}@{MODEL_REVISION}:token-window-weighted-mean-v1'
    dimensions = 384

    def __init__(self, cache_dir=Path('data/.models')):
        self.cache_dir = Path(cache_dir)
        self._encoder = None

    def load(self, download=False):
        if self._encoder is not None:
            return self._encoder
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise IndexError('Install local embedding dependencies with: python -m pip install -e ".[local]"') from exc
        try:
            encoder = SentenceTransformer(
                MODEL_ID, revision=MODEL_REVISION, cache_folder=str(self.cache_dir),
                local_files_only=not download, trust_remote_code=False, device='cpu',
                model_kwargs={'use_safetensors': True},
            )
        except (OSError, ValueError, RuntimeError) as exc:
            raise IndexError('Cannot load the local model. Run: python -m earnings_collector.indexing download-model') from exc
        dimension_method = getattr(encoder, 'get_embedding_dimension', None)
        if dimension_method is None:
            dimension_method = encoder.get_sentence_embedding_dimension
        if dimension_method() != self.dimensions:
            raise IndexError('Local model dimensions do not match the index configuration.')
        encoder.eval()
        self._encoder = encoder
        return encoder

    def embed(self, texts):
        if not texts:
            raise IndexError('Provide at least one passage.')
        for text in texts:
            validate_text(text)
        encoder = self.load()
        import torch
        tokenizer = encoder.tokenizer
        capacity = encoder.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
        outputs = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
            windows = token_windows(ids, capacity)
            if not windows:
                raise IndexError('Input contains no encodable tokens.')
            total = torch.zeros(self.dimensions)
            # Every token is covered: avoid MiniLM's default silent truncation.
            for start in range(0, len(windows), 16):
                batch = windows[start:start + 16]
                features = [tokenizer.prepare_for_model(w, add_special_tokens=True,
                            truncation=False, return_attention_mask=True, verbose=False) for w in batch]
                tensors = tokenizer.pad(features, padding=True, return_tensors='pt', verbose=False)
                with torch.inference_mode():
                    vectors = encoder(dict(tensors))['sentence_embedding']
                for vector, window in zip(vectors, batch):
                    total += vector.cpu() * len(window)
            outputs.append(unit_vector(total.tolist(), self.dimensions))
        return outputs
