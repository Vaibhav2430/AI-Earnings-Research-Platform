"""OpenAI embedding transport with explicit limits and validated vectors."""
import json
import math
import os
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class IndexError(Exception):
    """An actionable indexing or retrieval error."""


def validate_text(text):
    if not isinstance(text, str) or not text.strip():
        raise IndexError('Embedding input must be nonempty text.')
    # UTF-8 byte count is a conservative upper bound on byte-BPE token count.
    # Avoid a tokenizer dependency while staying below the 8,192-token limit.
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


class OpenAIEmbedder:
    model = 'text-embedding-3-small'
    dimensions = 1536
    provider = 'openai'

    def __init__(self, api_key=None):
        self.api_key = api_key if api_key is not None else os.environ.get('OPENAI_API_KEY', '')

    def embed(self, texts):
        if not self.api_key.strip():
            raise IndexError('Set OPENAI_API_KEY locally and export it before indexing or searching. Do not put the key in source code.')
        if not 1 <= len(texts) <= 16:
            raise IndexError('Embedding batches must contain 1 to 16 passages.')
        for text in texts:
            validate_text(text)
        body = json.dumps({'model': self.model, 'input': texts,
                           'dimensions': self.dimensions, 'encoding_format': 'float'}).encode()
        request = Request('https://api.openai.com/v1/embeddings', data=body, method='POST', headers={
            'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json',
        })
        payload = None
        for attempt in range(3):
            try:
                with urlopen(request, timeout=60) as response:
                    payload = json.load(response)
                break
            except HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise IndexError(f'OpenAI embedding request failed (HTTP {exc.code}); check credentials, quota, and API access.') from None
                retry = exc.headers.get('Retry-After', '') if exc.headers else ''
                delay = min(30, max(2 ** attempt, int(retry))) if retry.isdigit() else 2 ** attempt
            except (URLError, TimeoutError, OSError):
                if attempt == 2:
                    raise IndexError('Could not reach OpenAI. Check network access and trusted TLS certificates.') from None
                delay = 2 ** attempt
            except (ValueError, UnicodeError):
                raise IndexError('OpenAI returned invalid JSON.') from None
            time.sleep(delay)
        if not isinstance(payload, dict) or payload.get('model') != self.model:
            raise IndexError('OpenAI returned an unexpected embedding model.')
        items = payload.get('data')
        if not isinstance(items, list) or len(items) != len(texts):
            raise IndexError('OpenAI returned an unexpected number of embeddings.')
        ordered = [None] * len(texts)
        for item in items:
            if not isinstance(item, dict):
                raise IndexError('OpenAI returned an invalid embedding entry.')
            index = item.get('index')
            if type(index) is not int or not 0 <= index < len(texts) or ordered[index] is not None:
                raise IndexError('OpenAI returned invalid embedding indices.')
            ordered[index] = unit_vector(item.get('embedding'), self.dimensions)
        return ordered
