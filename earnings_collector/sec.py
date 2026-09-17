"""Small SEC client with bounded retries, throttling, and immutable-file caching."""
import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class CollectionError(Exception):
    pass


class SecClient:
    def __init__(self, user_agent, cache_dir, refresh=False):
        self.user_agent = user_agent
        self.cache_dir = Path(cache_dir)
        self.refresh = refresh
        self.last_request = 0.0

    def get(self, url, immutable=False):
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.hostname not in {'www.sec.gov', 'data.sec.gov'}:
            raise CollectionError(f'Unsupported source URL: {url}')
        cache = self.cache_dir / hashlib.sha256(url.encode()).hexdigest()
        if immutable and cache.exists() and not self.refresh:
            return cache.read_bytes()
        for attempt in range(3):
            time.sleep(max(0, 0.25 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                request = Request(url, headers={'User-Agent': self.user_agent, 'Accept-Encoding': 'identity'})
                with urlopen(request, timeout=30) as response:
                    data = response.read()
                if immutable:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_bytes(data)
                return data
            except HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                    raise CollectionError(f'SEC returned HTTP {exc.code} for {url}') from exc
                retry_after = exc.headers.get('Retry-After', '')
                delay = min(30, max(2 ** attempt, int(retry_after))) if retry_after.isdigit() else 2 ** attempt
            except (URLError, TimeoutError, OSError) as exc:
                if attempt == 2:
                    raise CollectionError(f'Could not retrieve {url}: {exc}') from exc
                delay = 2 ** attempt
            time.sleep(delay)
        raise CollectionError(f'Could not retrieve {url}')

    def json(self, url):
        try:
            return json.loads(self.get(url))
        except (ValueError, UnicodeError) as exc:
            raise CollectionError(f'Invalid JSON from {url}') from exc
