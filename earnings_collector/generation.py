"""Apple Silicon local language model. Downloads are explicit; inference is offline."""
import json
import os
import platform
from pathlib import Path

MODEL_ID = 'mlx-community/Qwen2.5-1.5B-Instruct-4bit'


class GenerationError(Exception):
    pass


class LocalGenerator:
    def __init__(self, data_dir=Path('data')):
        self.cache = Path(data_dir) / '.language-model'
        self.lock = self.cache / 'model-lock.json'
        self.model = None
        self.tokenizer = None
        self.revision = None

    def download(self):
        from huggingface_hub import HfApi, snapshot_download
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        revision = HfApi().model_info(MODEL_ID).sha
        snapshot_download(MODEL_ID, revision=revision, cache_dir=str(self.cache),
                          allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'])
        self.cache.mkdir(parents=True, exist_ok=True)
        info = {'model_id': MODEL_ID, 'revision': revision}
        self.lock.write_text(json.dumps(info, indent=2) + '\n')
        return info

    def load(self):
        if self.model is not None:
            return
        if platform.system() != 'Darwin' or platform.machine() != 'arm64':
            raise GenerationError('Local generation currently requires an Apple Silicon Mac.')
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        try:
            from huggingface_hub import snapshot_download
            from mlx_lm import load
            info = json.loads(self.lock.read_text())
            if info['model_id'] != MODEL_ID:
                raise GenerationError('Unexpected model identity in local model lock.')
            path = snapshot_download(MODEL_ID, revision=info['revision'], cache_dir=str(self.cache), local_files_only=True)
            self.model, self.tokenizer = load(path, tokenizer_config={'trust_remote_code': False})
            self.revision = info['revision']
        except ImportError as exc:
            if 'Metal device' in str(exc):
                raise GenerationError('MLX needs access to the Mac GPU. Run this command in a local terminal outside a GPU-restricted sandbox.') from exc
            raise GenerationError('Install dependencies with: python -m pip install -e ".[generation]"') from exc
        except (OSError, ValueError, KeyError) as exc:
            raise GenerationError('Install .[generation] dependencies and run the answers download-model command first.') from exc

    def generate(self, messages):
        self.load()
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if len(self.tokenizer.encode(prompt)) > 6500:
            raise GenerationError('Evidence exceeds the local context budget. Use fewer passages with --top-k.')
        return generate(self.model, self.tokenizer, prompt=prompt, max_tokens=550,
                        sampler=make_sampler(temp=0), verbose=False)
