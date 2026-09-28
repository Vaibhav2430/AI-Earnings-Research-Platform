"""Split collected text into bounded, source-traceable passages for later indexing."""
import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

VERSION = '1'


class ChunkingError(Exception):
    pass


def digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def is_heading(line):
    text = line.strip()
    return bool(text and '\t' not in text and len(text) <= 160 and (
        re.match(r'^(?:PART\s+[IVX]+|ITEM\s+\d+[A-Z]?[.:\s])', text, re.I)
        or (len(text) >= 4 and any(c.isalpha() for c in text) and text.isupper())
    ))


def spans(text, max_chars):
    """Keep lines intact where possible and attach headings to subsequent content.

    Offsets use Python Unicode character indices, with an exclusive end. Oversized
    lines are split at whitespace, or at the limit if no whitespace is available.
    Every source character belongs to exactly one passage.
    """
    if max_chars < 200:
        raise ChunkingError('max_chars must be at least 200')
    lines = []
    offset = 0
    for line in text.splitlines(keepends=True):
        lines.append((offset, offset + len(line), is_heading(line)))
        offset += len(line)
    # Group consecutive headings with the following line when it fits.
    blocks = []
    i = 0
    while i < len(lines):
        start, end, heading = lines[i]
        i += 1
        if heading:
            while i < len(lines) and end - start + lines[i][1] - lines[i][0] <= max_chars:
                _, end, next_heading = lines[i]
                i += 1
                if not next_heading:
                    break
        blocks.append((start, end))
    pieces = []
    for start, end in blocks:
        split = end - start > max_chars
        while end - start > max_chars:
            limit = start + max_chars
            whitespace = list(re.finditer(r'\s+', text[start:limit]))
            cut = start + whitespace[-1].end() if whitespace else limit
            if cut <= start:
                cut = limit
            pieces.append((start, cut, True))
            start = cut
        if end > start:
            pieces.append((start, end, split))
    pending = None
    for start, end, split in pieces:
        if pending is not None and end - pending[0] <= max_chars:
            pending = (pending[0], end, pending[2] or split)
        else:
            if pending is not None:
                yield pending
            pending = (start, end, split)
    if pending is not None:
        yield pending


def chunk_company(ticker, data_dir=Path('data'), max_chars=2400):
    ticker = ticker.strip().upper()
    if not re.fullmatch(r'[A-Z0-9][A-Z0-9.-]{0,14}', ticker):
        raise ChunkingError('Enter a valid ticker, such as AAPL.')
    if max_chars < 200:
        raise ChunkingError('max_chars must be at least 200')
    root = (Path(data_dir) / ticker).resolve()
    try:
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ChunkingError(f'Cannot read {root / "manifest.json"}: {exc}') from exc
    if not isinstance(manifest, dict) or manifest.get('ticker') != ticker or not isinstance(manifest.get('documents'), list):
        raise ChunkingError('Manifest must match the ticker and contain a documents list.')
    chunks, warnings, seen = [], [], set()
    processed = 0
    for index, doc in enumerate(manifest['documents']):
        label = f'Document {index + 1}'
        if not isinstance(doc, dict):
            warnings.append(f'{label}: invalid metadata; skipped.')
            continue
        relative = doc.get('text_path')
        if not relative:
            warnings.append(f'{label}: no extracted text; skipped.')
            continue
        if not isinstance(relative, str):
            warnings.append(f'{label}: invalid text path; skipped.')
            continue
        path = (root / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(root):
            warnings.append(f'{label}: text path escapes company directory; skipped.')
            continue
        if not all(isinstance(doc.get(k), str) and doc[k] for k in ('source_url', 'accession_number')):
            warnings.append(f'{label}: missing source URL or accession number; skipped.')
            continue
        try:
            # Decode bytes directly to preserve CRLF and exact character locations.
            text = path.read_bytes().decode('utf-8')
        except (OSError, UnicodeError) as exc:
            warnings.append(f'{label}: cannot read text ({exc}); skipped.')
            continue
        if not text.strip():
            warnings.append(f'{label}: empty text; skipped.')
            continue
        doc_id = digest(ticker + '\n' + doc['accession_number'] + '\n' + doc['source_url'])
        if doc_id in seen:
            warnings.append(f'{label}: duplicate source; skipped.')
            continue
        seen.add(doc_id)
        processed += 1
        text_hash = digest(text)
        source = {k: doc.get(k) for k in (
            'kind', 'form', 'accession_number', 'filing_date', 'sec_report_date',
            'reporting_period_end', 'source_url', 'raw_path', 'text_path', 'sha256',
            'classification', 'extraction_status',
        )}
        source.update({'ticker': ticker, 'company_name': manifest.get('company_name'),
                       'cik': manifest.get('cik'), 'text_sha256': text_hash})
        for number, (start, end, split) in enumerate(spans(text, max_chars)):
            chunk_id = digest(f'{VERSION}:{max_chars}:{doc_id}:{text_hash}:{start}:{end}')
            chunks.append({
                'id': chunk_id, 'document_id': doc_id, 'chunk_index': number,
                'text': text[start:end], 'source': source,
                'location': {'char_start': start, 'char_end': end,
                             'line_start': text.count('\n', 0, start) + 1,
                             'line_end': text.count('\n', 0, max(start, end - 1)) + 1},
                'oversized_line_split': split,
            })
    result = {
        'schema_version': 1, 'chunker_version': VERSION, 'ticker': ticker,
        'configuration': {'max_chars': max_chars, 'overlap_chars': 0, 'length_unit': 'Unicode characters'},
        'status': 'empty' if not chunks else ('partial' if warnings else 'complete'),
        'documents_processed': processed, 'chunk_count': len(chunks),
        'collection_status': manifest.get('status'),
        'collection_warnings': manifest.get('warnings', []),
        'warnings': warnings, 'chunks': chunks,
    }
    output = root / 'chunks.json'
    # Replace, never append: repeated runs cannot accumulate stale passages.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=root, delete=False) as handle:
            temp_path = Path(handle.name)
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
        os.replace(temp_path, output)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return output, result


def main():
    parser = argparse.ArgumentParser(description='Chunk collected documents with source metadata. No API key or network needed.')
    parser.add_argument('ticker')
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--max-chars', type=int, default=2400)
    args = parser.parse_args()
    try:
        path, result = chunk_company(args.ticker, args.data_dir, args.max_chars)
    except (ChunkingError, OSError) as exc:
        print(f'Chunking failed: {exc}', file=sys.stderr)
        return 1
    print(f"{result['ticker']}: {result['status']}. Saved {result['chunk_count']} chunks from {result['documents_processed']} documents.")
    print(f'Output: {path}')
    for warning in result['warnings']:
        print(f'Note: {warning}')
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    sys.exit(main())
