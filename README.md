# AI Earnings Research Platform

An AI-powered web app that turns a company search into an earnings research report and a source-backed conversation.

## What am I building

The goal is simple: search for a company by name or ticker, describe what you want to know, and get a report based on relevant earnings information. Users should not need to find transcripts, download filings, or upload documents themselves.

The platform will automatically discover and retrieve available earnings call transcripts, earnings releases, and company filings. It will use retrieval-augmented generation (RAG) to answer questions using those sources and provide citations so users can check the underlying evidence.

## Intended user experience

1. Search for a company by name or stock ticker.
2. Request an overview of its latest earnings or ask a specific question about its performance.
3. Let the platform gather and process relevant source material automatically.
4. Receive a structured report with financial results, business highlights, management commentary, guidance, and risks where supported by the sources.
5. Ask follow-up questions in a chat interface and explore the supporting evidence.

Example questions:

- How did revenue and earnings change compared with the same quarter last year?
- Which business segments drove growth?
- What did management say about demand and margins?
- How has the company's guidance changed?
- What risks did management discuss during the latest earnings call?

## Planned capabilities

- **Automatic source collection:** Retrieve available materials from company investor relations pages, public filings, and permitted data providers.
- **Earnings reports:** Summarize key metrics, reported results, management commentary, and outlook in one place.
- **Source-backed Q&A:** Retrieve relevant passages before generating answers and link claims to their sources.
- **Period comparisons:** Compare quarterly and year-over-year results when comparable data is available.
- **Clear source context:** Identify the reporting period, publication date, and source for the information used.
- **Transparent data gaps:** Explain when a transcript, metric, or historical comparison is unavailable instead of inventing an answer.
- **Reusable research:** Cache retrieved documents and their embeddings to support faster follow-up questions.

## How it will work

```text
Company search and research question
                 |
                 v
Resolve company and reporting period
                 |
                 v
Automatically retrieve available earnings materials
                 |
                 v
Extract text, preserve source metadata, and split into passages
                 |
                 v
Create embeddings and index passages in a vector database
                 |
                 v
Retrieve relevant evidence for the user's question
                 |
                 v
Generate a report or chat answer with citations
```

Source access and coverage will vary by company and provider. The ingestion pipeline will use permitted sources and report missing materials clearly. Uploaded documents will not be required for the core workflow.

## Proposed technology stack

The initial stack is still being finalized:

| Layer | Proposed technology |
| --- | --- |
| Data collection and processing | Python |
| Retrieval and orchestration | LangChain or LlamaIndex |
| Embeddings and answer generation | OpenAI API |
| Vector storage | SQLite with exact cosine search in the prototype; a hosted vector database can follow |
| Web interface | Streamlit for the initial prototype |
| Alternative web deployment | A custom frontend hosted on Vercel with a Python backend |

## Initial scope

Start with selected US public companies and their latest available earnings materials. The first working version should support company search, automatic retrieval, a cited earnings report, and follow-up Q&A. Broader company coverage and historical comparisons can follow once the core workflow is reliable.

## Project status

The first data collection prototype is implemented as a Python command-line tool. It resolves a ticker using the SEC company directory, downloads the latest available 10-K and 10-Q from recent submissions, and inspects Item 2.02 8-K filings for earnings release exhibits. The web application, AI reports, chat, and deployment are planned features.

## Run the collector

Requires Python 3.11 or newer. No third-party packages or AI API keys are required.

The SEC requires automated requests to identify the application and a contact. Set your own real contact email locally:

```sh
export SEC_USER_AGENT="AI Earnings Research your-email@example.com"
python3 -m earnings_collector AAPL
```

If you have saved the setting in the ignored `.env` file, load it first:

```sh
set -a
source .env
set +a
python3 -m earnings_collector AAPL
```

The CLI does not automatically load `.env`. Never commit your contact configuration. The request header is sent to the SEC, but is not included in output manifests.

If Python reports `CERTIFICATE_VERIFY_FAILED` on macOS, configure a trusted certificate bundle rather than disabling certificate checks. For a Mac with `/etc/ssl/cert.pem`, you can export `SSL_CERT_FILE=/etc/ssl/cert.pem` or add that setting to your local `.env`.

Optional arguments:

```sh
python3 -m earnings_collector AAPL --output data --max-earnings-filings 8
python3 -m earnings_collector AAPL --refresh
```

The collector stores its output under `data/AAPL/`:

- `manifest.json`: Company identity, source URLs, filing dates, period metadata, checksums, extraction status, and coverage warnings.
- `submissions.json`: The SEC discovery response used by the run.
- One folder per accession number containing original documents and extracted `.txt` files. Inspected earnings filing indexes are also preserved.

Archived document downloads are cached under `data/.cache/`; discovery metadata is fetched on every run. `--refresh` bypasses the document cache. Repeated runs replace the current manifest and may retain older document folders. All downloaded data is ignored by Git.

Exit codes: `0` means the scoped collection completed, `2` means partial coverage with a saved manifest, and `1` means collection failed before completion. Argument errors also use exit code `2` and print usage without a manifest.

## Current collection limits

- Discovery searches recent SEC submissions only, not older history shards.
- The latest 10-K, latest 10-Q, and earnings exhibits can cover different periods. They are not represented as a single matched quarter.
- Earnings release classification uses a conservative text heuristic. Matches are labeled `earnings_release_candidate`, not verified reports.
- An 8-K report date is an event date. Earnings exhibit period ends remain unset until they can be validated.
- HTML and text are extracted. Other exhibits are saved as originals with an unsupported extraction status. Table text retains basic row and cell separation; it is not a structured financial dataset.
- Amended filings, transcripts, investor relations fallbacks, and financial metric normalization are not implemented yet.
- SEC access denials and missing documents are reported explicitly. The collector does not bypass access restrictions.
- Requests are paced at no more than four per second per process, with bounded retries for transient failures. Run one collector process at a time to avoid multiplying request rates.

## Chunk documents for retrieval

After collecting a company, turn its extracted text into passages:

```sh
python3 -m earnings_collector.chunking AAPL
python3 -m earnings_collector.chunking AAPL --data-dir data --max-chars 2400
```

This step runs locally without a network connection, contact configuration, or API key. It reads only documents listed in the company's current `manifest.json` and writes `data/AAPL/chunks.json`. It does not create embeddings or generate AI answers yet.

Each passage includes:

- A stable chunk ID and document ID.
- The exact extracted source text, with company, ticker, CIK, document type, accession number, and source URL.
- Filing date, SEC report date, and reporting period end when available. Unknown periods remain null.
- Original-document and extracted-text checksums, plus original file locations.
- Zero-based character offsets with an exclusive end, and one-based inclusive line numbers in the extracted text file. These are not PDF page numbers or HTML offsets.

The default maximum is 2,400 Unicode characters, not tokens. The chunker keeps lines and table rows intact when they fit, and uses a heading heuristic for uppercase headings and SEC item/part headings. It groups detected headings with following content where possible. Oversized lines are split at whitespace or the character limit and flagged with `oversized_line_split`. It does not reconstruct HTML sections, carry table headers forward, or guarantee complete tables in one passage.

Passages are consecutive and have no overlap. Source text can be reconstructed by joining a document's chunks in order. Repeated runs replace the output atomically rather than appending duplicates. Identical input and settings produce identical output and IDs; changing source text, chunk size, or chunker version changes the affected IDs. A future indexer must remove obsolete IDs when replacing a document.

Missing, unsupported, empty, unsafe, or duplicate document entries are reported in output warnings. Collection warnings remain available separately, so successful chunking does not imply complete earnings coverage. Exit code `0` means chunking completed, `2` means partial or empty output, and `1` means failure. Argument errors also use `2` without writing output.

## Embed and search passages

The embedding client uses OpenAI `text-embedding-3-small` with 1,536 dimensions. Vectors and passage metadata are stored in `data/AAPL/index.sqlite3`. This prototype uses exact cosine search over locally stored vectors, without a database server or additional Python dependencies. See the [official embedding documentation](https://developers.openai.com/api/docs/guides/embeddings).

First, validate the passages and inspect the work required without API calls or database writes:

```sh
python3 -m earnings_collector.indexing index AAPL --dry-run
```

For live indexing, put `OPENAI_API_KEY="your-key"` in your ignored local `.env` file. Use a real API key privately, not in chat, source code, or GitHub. Load the configuration and run:

```sh
set -a
source .env
set +a
python3 -m earnings_collector.indexing index AAPL
python3 -m earnings_collector.indexing search AAPL "What drove revenue growth?" --top-k 5
```

Indexing sends passage text to OpenAI and search sends the question to OpenAI to create a query embedding. API usage is billed to your API account. Source metadata stays in the local database. Dry-run byte counts are not token counts or cost estimates.

Search outputs JSON containing ranked passages, cosine similarity scores, and all original citation metadata. Scores measure similarity, not factual confidence. This step returns evidence, not a generated answer, and does not infer which quarter the question means. Filter explicitly where needed:

```sh
python3 -m earnings_collector.indexing search AAPL "What were the main risks?" --form 10-K --top-k 3
python3 -m earnings_collector.indexing search AAPL "Revenue growth" --period-end 2026-06-27
```

Period filters require an exact known reporting period end. Exhibits whose periods are unknown do not match. Results retain collection and chunking warnings. Both commands accept `--data-dir` for a custom collection directory.

Indexing behavior:

- Embeddings are cached by provider, model, dimensions, and passage text hash. An unchanged rerun does not request embeddings again.
- Successful batches are cached even if a later request fails, allowing the next run to resume. The searchable snapshot changes only after all vectors are ready.
- Refreshing replaces all active passages for that company, including source metadata, and removes stale passages from search. Older cached vectors remain available for reuse.
- Search requires the current `chunks.json` to match the indexed snapshot. After recollecting documents, rerun chunking and indexing in that order.
- Requests use batches of at most 16 passages and bounded retries for transient failures. Inputs are conservatively capped at 8,000 UTF-8 bytes each. Oversized text is rejected before requests; rechunk with a smaller `--max-chars` value.
- Run one index writer per company at a time. Exact search loads that company's vectors into memory and is intended for a small prototype, not a large production corpus.

Both commands return exit code `0` on success and `1` on operational failure; argument errors use `2`. Indexing may successfully index partial source coverage, which remains explicit in search output.

The API transport, cache, refresh, filtering, and ranking are tested with synthetic embedding responses. Live indexing and semantic retrieval quality require an API key and a separate integration check. No live embedding run is implied by a successful dry run.

## Tests

```sh
python3 -m unittest discover -s tests -v
```

The tests use synthetic SEC responses and require no network access. GitHub Actions runs the suite on pushes and pull requests. A live collection is a separate integration check.

## What this project demonstrates

The project brings together automated data ingestion, document processing, semantic retrieval, language model integration, and financial research in an end-to-end web application. Its central goal is to make source-backed earnings research accessible through a simple company search.
