# AI Earnings Research Platform

An AI-powered web app that turns a company search into an earnings research report and a source-backed conversation.

## What we are building

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
| Vector storage | Chroma for a local prototype or Pinecone for a hosted service |
| Web interface | Streamlit for the initial prototype |
| Alternative web deployment | A custom frontend hosted on Vercel with a Python backend |

## Initial scope

Start with selected US public companies and their latest available earnings materials. The first working version should support company search, automatic retrieval, a cited earnings report, and follow-up Q&A. Broader company coverage and historical comparisons can follow once the core workflow is reliable.

## Project status

This repository currently documents the project vision. The application, data pipeline, and deployment have not been implemented yet. The features above describe the intended product.

## What this project demonstrates

The project brings together automated data ingestion, document processing, semantic retrieval, language model integration, and financial research in an end-to-end web application. Its central goal is to make source-backed earnings research accessible through a simple company search.
