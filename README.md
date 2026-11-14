# Local Restaurant Review RAG (Ollama + LangChain + ChromaDB)

A fully local, privacy-friendly retrieval-augmented generation (RAG) assistant that answers questions about restaurant reviews stored in CSV files. Nothing leaves your machine.

| Role | Model |
| --- | --- |
| Answer generation | `qwen3-coder:30b` |
| Embeddings (CSV review text and questions to vectors) | `mxbai-embed-large` |

## Features

- **Multiple restaurants and datasets.** Drop any number of CSV files into `data/`. Each file is one restaurant (or several, via a `Restaurant` column).
- **Hybrid retrieval.** Semantic search (ChromaDB, with MMR for diverse results) is combined with BM25 keyword search and fused with Reciprocal Rank Fusion. Exact words such as "gluten free" or "parking" are not missed.
- **Smart scoping.** The restaurant named in your question is detected automatically, or fix one with `/use`. Questions about "complaints" favour low-star reviews and questions about "praise" favour high-star reviews.
- **Follow-up questions.** Short conversation memory plus query rewriting, so "and is it good for kids?" works after a question about Sakura.
- **Real statistics.** Review counts, average ratings and star distributions are computed with pandas and given to the model. It does not have to guess them from five excerpts.
- **Grounded answers.** The model cites reviews as `[R1]`, `[R2]` and admits when the evidence is insufficient. Use `--sources` or `/sources on` to see the retrieved reviews.
- **Incremental indexing.** Only new or changed CSV files are re-embedded. Deleted files are removed from the index.
- **Streaming output** and helpful error messages if Ollama is not running.

## Project layout

```text
main.py        Chat loop, commands, query rewriting, prompt, streaming answers
retriever.py   Hybrid retriever (vector MMR + BM25 + RRF), restaurant/sentiment detection
vector.py      CSV loading and validation, chunking, incremental Chroma indexing, statistics
config.py      All settings (override with environment variables)
data/          Your review CSV files
chroma_langchain_db/   Generated vector index (safe to delete)
```

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com/) installed and running
- Enough memory for `qwen3-coder:30b` (about 18 GB on disk; plan for roughly 20 GB or more of RAM/VRAM). Use a smaller model if needed (see [Configuration](#configuration)).

## Setup

### 1. Download the models

```bash
ollama pull qwen3-coder:30b
ollama pull mxbai-embed-large
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install the packages

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run

```bash
python main.py
```

The first run embeds every review, which takes a short while. Later runs reuse the saved index and start quickly.

## Using the assistant

Example questions:

- What do customers say about the vegan options at Sakura?
- What are the main complaints at Spice Route?
- Which restaurant is best for working on a laptop?
- Are there gluten free choices? (asks across all restaurants)
- Which restaurant has the best average rating?
- Is it family friendly? (a follow-up that uses the current scope or conversation)

Commands inside the chat:

| Command | Purpose |
| --- | --- |
| `/restaurants` | List indexed restaurants |
| `/use <name>` or `/use all` | Restrict answers to one restaurant, or clear the restriction. Partial names work, e.g. `/use sakura` |
| `/stats` | Review counts and average ratings |
| `/sources on` / `/sources off` | Show or hide the retrieved reviews under each answer |
| `/reindex` | Rebuild the whole index from scratch |
| `/clear` | Forget the conversation history |
| `/help` | Show the command list |
| `q` | Quit |

Command-line options:

```bash
python main.py --restaurant "spice route"   # start scoped to one restaurant
python main.py --sources                    # always show retrieved reviews
python main.py --reindex                    # rebuild the index before starting
```

## Included datasets

| File | Restaurant | Reviews |
| --- | --- | --- |
| `data/bella_napoli_pizzeria.csv` | Bella Napoli Pizzeria | 10 |
| `data/sakura_ramen_house.csv` | Sakura Ramen House | 27 |
| `data/spice_route_indian_kitchen.csv` | Spice Route Indian Kitchen | 27 |
| `data/harbor_light_cafe.csv` | Harbor Light Cafe | 22 |

These are synthetic sample reviews written for this demo.

## Changing or adding datasets

1. Create a CSV in `data/` with these columns (names are case sensitive):

   | Column | Required | Description |
   | --- | --- | --- |
   | `Title` | yes | Short review headline |
   | `Review` | yes | Full review text |
   | `Rating` | yes | Number from 1 to 5 |
   | `Date` | yes | For example `2025-03-14` |
   | `Restaurant` | no | Restaurant name. If missing, the file name is used (`my_diner.csv` becomes "My Diner") |

   ```csv
   Restaurant,Title,Date,Rating,Review
   Corner Diner,Great pancakes,2025-06-01,5,"Fluffy pancakes and endless coffee."
   ```

2. Run `python main.py`. Only new or changed files are embedded.
3. To use only your own data, delete the sample CSVs from `data/` (their index entries are removed automatically on the next run).
4. To load data from elsewhere, set `RAG_DATA_DIR` (see below).

Tips: put one restaurant in each file, or use the `Restaurant` column to mix several in one file. Quote review text that contains commas. Keep the same restaurant spelling throughout, because scoping and statistics group by this exact name.

## Configuration

Set environment variables before running (all optional):

| Variable | Default | Meaning |
| --- | --- | --- |
| `RAG_LLM_MODEL` | `qwen3-coder:30b` | Answer-generation model |
| `RAG_EMBEDDING_MODEL` | `mxbai-embed-large` | Embedding model. Changing it triggers a full re-index |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `RAG_DATA_DIR` | `./data` | Folder containing the CSV files |
| `RAG_DB_DIR` | `./chroma_langchain_db` | Vector index location |
| `RAG_TEMPERATURE` | `0.2` | LLM temperature |
| `RAG_NUM_CTX` | `8192` | LLM context window in tokens |
| `RAG_TOP_K` | `6` | Reviews sent to the model per question |
| `RAG_FETCH_K` / `RAG_KEYWORD_K` | `20` / `8` | Candidate pool for vector (MMR) and keyword search |
| `RAG_MMR_LAMBDA` | `0.7` | 1 favours relevance, 0 favours diversity |

PowerShell example using a smaller model:

```powershell
$env:RAG_LLM_MODEL = "qwen3:8b"
python main.py
```

## How it works

1. **Load.** Pandas reads and validates every CSV in `data/`. Rows become LangChain `Document`s labelled with restaurant, rating, date and source file. Very long reviews are split into overlapping chunks.
2. **Index.** `mxbai-embed-large` embeds each review. ChromaDB stores vectors locally in `chroma_langchain_db/`. A manifest of file hashes decides what must be re-embedded.
3. **Understand the question.** With conversation history, the LLM rewrites the question into a standalone search query. Restaurant names and sentiment words are detected to filter the search.
4. **Retrieve.** Vector search (MMR) and BM25 each return candidates. Reciprocal Rank Fusion merges them into the top reviews.
5. **Generate.** `qwen3-coder:30b` receives the reviews, the computed statistics and the history, and streams an answer with citations.

## Troubleshooting

- **`Startup failed` or a connection error:** start Ollama (`ollama serve` or open the Ollama app) and check `ollama list`.
- **Model not found:** run the two `ollama pull` commands above.
- **Slow or out-of-memory answers:** set `RAG_LLM_MODEL` to a smaller model, or lower `RAG_NUM_CTX` and `RAG_TOP_K`.
- **CSV is missing columns:** check the header spelling and case: `Title`, `Review`, `Rating`, `Date`.
- **Answers seem to use old data:** run `/reindex`, or delete `chroma_langchain_db/`.
- **Wrong restaurant in an answer:** use `/use <name>` to scope the question.

## Limitations

- This is a RAG question-answering assistant, not a tool-using autonomous agent.
- Answer quality depends on the reviews. The assistant is instructed to say when evidence is thin, but small local models can still make mistakes, so use `--sources` to verify important answers.
- Average ratings and counts are exact. Other aggregate claims (for example "most people complained about X") rest on the retrieved sample of reviews only.
