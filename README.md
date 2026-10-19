# OGD4All ![Python Version](https://img.shields.io/badge/Python-3.12-blue) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

OGD4All is a prototype system enabling an easy-to-use, transparent interaction with Geospatial Open Government Data through natural language. It was developed and tested with 430 tabular and geospatial datasets of the city of Zurich.

> [!NOTE]
> OGD4All was presented at IEEE CAI 2026. Check out our [paper](https://ieeexplore.ieee.org/abstract/document/11536530)!

## Architecture
<img width="2948" height="869" alt="highLevelOverview" src="https://github.com/user-attachments/assets/828f68c2-470b-4c59-b80f-8b371cac0d29" />


## ✨ Demo
https://github.com/user-attachments/assets/26c2588e-1a6d-4b68-a539-a84a9e6b98aa



<details>
<summary>

## Setup

</summary>

> If you only care about getting the benchmark, ignore the subsequent steps and instead load it directly via [this](https://huggingface.co/datasets/michael7ma/ogd4all-benchmark) HuggingFace dataset.

### 1. Code Environment Setup
Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run:
```bash
uv sync
```
This creates a `.venv` with all pinned dependencies automatically.

### 2. Obtaining Zurich OGD
The system was tested with 430 tabular and geospatial datasets of the city of Zurich, but could also be used with other types of data (with some additional steps).
While the associated metadata is already contained in this repository, you have two options for getting the actual data:
- Running the app with the `--lazy-download` flag, which only downloads needed datasets from [this HF dataset](https://huggingface.co/datasets/michael7ma/ogd4all-benchmark)
- Downloading all needed open data upfront from [this Google Drive](https://drive.google.com/file/d/1MdqwSZW0i__oRXNFUJAqlwYcFqypGpia/view?usp=sharing). Then, extract the ZIP folder into the directory `data\opendata\50000006\extracted` (make sure it is not further nested).

These datasets were manually exported. Note that this was performed in March and May of 2025, meaning some datasets may be out-of-date. All extraction timestamp, dataset titles and filenames are listed in `data\opendata\50000006\downloads.csv`, feel free to replace datasets with more recent versions.
If the data is available, you should see the following messages at startup:
```
[info     ] Will use 430 files for group owner 50000006.
[info     ] 430/548 metadata embeddings remaining after filtering for existing data.
```

### 3. Vector Store with Embeddings
By default, OGD4All uses OpenAI's `text-embedding-3-large` for semantic search. Corresponding embeddings were generated for the metadata of all datasets, and is stored in `data\metadata\50000006\processed_metadata_embeddings.csv`. You have two options:
- **Keep text-embedding-3-large**: You'll need to set either the environment variable `OPENAI_API_KEY` (with your OpenAI API key) or both `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT_EMBEDDING_LARGE` (with your Azure OpenAI API key and endpoint).
- **Use custom embeddings**: First, update the `processed_metadata_embeddings.csv` file by modifying the `scraping\generate_embeddings.py` script with your own method. Then, update the `get_embedding` function in `retrieval\retriever.py`.

### 4. LLM Setup
Support for LLMs provided by Azure OpenAI, Azure, OpenAI, OpenRouter, and Ollama is implemented.
However, you'll need to provide your API key by copying `example.env` and renaming it to `.env`, then filling the fields for your desired provider.
In the case of OpenAI, the key is enough; for Azure, you need to further set the variables for the model and embedding endpoints.

If your desired model/provider is not listed in `SUPPORTED_LLMS` in `utils.py`, just add it to the list and adapt the `get_llm_client` function.
</details>

  
## Usage
```
uv run python main.py
```
This will start a local gradio server where you can interact with 430 geospatial and tabular datasets of the city of Zurich through a simple interface.

### Command-Line Arguments
| Argument                | Default           | Choices                                                        | Description                                                                                                                                               |
| ----------------------- | ----------------- | -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--groupOwner`          | `50000006`        | –                                                              | The `groupOwner` ID whose metadata should be queried (e.g., the city of Zurich).                                                                          |
| `--top_n`               | `10`              | Any number > 0                                                 | The number of documents to retrieve for a single KNN search. Cost considerations and your model's context window should determine how high you push this. |
| `--retriever`           | `agentic`         | `agentic`, `knn`, `verified`                                   | The retrieval strategy to use.                                                                                                                            |
| `--analyzer`            | `iterative_local` | `simple_local_v2`, `simple`, `iterative_local`                 | The analyzer type to use.                                                                                                                                 |
| `--retrieval_llm`       | `gpt-4.1`         | `SUPPORTED_LLMS`                                               | The LLM to use for retrieval tasks.                                                                                                                       |
| `--retrieval_check_llm` | `gpt-4o`          | `SUPPORTED_LLMS`                                               | The LLM to use for checking whether a follow-up retrieval is needed. Should ideally be fast.                                                              |
| `--coding_llm`          | `gpt-4.1`         | `SUPPORTED_LLMS`                                               | The LLM to use for coding tasks/analysis.                                                                                                                 |
| `--hybrid_search`       | `Not set`         | –                                                              | Enable hybrid search with Milvus (default is semantic search).                                                                                            |
| `--bm25_search`         | `Not set`         | –                                                              | Enable BM25 search with Milvus (default is semantic search).                                                                                              |
| `--no_streaming`        | `Not set`         | –                                                              | Disable streaming for the coding LLM. This enables validation of LLM responses and token counting, but makes the system feel less responsive.             |

```
SUPPORTED_LLMS = ['gpt-5', 'gpt-5-mini', 'gpt-5-nano', 'gpt-4.1', 'gpt-4.1-mini', 'gpt-4o', 'gpt-o1', 'gpt-o3-mini-preview', 'gemini-2.5-pro', 'gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17', 'meta-llama-70B-instruct-latest', 'mistral-codestral', 'mistral-large', 'openai/gpt-oss-120b', 'meta-llama/llama-4-maverick']
```

## Citation
```
@INPROCEEDINGS{11536530,
  author={Siebenmann, Michael and Sánchez-Vaquerizo, Javier Argota and Arisona, Stefan and Samp, Krystian and Gisler, Luis and Helbing, Dirk},
  booktitle={2026 IEEE Conference on Artificial Intelligence (CAI)}, 
  title={OGD4All: A Framework for Accessible Interaction with Geospatial Open Government Data Based on Large Language Models}, 
  year={2026},
  volume={},
  number={},
  pages={882-888},
  keywords={Modeling;Large language models;Codes;Generative Pre-trained transformer;Manuals;Cognition;Cognitive systems;Tools;Artificial intelligence;Python},
  doi={10.1109/CAI68641.2026.11536530}}
```

## Acknowledgement
This prototype was implemented as part of Michael Siebenmann's Master's thesis at the Professorship of Computational Social Science and Esri R&D Center Zurich.
