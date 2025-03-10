# Technical Specification

## System Overview
The LongBench system is designed to facilitate the benchmarking and evaluation of various retrieval models and methodologies. It encompasses several scripts and components that handle the generation of scores, embeddings, and retrieval results using different algorithms and models. The system is structured to ensure efficient processing of large datasets through concurrent execution and chunk processing. The main components include shell scripts for orchestrating tasks, Python scripts for performing core computations, and BibTeX files for managing academic references.

## Core Functionality
The core functionality of the LongBench system revolves around the generation of scores and embeddings, and performing retrieval tasks using different models. The primary features and their implementations are as follows:

### BM25 Scoring
- **File**: `LongBench/retrieval/BM25/BM25.sh`
- **Functionality**: Generates BM25 scores for specified JSONL files.
- **Critical Logic**:
  - **File Filtering**: Filters JSONL files based on allowed names using a predefined list.
  - **Concurrent Execution**: Executes BM25 generation in groups to optimize resource usage.
- **Data Flow**:
  - Reads JSONL files from `source_dir`.
  - Writes BM25 scores to `dest_dir`.
- **Connection Points**:
  - `generate_BM25.py`: Python script that performs the actual BM25 scoring.
- **Complex Logic**:
  - **Chunk Processing**: Processes files in chunks defined by `chunk_size`.
  - **Concurrency**: Uses bash arrays and loops to manage concurrent executions.

### Contriever Model Embedding and Retrieval
- **File**: `LongBench/retrieval/contriever/mContriever.sh`
- **Functionality**: Generates passage embeddings and performs retrieval using the Contriever model.
- **Critical Logic**:
  - **Directory Traversal**: Traverses subdirectories to process each dataset.
  - **Concurrent Execution**: Executes embedding generation and retrieval in groups to optimize resource usage.
- **Data Flow**:
  - Reads TSV files from `split_dir`.
  - Generates embeddings in `embed_dir`.
  - Performs retrieval and writes results to `retrieved_dir`.
- **Connection Points**:
  - `LB2mC.py`: Python script to split data into chunks.
  - `generate_passage_embeddings.py`: Python script to generate embeddings.
  - `passage_retrieval.py`: Python script to perform retrieval.
  - `merge_output.py`: Python script to merge retrieval results.
- **Complex Logic**:
  - **Chunk Processing**: Processes files in chunks defined by `chunk_size`.
  - **Concurrency**: Uses bash arrays and loops to manage concurrent executions.
  - **Model Integration**: Integrates with the Contriever model for embedding and retrieval.

### OpenAI Embedding Generation
- **File**: `LongBench/retrieval/embedding/openai_embedding.sh`
- **Functionality**: Generates embeddings using the OpenAI model for specified JSONL files.
- **Critical Logic**:
  - **File Filtering**: Filters JSONL files based on allowed names.
  - **Concurrent Execution**: Executes embedding generation in groups to optimize resource usage.
- **Data Flow**:
  - Reads JSONL files from `source_dir`.
  - Generates embeddings and writes them to `dest_dir`.
- **Connection Points**:
  - `generate_openai_embedding.py`: Python script that performs the actual embedding generation.
- **Complex Logic**:
  - **Chunk Processing**: Processes files in chunks defined by `chunk_size`.
  - **Concurrency**: Uses bash arrays and loops to manage concurrent executions.

## Architecture
The LongBench system is structured to handle large-scale data processing efficiently. The architecture involves shell scripts that orchestrate the execution of Python scripts responsible for the core computations. Data flows from input directories containing raw data files (JSONL or TSV) to output directories where processed results (scores, embeddings, retrieval results) are stored. Concurrency and chunk processing are employed to manage resource usage and ensure scalability.

### Data Flow Patterns
1. **Input**: Raw data files (JSONL or TSV) are read from specified source directories.
2. **Processing**: 
   - Shell scripts filter and chunk the data.
   - Python scripts perform the core computations (BM25 scoring, embedding generation, retrieval).
3. **Output**: Processed results are written to designated output directories.
4. **Concurrency**: Concurrent execution is managed through bash arrays and loops to optimize resource usage.