# RAPTOR - Advanced Retrieval

## Overview

RAPTOR is a text clustering and summarization toolkit designed to help you analyze and condense large volumes of textual data. It leverages:

- **UMAP** for dimensionality reduction  
- **Gaussian Mixture Models** (GMM) for clustering  
- **OpenAI’s GPT models** for generating concise summaries of each cluster  
- **LangChain** for document loading, splitting, and embedding  

Through an iterative process, RAPTOR refines cluster assignments and produces a final summary that captures the core themes present in your data.

## Features

- **Flexible Document Loading**: Automatically ingest all `.txt` files from a specified directory.  
- **Chunked Text Splitting**: Split long documents into manageable chunks to respect token limits.  
- **Embedding Generation**: Create semantic embeddings using `OpenAIEmbeddings`.  
- **Dimensionality Reduction**: Use UMAP to project high-dimensional embeddings into a lower-dimensional space for clustering.  
- **Automatic Cluster Detection**: Determine the optimal number of clusters via BIC on Gaussian Mixture Models.  
- **Iterative Summarization**: Recursively cluster and summarize until a single coherent summary remains.  
- **Jupyter Notebook Demonstration**: Explore the pipeline interactively in `raptor_notebook.ipynb`.  

## Installation

1. **Clone the repository**  
    ```bash
    git clone https://github.com/yourusername/raptor-advanced-retrieval.git
    cd raptor-advanced-retrieval
    ```

2. **Create a virtual environment**  
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. **Install dependencies**  
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure environment variables**  
    - Copy `.env.example` or create a new `.env` file in the project root  
    - Add your OpenAI API key:  
      ```ini
      OPENAI_API_KEY=your_openai_api_key_here
      ```