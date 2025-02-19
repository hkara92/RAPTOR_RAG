import os  # for loading environment variables and file paths

import numpy as np  # numerical operations on arrays
import pandas as pd  # data handling and analysis
import umap  # for reducing dimensions of vectors
import tiktoken  # for counting tokens in text

from dotenv import find_dotenv, load_dotenv  # loading .env files
from sklearn.mixture import GaussianMixture  # clustering algorithm
from langchain.prompts import ChatPromptTemplate  # building chat prompts
from langchain_core.output_parsers import StrOutputParser  # parsing chat output
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  # embedding and chat models
from langchain_text_splitters import RecursiveCharacterTextSplitter  # splitting long text\ nfrom langchain_community.document_loaders import DirectoryLoader  # loading text files

# Load environment variables from a .env file (like API keys)
load_dotenv(find_dotenv())

class TextClusterSummarizer:
    # This class loads documents, splits them into chunks, embeds, clusters, and summarizes
    def __init__(
        self,
        token_limit,
        data_directory,
        glob_pattern="**/*.txt",
    ):
        print("Initializing TextClusterSummarizer...")
        self.token_limit = token_limit  # max tokens allowed for summary
        # loader will search for files matching glob in data_directory
        self.loader = DirectoryLoader(data_directory, glob=glob_pattern)
        # text_splitter breaks long texts into chunks of ~200 chars with 20 chars overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )
        # model to get text embeddings
        self.embedding_model = OpenAIEmbeddings()
        # chat model to generate summaries
        self.chat_model = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
        # keep track of summaries at each step
        self.iteration_summaries = []

    def load_and_split_documents(self):
        # Load all docs and split into smaller chunks
        print("Loading and splitting documents...")
        docs = self.loader.load()
        return self.text_splitter.split_documents(docs)

    def embed_texts(self, texts):
        # Turn each text chunk into a vector representation
        print("Embedding texts...")
        return [self.embedding_model.embed_query(txt) for txt in texts]

    def reduce_dimensions(self, embeddings, dim, n_neighbors=None, metric="cosine"):
        # Reduce vector size to 2 or more dimensions for clustering
        print(f"Reducing dimensions to {dim}...")
        if n_neighbors is None:
            # default neighbors roughly sqrt(n)
            n_neighbors = int((len(embeddings) - 1) ** 0.5)
        return umap.UMAP(
            n_neighbors=n_neighbors, n_components=dim, metric=metric
        ).fit_transform(embeddings)

    def num_tokens_from_string(self, string: str) -> int:
        # Count how many tokens are in a string, for API limits
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def cluster_embeddings(self, embeddings, threshold, random_state=0):
        # Group embeddings into clusters using Gaussian Mixture
        print("Clustering embeddings...")
        n_clusters = self.get_optimal_clusters(embeddings)
        gm = GaussianMixture(n_components=n_clusters, random_state=random_state).fit(
            embeddings
        )
        probs = gm.predict_proba(embeddings)
        # assign points to clusters if probability > threshold
        return [np.where(prob > threshold)[0] for prob in probs], n_clusters

    def get_optimal_clusters(self, embeddings, max_clusters=50, random_state=1234):
        # Find best number of clusters by minimizing BIC
        print("Calculating optimal number of clusters...")
        max_clusters = min(max_clusters, len(embeddings))
        bics = [
            GaussianMixture(n_components=n, random_state=random_state)
            .fit(embeddings)
            .bic(embeddings)
            for n in range(1, max_clusters)
        ]
        optimal = np.argmin(bics) + 1
        print(f"Optimal number of clusters: {optimal}")
        return optimal

    def format_cluster_texts(self, df):
        # Combine texts in each cluster into one long string
        print("Formatting cluster texts...")
        clustered_texts = {}
        for cluster in df["Cluster"].unique():
            cluster_texts = df[df["Cluster"] == cluster]["Text"].tolist()
            # join texts with a separator
            clustered_texts[cluster] = " --- ".join(cluster_texts)
        return clustered_texts
