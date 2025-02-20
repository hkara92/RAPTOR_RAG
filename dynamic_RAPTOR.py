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

    def generate_summaries(self, texts):
        # Use chat model to make a summary for each cluster text
        print("Generating summaries...")
        template = """You are an assistant to create a detailed summary of the text input provided.
Text:
{text}
"""
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.chat_model | StrOutputParser()

        summaries = {}
        for cluster, text in texts.items():
            token_count = self.num_tokens_from_string(text)

            if token_count > self.token_limit:
                # stop if text is too long
                raise ValueError(
                    f"Token limit exceeded for cluster {cluster} with {token_count} tokens. Unable to generate summary."
                )

            summary = chain.invoke({"text": text})
            summaries[cluster] = summary
        return summaries

    def run(self):
        # Main loop: load, embed, reduce, cluster, summarize until one cluster remains
        print("Running TextClusterSummarizer...")
        docs = self.load_and_split_documents()
        texts = [doc.page_content for doc in docs]
        all_summaries = texts

        iteration = 1
        # record initial state (no summaries)
        self.iteration_summaries.append(
            {"iteration": 0, "texts": texts, "summaries": []}
        )

        while True:
            print(f"Iteration {iteration}")
            embeddings = self.embed_texts(all_summaries)

            # Determine neighbors for UMAP
            n_neighbors = min(int((len(embeddings) - 1) ** 0.5), len(embeddings) - 1)
            if n_neighbors < 2:
                # if too few points, stop
                print("Not enough data points for UMAP reduction. Stopping iterations.")
                break

            embeddings_reduced = self.reduce_dimensions(
                embeddings, dim=2, n_neighbors=n_neighbors
            )
            labels, num_clusters = self.cluster_embeddings(
                embeddings_reduced, threshold=0.5
            )

            if num_clusters == 1:
                # one cluster left, done
                print("Reduced to a single cluster. Stopping iterations.")
                break

            # pick the first label each point belongs to
            simple_labels = [label[0] if len(label) > 0 else -1 for label in labels]
            df = pd.DataFrame(
                {
                    "Text": all_summaries,
                    "Embedding": list(embeddings_reduced),
                    "Cluster": simple_labels,
                }
            )

            clustered_texts = self.format_cluster_texts(df)
            summaries = self.generate_summaries(clustered_texts)

            # prepare for next iteration
            all_summaries = list(summaries.values())
            self.iteration_summaries.append(
                {
                    "iteration": iteration,
                    "texts": all_summaries,
                    "summaries": list(summaries.values()),
                }
            )
            iteration += 1

        # after loop, return the final summary and iteration data
        final_summary = all_summaries[0] if all_summaries else ""
        return {
            "initial_texts": texts,
            "iteration_summaries": self.iteration_summaries,
            "final_summary": final_summary,
        }

# Example of running the summarizer when this script is executed directly
if __name__ == "__main__":
    summarizer = TextClusterSummarizer(token_limit=200, data_directory="data")
    final_output = summarizer.run()
