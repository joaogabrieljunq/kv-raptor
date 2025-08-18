import os
import pandas as pd
from datasets import load_dataset
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter


class NarrativeQAPreprocessor:
    def __init__(self, dataset_name='deepmind/narrativeqa', sample_size=1572, repeat_count=1, sampled_filename="sampled_summaries.pkl"):
        self.dataset_name = dataset_name
        self.sample_size = sample_size
        self.repeat_count = repeat_count
        self.sampled_filename = sampled_filename

        self.df = None
        self.unique_summaries = None
        self.sampled_summaries = None
        self.repeated_summaries = None
        self.chunk_documents = None
        self.chunk_ids = None
        self.chunks = None

    def load_and_merge_dataset(self):
        print(f"Loading dataset: {self.dataset_name}")
        ds = load_dataset(self.dataset_name)
        dfs = [ds[split].to_pandas() for split in ['train', 'test', 'validation']]
        self.df = pd.concat(dfs, ignore_index=True)
        self.df['summary_text'] = self.df['document'].apply(lambda doc: doc['summary']['text'])
        self.unique_summaries = self.df.drop_duplicates(subset='summary_text').reset_index(drop=True)
        print(f"Total rows: {len(self.df)} | Unique summaries: {len(self.unique_summaries)}")

    def sample_summaries(self):
        if os.path.exists(self.sampled_filename):
            self.sampled_summaries = pd.read_pickle(self.sampled_filename)
            print(f"Loaded sampled summaries from {self.sampled_filename}")
        else:
            self.sampled_summaries = self.unique_summaries.sample(n=self.sample_size, random_state=42)
            if "question" not in self.sampled_summaries.columns:
                self.sampled_summaries["question"] = self.sampled_summaries["summary_text"].apply(lambda x: {"text": x})
            self.sampled_summaries.to_pickle(self.sampled_filename)
            print(f"Saved sampled summaries to {self.sampled_filename}")

        self.repeated_summaries = self.sampled_summaries.loc[
            self.sampled_summaries.index.repeat(self.repeat_count)
        ].reset_index(drop=True)
        print(f"Sampled rows: {len(self.sampled_summaries)} | Repeated rows: {len(self.repeated_summaries)}")

    def chunk_documents_from_summaries(self, chunk_size=100, chunk_overlap=0):
        documents = [Document(text=row['summary_text']) for _, row in self.unique_summaries.iterrows()]
        parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.chunks = parser.get_nodes_from_documents(documents)
        self.chunk_documents = [chunk.get_content() for chunk in self.chunks]
        self.chunk_ids = [f"chunk_{i}" for i in range(len(self.chunks))]
        print(f"Total documents: {len(documents)} | Chunks created: {len(self.chunks)}")

    def get_queries(self):
        return self.repeated_summaries

    def get_chunks(self):
        return self.chunk_documents, self.chunk_ids