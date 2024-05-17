from datasets import load_dataset

# from llama_index.core.storage.docstore import SimpleDocumentStore
from .mys.docstore import MySimpleDocumentStore
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from ragutils.utils import create_uuid_from_string
from llama_index.core.retrievers import BaseRetriever

# from llama_index.core.schema import TextNode, BaseNode

import os
import json
import pandas as pd
from typing import cast, List, Dict

from .globals import VERBOSE, USE_FIRST_N_DATA, logger
from .utils import create_uuid_from_string
from .mys.node import MyTextNode

CHUNK_OPTIONS = {
    "chunk_size": 2000,
    "chunk_overlap": 200,
}


class Data:
    def __init__(
        self, chunk_options: dict = CHUNK_OPTIONS, context_window: int = 1
    ) -> None:
        self.nodes = None
        self.pdfs = dict(
            pojk="data/SAL_POJK_SLIK.pdf",
            uu="data/2022uu27.pdf",
            # uu_draft="data/pdf/mandiri/AKSANAAN-UU-NOMOR-27-TAHUN-2022-TENTANG-PELINDUNGAN-DATA-PRIBADI.pdf",
        )
        self.chunk_options = chunk_options
        self.context_windows = context_window
        self._docstore = None

        self.resolve_paths()

    def resolve_paths(self):
        chunk_size = self.chunk_options["chunk_size"]
        chunk_overlap = self.chunk_options["chunk_overlap"]
        context_window = self.context_windows
        store = os.path.join(
            "./store", f"docstore_{chunk_size}_{chunk_overlap}_{context_window}.json"
        )
        self.paths = dict(
            data="./data",
            store=store,
        )

    def get_nodes(self):
        if os.path.exists(self.paths["store"]):
            self.nodes = self.load_nodes_from_docstore()
            # self.add_context_to_nodes()
        else:
            self.nodes = self.create_nodes_from_files()
            self.add_context_to_nodes()
            self.save_nodes_to_docstore()
        return self.nodes

    def load_nodes_from_docstore(self):
        docstore = MySimpleDocumentStore.from_persist_path(self.paths["store"])
        self._docstore = docstore

        nodes = cast(List[MyTextNode], list(docstore.docs.values()))
        logger.info(len(nodes))
        for node in nodes[:3]:
            logger.info(node)

        return nodes

    def create_nodes_from_files(self):
        pdfs = self.pdfs

        documents = SimpleDirectoryReader(
            "./data", input_files=pdfs.values()
        ).load_data()
        # len(documents), documents[:5]
        logger.info(len(documents))
        for ducument in documents[:3]:
            logger.info(ducument)

        node_parser = SentenceSplitter(**self.chunk_options)
        nodes = node_parser.get_nodes_from_documents(documents)
        # len(nodes), nodes[:5]
        logger.info(len(nodes))
        for node in nodes[:3]:
            logger.info(node)

        return nodes

    def save_nodes_to_docstore(self):
        if self.nodes is None:
            raise ValueError("Nodes is None")
        docstore = MySimpleDocumentStore()
        docstore.add_documents(self.nodes)
        docstore.persist(self.paths["store"])

        self._docstore = docstore

    def get_docstore(self):
        if self._docstore is None:
            self._docstore = MySimpleDocumentStore.from_persist_path(
                self.paths["store"]
            )
        return self._docstore

    def add_context_to_nodes(self):
        context_window = self.context_windows
        nodes = self.nodes
        new_nodes = [None] * len(nodes)  # Pre-allocate memory

        for idx, node in enumerate(nodes):
            start = max(0, idx - context_window)
            end = min(len(nodes), idx + context_window + 1)
            context_texts = (n.text for n in nodes[start:end])  # Use a generator
            context_ids = (n.id_ for n in nodes[start:end])  # Use a generator

            # 获取当前节点的所有属性，并创建新的 MyTextNode 实例
            node_attrs = node.__dict__.copy()  # 获取所有属性
            new_nodes[idx] = MyTextNode(
                context_texts=context_texts, context_ids=context_ids, **node_attrs
            )

        self.nodes = new_nodes


def load_data():
    dataset = load_dataset("rajpurkar/squad", trust_remote_code=True)

    logger.info(dataset)
    logger.info(dataset["train"][0])
    logger.info(len(dataset["train"]))

    useData = dataset["train"][:5000]

    docids = [
        str(create_uuid_from_string("context" + useData["id"][i]))
        for i in range(len(useData["id"]))
    ]
    texts = [useData["context"][i] for i in range(len(useData["context"]))]

    # Assign same text to same docid
    prev_text = ""
    prev_docid = ""
    for dicid, text, idx in zip(docids, texts, range(len(docids))):
        if text == prev_text:
            docids[idx] = prev_docid
        else:
            prev_text = text
            prev_docid = dicid

    assert len(docids) == len(texts)

    # print(docids[:10], texts[:10])
    logger.info(docids[:10])
    logger.info(texts[:10])
    # logger.info("\n".join(docids[:10]))
    # logger.info("\n".join(texts[:10]))

    text_chunks = dict(zip(docids, texts))

    # len(text_chunks), list(text_chunks.items())[:2]
    logger.info(len(text_chunks))
    logger.info(list(text_chunks.items())[:2])

    return text_chunks


def create_nodes(text_chunks: dict):
    # create nodes

    from llama_index.core.schema import TextNode

    nodes = []
    for k, v in text_chunks.items():
        node = TextNode(
            text=v,
            id_=k,
        )
        # src_doc_idx = doc_idxs[idx]
        # src_page = doc[src_doc_idx]
        nodes.append(node)

    # save docstore
    docstore = MySimpleDocumentStore()
    docstore.add_documents(nodes)

    # print(len(nodes), nodes[:5])
    logger.info(len(nodes))
    logger.info(nodes[:5])


# def build_qa_dataset(dataset, text_chunks: Dict):
#     query_ids = [str(create_uuid_from_string(item)) for item in dataset["id"]]
#     queries = dataset["question"]
#     queries_dict = dict(zip(query_ids, queries))
#     corpus_dict = text_chunks
#     positive_docids = [[item] for item in text_chunks.keys()]
#     relevant_docs_dict = dict(zip(query_ids, positive_docids))
#     # save the dataset as json
#     qa_dataset = {
#         "queries": queries_dict,
#         "corpus": corpus_dict,
#         "relevant_docs": relevant_docs_dict,
#     }
#     json.dump(qa_dataset, open("qa_squad_dataset_sub_uuid.json", "w"), indent=4)


class QADataSet:
    def __init__(self):
        self.csvs = dict(
            task1="./data/Task1-20_Q&A.csv",
            task2_p1="./data/Task2-80_Q&A(New)_p1.csv",
            task2_p2="./data/Task2-80_Q&A(New)_p2.csv",
        )
        self.tables = {key: pd.read_csv(value) for key, value in self.csvs.items()}
        # merge all tables
        self.table = pd.concat(self.tables.values(), ignore_index=True)

    def _get_column_from_table(self, column_name):
        return {
            str(create_uuid_from_string(value)): value
            for _, value in self.table[column_name].items()
        }

    def persist(self, filepath="./data/qa_dataset.json"):
        with open(filepath, "w") as json_file:
            json.dump(self.qa_dataset, json_file, indent=4)

    def from_persist_path(self, filepath="./data/qa_dataset.json"):
        with open(filepath, "r") as json_file:
            self.qa_dataset = json.load(json_file)

    @classmethod
    def from_nodes(cls, nodes, retriever: BaseRetriever, keep_num: int = 1):
        """
        build a dataset from a list of nodes using context info in MyNode, including context_texts and context_ids.
        there are QA pairs in tables, and we need to find the corresponding context for each QA pair.
        """
        instance = cls()

        ans_all = instance._get_column_from_table("Answer")
        qst_all = instance._get_column_from_table("Question")
        labels = []

        for _, ans_text in ans_all.items():
            retrieved_nodes = retriever.retrieve(ans_text)
            keep_nodes_ids = [node.node.node_id for node in retrieved_nodes[:keep_num]]
            labels.append(
                [
                    node.node_id
                    for node in nodes
                    if set(keep_nodes_ids).intersection(node.context_ids)
                ]
            )

        corpus = {node.node_id: node.text for node in nodes}
        qa_dataset = {
            "queries": qst_all,
            "corpus": corpus,
            "relevant_docs": dict(zip(qst_all.keys(), labels)),
        }

        instance.qa_dataset = qa_dataset

        return instance
