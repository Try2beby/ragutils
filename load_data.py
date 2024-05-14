from datasets import load_dataset
from llama_index.core.storage.docstore import SimpleDocumentStore
import json
from typing import cast, List, Dict
from llama_index.core.schema import BaseNode

from .globals import VERBOSE, USE_FIRST_N_DATA, logger
from .utils import create_uuid_from_string


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
    docstore = SimpleDocumentStore()
    docstore.add_documents(nodes)

    # print(len(nodes), nodes[:5])
    logger.info(len(nodes))
    logger.info(nodes[:5])


def load_nodes_from_docstore():
    docstore = SimpleDocumentStore.from_persist_path("./store/docstore.json")
    nodes = cast(List[BaseNode], list(docstore.docs.values()))
    logger.info(len(nodes))
    for node in nodes[:5]:
        logger.info(node)

    return nodes


def build_qa_dataset(dataset, text_chunks: Dict):
    query_ids = [str(create_uuid_from_string(item)) for item in dataset["id"]]
    queries = dataset["question"]
    queries_dict = dict(zip(query_ids, queries))
    corpus_dict = text_chunks
    positive_docids = [[item] for item in text_chunks.keys()]
    relevant_docs_dict = dict(zip(query_ids, positive_docids))
    # save the dataset as json
    qa_dataset = {
        "queries": queries_dict,
        "corpus": corpus_dict,
        "relevant_docs": relevant_docs_dict,
    }
    json.dump(qa_dataset, open("qa_squad_dataset_sub_uuid.json", "w"), indent=4)
