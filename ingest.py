import os
import glob
from typing import List
from multiprocessing import Pool
from tqdm import tqdm
import logging
from dotenv import find_dotenv, load_dotenv
from datetime import datetime
import pickle

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings

from settings import SOURCE_PATH, DATABASE_PATH, embeddings, chunk_size, chunk_overlap

logging.basicConfig(level=logging.INFO)

# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}


def load_single_document(file_path: str):
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]

    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []):
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    all_files = []
    for ext in LOADER_MAPPING:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    filtered_files = [
        file_path for file_path in all_files if file_path not in ignored_files
    ]

    with Pool(processes=os.cpu_count()) as pool:
        documents = []
        with tqdm(
            total=len(filtered_files), desc="Loading new documents", ncols=80
        ) as pbar:
            for i, doc in enumerate(
                pool.imap_unordered(load_single_document, filtered_files)
            ):
                documents.append(doc)
                pbar.update()

    return documents


def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"], chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def get_vectorstore(document_chunks):
    db = FAISS.from_documents(document_chunks, embeddings)
    return db


def process_documents(ignored_files: List[str] = [], cache: bool = False):
    """
    Load documents and split in chunks
    """
    logging.info(f"Loading documents from {SOURCE_PATH}")
    documents = load_documents(SOURCE_PATH, ignored_files)
    if not documents:
        print("No new documents to load")
        exit(0)
    logging.info(f"Loaded {len(documents)} new documents from {SOURCE_PATH}")

    document_chunks = get_text_chunks(documents)
    logging.info(
        f"Split into {len(document_chunks)} chunks of text (max. {chunk_size} tokens each)"
    )

    logging.info(f"Creating vectorstore")
    db = get_vectorstore(document_chunks)

    logging.info(f"Saving database to {DATABASE_PATH}")
    save_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cache:
        pickle.dump(db, open(DATABASE_PATH + f"/db_{save_time}.pkl", "wb"))
    return db


if __name__ == "__main__":
    db = process_documents(cache=True)
