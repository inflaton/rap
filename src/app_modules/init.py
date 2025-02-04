"""Main entrypoint for the app."""

import os
from timeit import default_timer as timer
from typing import List, Optional

from dotenv import find_dotenv, load_dotenv
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.vectorstores.faiss import FAISS

from app_modules.llm_loader import LLMLoader
from app_modules.utils import get_device_types, init_settings

found_dotenv = find_dotenv(".env")

if len(found_dotenv) == 0:
    found_dotenv = find_dotenv(".env.example")
print(f"loading env vars from: {found_dotenv}")
load_dotenv(found_dotenv, override=False)

# Constants
init_settings()

if os.environ.get("LANGCHAIN_DEBUG") == "true":
    from langchain.globals import set_debug

    set_debug(True)

from app_modules.llm_qa_chain import QAChain
from app_modules.llm_chat_chain import ChatChain
import nltk


def app_init():
    # https://github.com/huggingface/transformers/issues/17611
    os.environ["CURL_CA_BUNDLE"] = ""

    nltk.download("punkt")

    hf_embeddings_device_type, hf_pipeline_device_type = get_device_types()
    print(f"hf_embeddings_device_type: {hf_embeddings_device_type}")
    print(f"hf_pipeline_device_type: {hf_pipeline_device_type}")

    hf_embeddings_model_name = (
        os.environ.get("HF_EMBEDDINGS_MODEL_NAME") or "hkunlp/instructor-xl"
    )

    n_threds = int(os.environ.get("NUMBER_OF_CPU_CORES") or "4")
    index_path = os.environ.get("FAISS_INDEX_PATH") or os.environ.get(
        "CHROMADB_INDEX_PATH"
    )
    using_faiss = os.environ.get("FAISS_INDEX_PATH") is not None
    llm_model_type = os.environ.get("LLM_MODEL_TYPE")

    qa_with_rag = os.getenv("QA_WITH_RAG", "true").lower() == "true"
    print(f"qa_with_rag: {qa_with_rag}")

    retrieve_from_questions_file = os.getenv("RETRIEVER_TYPE") == "questions_file"
    print(f"retrieve_from_questions_file: {retrieve_from_questions_file}", flush=True)

    if qa_with_rag and not retrieve_from_questions_file:
        print(f"hf_embeddings_model_name: {hf_embeddings_model_name}")
        start = timer()
        embeddings = HuggingFaceInstructEmbeddings(
            model_name=hf_embeddings_model_name,
            model_kwargs={"device": hf_embeddings_device_type},
        )
        end = timer()

        print(f"Completed in {end - start:.3f}s")

    vectorstore = None
    if qa_with_rag and not retrieve_from_questions_file:
        start = timer()

        print(
            f"Load index from {index_path} with {'FAISS' if using_faiss else 'Chroma'}"
        )

        if not os.path.isdir(index_path):
            raise ValueError(f"{index_path} does not exist!")
        elif using_faiss:
            vectorstore = FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            )
        else:
            vectorstore = Chroma(
                embedding_function=embeddings, persist_directory=index_path
            )

        end = timer()

        print(f"Completed in {end - start:.3f}s")

    start = timer()
    llm_loader = LLMLoader(llm_model_type)
    llm_loader.init(n_threds=n_threds, hf_pipeline_device_type=hf_pipeline_device_type)
    if qa_with_rag:
        qa_chain = QAChain(vectorstore, llm_loader)
    else:
        qa_chain = ChatChain(llm_loader)
    end = timer()
    print(f"Completed in {end - start:.3f}s")

    return llm_loader, qa_chain
