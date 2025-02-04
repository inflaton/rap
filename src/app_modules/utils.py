# -*- coding:utf-8 -*-
from __future__ import annotations

import json
import logging
import os
import platform
import re
from pathlib import Path
import evaluate
import pandas as pd
import requests
import torch
from tqdm import tqdm


class LogRecord(logging.LogRecord):
    def getMessage(self):
        msg = self.msg
        if self.args:
            if isinstance(self.args, dict):
                msg = msg.format(**self.args)
            else:
                msg = msg.format(*self.args)
        return msg


class Logger(logging.Logger):
    def makeRecord(
        self,
        name,
        level,
        fn,
        lno,
        msg,
        args,
        exc_info,
        func=None,
        extra=None,
        sinfo=None,
    ):
        rv = LogRecord(name, level, fn, lno, msg, args, exc_info, func, sinfo)
        if extra is not None:
            for key in extra:
                rv.__dict__[key] = extra[key]
        return rv


def init_settings():
    logging.setLoggerClass(Logger)
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    )


def remove_extra_spaces(text):
    return re.sub(" +", " ", text.strip())


def print_llm_response(llm_response, debug_retrieval=True):
    answer = llm_response["answer"] if "answer" in llm_response else None
    if answer is None:
        answer = llm_response["response"] if "response" in llm_response else None

    if answer is not None:
        print("\n\n***Answer:")
        print(answer)

    source_documents = (
        llm_response["source_documents"] if "source_documents" in llm_response else None
    )
    if source_documents is None:
        source_documents = (
            llm_response["sourceDocs"] if "sourceDocs" in llm_response else None
        )

    if debug_retrieval and source_documents is not None:
        print("\nSources:")
        for index, source in enumerate(source_documents):
            metadata = source["metadata"] if "metadata" in source else source.metadata
            if "page" in metadata:
                print(f" Page:  {metadata['page']}", end="")

            print(
                f" Source {index + 1}: "
                + str(metadata["url"] if "url" in metadata else metadata["source"])
            )
            print(
                source["page_content"]
                if "page_content" in source
                else source.page_content
            )

    if "chat_history" in llm_response:
        print("\nChat History:")
        print(llm_response["chat_history"])


def get_device_types():
    print("Running on: ", platform.platform())
    print("MPS is", "NOT" if not torch.backends.mps.is_available() else "", "available")
    print("CUDA is", "NOT" if not torch.cuda.is_available() else "", "available")
    device_type_available = "cpu"

    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print(
                "MPS not available because the current PyTorch install was not "
                "built with MPS enabled."
            )
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )
    else:
        device_type_available = "mps"

    if torch.cuda.is_available():
        print("CUDA is available, we have found ", torch.cuda.device_count(), " GPU(s)")
        print(torch.cuda.get_device_name(0))
        print("CUDA version: " + torch.version.cuda)
        device_type_available = f"cuda:{torch.cuda.current_device()}"

    return (
        os.environ.get("HF_EMBEDDINGS_DEVICE_TYPE") or device_type_available,
        os.environ.get("HF_PIPELINE_DEVICE_TYPE") or device_type_available,
    )


def ensure_model_is_downloaded(llm_model_type):
    if llm_model_type.startswith("gpt4all"):
        local_path = (
            os.environ.get("GPT4ALL_J_MODEL_PATH")
            if llm_model_type == "gpt4all-j"
            else os.environ.get("GPT4ALL_MODEL_PATH")
        )
        url = (
            os.environ.get("GPT4ALL_J_DOWNLOAD_LINK")
            if llm_model_type == "gpt4all-j"
            else os.environ.get("GPT4ALL_DOWNLOAD_LINK")
        )
    elif llm_model_type == "llamacpp":
        local_path = os.environ.get("LLAMACPP_MODEL_PATH")
        url = os.environ.get("LLAMACPP_DOWNLOAD_LINK")
    elif llm_model_type == "ctransformers":
        local_path = os.environ.get("CTRANSFORMERS_MODEL_PATH")
        url = os.environ.get("CTRANSFORMERS_DOWNLOAD_LINK")
    else:
        raise ValueError(f"wrong model typle: {llm_model_type}")

    path = Path(local_path)

    if path.is_file():
        print(f"model: {local_path} exists")
    else:
        print(f"downloading model: {local_path} from {url} ...")
        path.parent.mkdir(parents=True, exist_ok=True)

        # send a GET request to the URL to download the file. Stream since it's large
        response = requests.get(url, stream=True)

        # open the file in binary mode and write the contents of the response to it in chunks
        # This is a large file, so be prepared to wait.
        with open(local_path, "wb") as f:
            for chunk in tqdm(response.iter_content(chunk_size=8192)):
                if chunk:
                    f.write(chunk)

    return local_path


bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bert_score = evaluate.load("bertscore")


def calc_perf_scores(predictions, references, debug=False):
    if debug:
        print("predictions:", predictions)
        print("references:", references)

    bleu_scores = bleu.compute(
        predictions=predictions, references=references, max_order=1
    )
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bert_scores = bert_score.compute(
        predictions=predictions,
        references=references,
        lang="en",
        model_type="microsoft/deberta-large-mnli",
    )
    result = {
        "bleu_scores": bleu_scores,
        "rouge_scores": rouge_scores,
        "bert_scores": bert_scores,
    }

    if debug:
        print("result:", result)

    return result


def calc_bleu_rouge_scores(predictions, references, debug=False):
    if debug:
        print("predictions:", predictions)
        print("references:", references)

    bleu_scores = bleu.compute(
        predictions=predictions, references=references, max_order=1
    )
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    result = {"bleu_scores": bleu_scores, "rouge_scores": rouge_scores}

    if debug:
        print("result:", result)

    return result


def calc_metrics(df):
    predictions = [df["answer"][i] for i in range(len(df))]
    references = [df["ground_truth"][i] for i in range(len(df))]

    return calc_bleu_rouge_scores(predictions, references)


pattern_abnormal_newlines = re.compile(r"\n{5,}")
pattern_text_repetitions = re.compile(r"\b(\w.+?)\b(\1+)", re.M | re.DOTALL)
exception_pattern = re.compile(r"(\w+\.)\1")


# final version for repetition detection
def detect_repetitions(
    text, debug=False, pattern_text_repetitions=pattern_text_repetitions
):
    subtotals = [0, 0]

    if isinstance(text, str):
        patterns = [pattern_abnormal_newlines, pattern_text_repetitions]
        for i, pattern in enumerate(patterns):
            if debug:
                print(
                    f"----detect {'abnormal newlines' if i == 0 else 'text repetitions'}----"
                )
            matches = pattern.finditer(text)
            for match in matches:
                if debug:
                    print(match)
                    for groupNum in range(0, len(match.groups())):
                        groupNum = groupNum + 1
                        print(
                            "Group {groupNum} found at {start}-{end}: `{group}`".format(
                                groupNum=groupNum,
                                start=match.start(groupNum),
                                end=match.end(groupNum),
                                group=match.group(groupNum),
                            )
                        )

                if exception_pattern.match(match[0]):
                    if debug:
                        print("ignored: ", match[0])
                    continue

                start, end = match.span()
                subtotals[i] += end - start

    result = (subtotals[0], subtotals[1], subtotals[0] + subtotals[1])

    if debug:
        print(result)
    return result


def detect_abnormal_newlines(text, debug=False):
    return detect_repetitions(text, debug=debug)[0]


def detect_text_repetitions(text, debug=False):
    return detect_repetitions(text, debug=debug)[1]


def detect_repetition_scores(text, debug=False):
    newline_score, repetition_score, total_repetitions = detect_repetitions(
        text, debug=debug
    )
    return pd.Series([newline_score, repetition_score, total_repetitions])
