import os
import sys
import torch
from dotenv import find_dotenv, load_dotenv

found_dotenv = find_dotenv(".env")

if len(found_dotenv) == 0:
    found_dotenv = find_dotenv(".env.example")
print(f"loading env vars from: {found_dotenv}")
load_dotenv(found_dotenv, override=False)

path = os.path.dirname(found_dotenv)
print(f"Adding {path} to sys.path")
sys.path.append(path)

from llm_toolkit.llm_utils import *
from llm_toolkit.translation_utils import *

device = check_gpu()
is_cuda = torch.cuda.is_available()

model_name = os.getenv("MODEL_NAME")
adapter_name_or_path = os.getenv("ADAPTER_NAME_OR_PATH")
load_in_4bit = os.getenv("LOAD_IN_4BIT") == "true"
data_path = os.getenv("DATA_PATH")
results_path = os.getenv("RESULTS_PATH")
batch_size = int(os.getenv("BATCH_SIZE", 1))
use_english_datasets = os.getenv("USE_ENGLISH_DATASETS") == "true"
using_chat_template = os.getenv("USING_CHAT_TEMPLATE") == "true"
max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", 2048))
start_repetition_penalty = float(os.getenv("START_REPETITION_PENALTY", 1.0))
end_repetition_penalty = float(os.getenv("END_REPETITION_PENALTY", 1.3))

print(
    model_name,
    adapter_name_or_path,
    load_in_4bit,
    data_path,
    results_path,
    use_english_datasets,
    max_new_tokens,
    batch_size,
)

if is_cuda:
    torch.cuda.empty_cache()
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"(0) GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    torch.cuda.empty_cache()

model, tokenizer = load_model(
    model_name, load_in_4bit=load_in_4bit, adapter_name_or_path=adapter_name_or_path
)

if is_cuda:
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"(2) GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

datasets = load_translation_dataset(
    data_path, tokenizer, using_chat_template=using_chat_template
)

if len(sys.argv) > 1:
    num = int(sys.argv[1])
    if num > 0:
        print(f"--- evaluating {num} entries")
        datasets["test"] = datasets["test"].select(range(num))

print_row_details(datasets["test"].to_pandas(), indices=[0, -1])


def on_repetition_penalty_step_completed(model_name, predictions):
    save_results(
        model_name,
        results_path,
        datasets["test"],
        predictions,
    )

    metrics = calc_metrics(
        datasets["test"]["english"],
        predictions,
        datasets["test"]["chinese"],
        debug=True,
    )
    print(f"{model_name} metrics: {metrics}")


if adapter_name_or_path is not None:
    model_name += "/" + adapter_name_or_path.split("/")[-1]

evaluate_model_with_repetition_penalty(
    model,
    tokenizer,
    model_name,
    datasets["test"],
    on_repetition_penalty_step_completed,
    start_repetition_penalty=start_repetition_penalty,
    end_repetition_penalty=end_repetition_penalty,
    step_repetition_penalty=0.02,
    batch_size=batch_size,
    max_new_tokens=max_new_tokens,
    device=device,
)

if is_cuda:
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"(3) GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
