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
max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", 2048))
start_num_shots = int(os.getenv("START_NUM_SHOTS", 0))

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


def on_num_shots_step_completed(model_name, dataset, predictions):
    save_results(
        model_name,
        results_path,
        dataset,
        predictions,
    )

    metrics = calc_metrics(dataset["english"], predictions, debug=True)
    print(f"{model_name} metrics: {metrics}")


if adapter_name_or_path is not None:
    model_name += "/" + adapter_name_or_path.split("/")[-1]


def evaluate_model_with_num_shots(
    model,
    tokenizer,
    model_name,
    data_path,
    start_num_shots=0,
    range_num_shots=[0, 1, 3, 5, 10, 50],
    batch_size=1,
    max_new_tokens=2048,
    device="cuda",
):
    print(f"Evaluating model: {model_name} on {device}")

    for num_shots in range_num_shots:
        if num_shots < start_num_shots:
            continue

        print(f"*** Evaluating with num_shots: {num_shots}")

        datasets = load_translation_dataset(data_path, tokenizer, num_shots=num_shots)
        print_row_details(datasets["test"].to_pandas())

        predictions = eval_model(
            model,
            tokenizer,
            datasets["test"],
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )

        model_name_with_rp = f"{model_name}/shots-{num_shots:02d}"

        try:
            on_num_shots_step_completed(
                model_name_with_rp,
                datasets["test"],
                predictions,
            )
        except Exception as e:
            print(e)


evaluate_model_with_num_shots(
    model,
    tokenizer,
    model_name,
    data_path,
    batch_size=batch_size,
    max_new_tokens=max_new_tokens,
    device=device,
    start_num_shots=start_num_shots,
)

if is_cuda:
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"(3) GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
