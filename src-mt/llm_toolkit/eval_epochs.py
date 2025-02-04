import os
import sys
import subprocess
from dotenv import find_dotenv, load_dotenv

found_dotenv = find_dotenv(".env")

if len(found_dotenv) == 0:
    found_dotenv = find_dotenv(".env.example")
print(f"loading env vars from: {found_dotenv}")
load_dotenv(found_dotenv, override=False)

workding_dir = os.path.dirname(found_dotenv)
os.chdir(workding_dir)
sys.path.append(workding_dir)
print("workding dir:", workding_dir)
print(f"adding {workding_dir} to sys.path")
sys.path.append(workding_dir)

from llm_toolkit.llm_utils import *
from llm_toolkit.translation_utils import *


def evaluate_model_all_epochs(
    model,
    tokenizer,
    model_name,
    adapter_path_base,
    dataset,
    results_path,
    start_epoch=0,
    end_epoch=-1,
    batch_size=1,
    max_new_tokens=300,
    checkpoints_per_epoch=1,
    device="cuda",
):
    if adapter_path_base is None:
        num_train_epochs = 0
        print(f"No adapter path provided. Running with base model:{model_name}")
    else:
        # find subdirectories in adapter_path_base
        # and sort them by epoch number
        subdirs = [
            d
            for d in os.listdir(adapter_path_base)
            if os.path.isdir(os.path.join(adapter_path_base, d))
        ]

        subdirs = sorted(subdirs, key=lambda x: int(x.split("-")[-1]))
        num_train_epochs = len(subdirs) // checkpoints_per_epoch
        if checkpoints_per_epoch > 1:
            subdirs = subdirs[checkpoints_per_epoch - 1 :: checkpoints_per_epoch]
        print(f"found {num_train_epochs} checkpoints: {subdirs}")

        if end_epoch < 0 or end_epoch > num_train_epochs:
            end_epoch = num_train_epochs

        print(f"Running from epoch {start_epoch} to {end_epoch}")

    for i in range(start_epoch, end_epoch + 1):
        print(f"Epoch {i}")
        if i > 0:
            adapter_name = subdirs[i - 1]
            adapter_path = adapter_path_base + "/" + adapter_name
            print(f"loading adapter: {adapter_path}")
            model.load_adapter(adapter_path, adapter_name=adapter_name)
            model.active_adapters = adapter_name

        predictions = eval_model(
            model,
            tokenizer,
            dataset,
            device=device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )

        model_name_with_epochs = f"{model_name}/epochs-{i:02d}"

        save_results(
            model_name_with_epochs,
            results_path,
            dataset,
            predictions,
        )

        metrics = calc_metrics(dataset["english"], predictions, debug=True)
        print(f"{model_name_with_epochs} metrics: {metrics}")


if __name__ == "__main__":
    model_name = os.getenv("MODEL_NAME")
    adapter_path_base = os.getenv("ADAPTER_PATH_BASE")
    checkpoints_per_epoch = int(os.getenv("CHECKPOINTS_PER_EPOCH", 1))
    start_epoch = int(os.getenv("START_EPOCH", 1))
    end_epoch = os.getenv("END_EPOCH", -1)
    load_in_4bit = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"
    results_path = os.getenv("RESULTS_PATH", None)
    data_path = os.getenv("DATA_PATH")

    print(
        model_name,
        adapter_path_base,
        load_in_4bit,
        start_epoch,
        results_path,
    )

    device = check_gpu()
    is_cuda = torch.cuda.is_available()

    print(f"Evaluating model: {model_name} on {device}")

    if is_cuda:
        torch.cuda.empty_cache()
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"(0) GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

    model, tokenizer = load_model(model_name, load_in_4bit=load_in_4bit)

    datasets = load_translation_dataset(data_path, tokenizer, num_shots=0)
    print_row_details(datasets["test"].to_pandas())

    if is_cuda:
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"(1) GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

    evaluate_model_all_epochs(
        model,
        tokenizer,
        model_name,
        adapter_path_base,
        datasets["test"],
        results_path,
        checkpoints_per_epoch=checkpoints_per_epoch,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        device=device,
    )

    if is_cuda:
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"(3) GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")
