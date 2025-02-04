import os
import sys
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

model_name = os.getenv("MODEL_NAME")
data_path = os.getenv("DATA_PATH")
results_path = os.getenv("RESULTS_PATH")
max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", 2048))

print(
    model_name,
    data_path,
    results_path,
    max_new_tokens,
)


def on_num_shots_step_completed(model_name, dataset, predictions, results_path):
    save_results(
        model_name,
        results_path,
        dataset,
        predictions,
    )

    metrics = calc_metrics(dataset["english"], predictions, debug=True)
    print(f"{model_name} metrics: {metrics}")


def evaluate_model_with_num_shots(
    model_name,
    data_path,
    results_path=None,
    range_num_shots=[0, 1, 3, 5, 10, 50],
    max_new_tokens=2048,
    result_column_name=None,
):
    print(f"Evaluating model: {model_name}")

    datasets = load_translation_dataset(data_path)
    print_row_details(datasets["test"].to_pandas())

    for num_shots in range_num_shots:
        print(f"*** Evaluating with num_shots: {num_shots}")

        predictions = eval_openai(num_shots, datasets, max_new_tokens=max_new_tokens)
        model_name_with_shorts = (
            result_column_name
            if result_column_name
            else f"{model_name}/shots-{num_shots:02d}"
        )

        try:
            on_num_shots_step_completed(
                model_name_with_shorts, datasets["test"], predictions, results_path
            )
        except Exception as e:
            print(e)


if __name__ == "__main__":
    evaluate_model_with_num_shots(
        model_name,
        data_path,
        results_path=results_path,
        max_new_tokens=max_new_tokens,
    )
