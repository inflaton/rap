import os
import re
import glob
import pandas as pd
import evaluate
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm
from eval_modules.calc_repetitions import *
from llm_toolkit.llm_utils import load_tokenizer, print_row_details

print(f"loading {__file__}")

bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")
accuracy = evaluate.load("accuracy")
sacrebleu = evaluate.load("sacrebleu")
comet = evaluate.load("comet")


def extract_answer(text, debug=False):
    if text and isinstance(text, str):
        # Remove the begin and end tokens
        text = re.sub(
            r".*?(assistant|\[/INST\]).+?\b", "", text, flags=re.DOTALL | re.MULTILINE
        )
        if debug:
            print("--------\nstep 1:", text)

        text = re.sub(r"<.+?>.*", "", text, flags=re.DOTALL | re.MULTILINE)
        if debug:
            print("--------\nstep 2:", text)

        text = re.sub(
            r".*?end_header_id\|>\n\n", "", text, flags=re.DOTALL | re.MULTILINE
        )
        if debug:
            print("--------\nstep 3:", text)

    return text


def calc_metrics(references, predictions, sources=None, debug=False):
    assert len(references) == len(
        predictions
    ), f"lengths are difference: {len(references)} != {len(predictions)}"

    predictions = [extract_answer(text) for text in predictions]
    results = {}

    results["comet"] = comet.compute(
        predictions=predictions, references=references, sources=sources
    )["mean_score"]

    results["meteor"] = meteor.compute(predictions=predictions, references=references)[
        "meteor"
    ]

    results["sacrebleu"] = sacrebleu.compute(
        predictions=predictions, references=references
    )

    results["bleu_scores"] = bleu.compute(
        predictions=predictions, references=references, max_order=4
    )
    results["rouge_scores"] = rouge.compute(
        predictions=predictions, references=references
    )

    correct = [1 if ref == pred else 0 for ref, pred in zip(references, predictions)]
    accuracy = sum(correct) / len(references)

    results["accuracy"] = accuracy
    if debug:
        correct_ids = [i for i, c in enumerate(correct) if c == 1]
        results["correct_ids"] = correct_ids

    return results


def save_results(model_name, results_path, dataset, predictions, debug=False):
    if not os.path.exists(results_path):
        # Get the directory part of the file path
        dir_path = os.path.dirname(results_path)

        # Create all directories in the path (if they don't exist)
        os.makedirs(dir_path, exist_ok=True)
        df = dataset.to_pandas()
        df.drop(columns=["text", "prompt"], inplace=True, errors="ignore")
    else:
        df = pd.read_csv(results_path, on_bad_lines="warn")

    df[model_name] = predictions

    if debug:
        print(df.head(1))

    df.to_csv(results_path, index=False)


system_prompt = "You are a helpful assistant that translates Chinese to English."


def get_few_shot_prompt(dataset, num_shots=5):
    translation_prompt = "You will be given a Chinese sentence to translate. If it is an incomplete sentence, or if you are unsure about the meaning, simply copy the input text as your output. Do not output any additional sentence such as explanation or reasoning.\n\n"
    if num_shots > 0:
        example_translations = "Example Translations:\n"
        for i in range(num_shots):
            example_translations += f"Chinese: {dataset[i]['chinese']}\n"
            example_translations += f"English: {dataset[i]['english']}\n"
        translation_prompt = translation_prompt + example_translations + "\n"

    translation_prompt = translation_prompt + "Chinese: {input}\nEnglish:"
    return translation_prompt


def load_translation_dataset(
    data_path, tokenizer=None, num_shots=0, for_openai=False, using_chat_template=True
):
    train_data_file = data_path.replace(".tsv", "-train.tsv")
    test_data_file = data_path.replace(".tsv", "-test.tsv")

    if not os.path.exists(train_data_file):
        print("generating train/test data files")
        dataset = load_dataset(
            "csv", data_files=data_path, delimiter="\t", split="train"
        )
        print(len(dataset))
        dataset = dataset.filter(lambda x: x["chinese"] and x["english"])

        datasets = dataset.train_test_split(test_size=0.2)
        print(len(dataset))

        # Convert to pandas DataFrame
        train_df = pd.DataFrame(datasets["train"])
        test_df = pd.DataFrame(datasets["test"])

        # Save to TSV
        train_df.to_csv(train_data_file, sep="\t", index=False)
        test_df.to_csv(test_data_file, sep="\t", index=False)

    print("loading train/test data files")
    datasets = load_dataset(
        "csv",
        data_files={"train": train_data_file, "test": test_data_file},
        delimiter="\t",
    )

    if tokenizer or for_openai:
        translation_prompt = get_few_shot_prompt(datasets["train"], num_shots)

        def formatting_prompts_func(examples):
            inputs = examples["chinese"]
            outputs = examples["english"]

            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                None,
            ]

            model_name = os.getenv("MODEL_NAME")

            # if "mistral" in model_name.lower():
            # messages = messages[1:]

            texts = []
            prompts = []
            for input, output in zip(inputs, outputs):
                prompt = translation_prompt.format(input=input)
                messages[-1] = {"role": "user", "content": prompt}

                if for_openai:
                    prompts.append(messages.copy())
                    text = messages.copy()
                    text.append(
                        {
                            "role": "assistant",
                            "content": output,
                        }
                    )
                    texts.append(text)
                else:
                    prompt = (
                        tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        if using_chat_template
                        else prompt
                    )

                    prompts.append(prompt)
                    texts.append(prompt + output + tokenizer.eos_token)

            return {"text": texts, "prompt": prompts}

        datasets = datasets.map(
            formatting_prompts_func,
            batched=True,
        )

    print(datasets)
    return datasets


def count_entries_with_max_tokens(entries, max_tokens):
    """
    Count the number of entries with the max output tokens or more.

    Parameters:
    entries (list of int): List of token counts for each entry.
    max_tokens (int): The maximum token threshold.

    Returns:
    int: The number of entries with token counts greater than or equal to max_tokens.
    """
    count = 0
    for tokens in entries:
        if tokens >= max_tokens:
            count += 1
    return count


def detect_repetition_scores(row, col, debug=False):
    text = row[col] if isinstance(row[col], str) else ""

    newline_score, repetition_score, total_repetitions = detect_scores(
        row, debug=debug, answer_col=col, ground_truth_col="english"
    )

    return pd.Series(
        [
            newline_score if newline_score > 0 else 0,
            repetition_score if repetition_score > 0 else 0,
            total_repetitions if total_repetitions > 0 else 0,
            len(text),
        ]
    )


def count_chinese_characters(text):
    if isinstance(text, str) is False:
        return 0

    # Define a regular expression pattern for Chinese characters
    chinese_char_pattern = r"[\u4e00-\u9fff]"

    # Use re.findall to find all Chinese characters in the text
    chinese_chars = re.findall(chinese_char_pattern, text)

    # Return the count of Chinese characters
    return len(chinese_chars)


def get_metrics(df, max_output_tokens=2048, variant="rpp", existing_metrics_df=None):
    metrics_df = pd.DataFrame(df.columns.T)[2:]
    metrics_df.rename(columns={0: "model"}, inplace=True)
    metrics_df[variant] = metrics_df["model"].apply(
        lambda x: x.split(f"{variant}-")[-1]
    )
    metrics_df["model"] = metrics_df["model"].apply(
        lambda x: x.split(f"/{variant}-")[0].split("/checkpoint")[0]
    )

    metrics_df.reset_index(inplace=True)
    metrics_df = metrics_df.drop(columns=["index"])

    models = [
        model
        for model in metrics_df["model"].unique()
        if ("/" in model or "gpt" in model)
        and "ground_truth_" not in model
        and "count_" not in model
        and "output_" not in model
    ]
    print(models)

    tokenizers = {model: load_tokenizer(model) for model in models}

    comet = []
    meteor = []
    spbleu = []
    bleu_1 = []
    rouge_l = []
    ews_score = []
    repetition_score = []
    total_repetitions = []
    rr = []
    num_max_output_tokens = []
    translation_completeness = []
    percentage_of_repeated_entries = []
    columns = df.columns[2:]

    new_col = f"count_chinese_characters-ground_truth"
    df[new_col] = df["chinese"].apply(count_chinese_characters)

    for col in columns:
        metrics = None
        if existing_metrics_df is not None:
            parts = col.split(f"/{variant}-")
            if len(parts) == 1:
                break
            print(parts)
            val = float(parts[1]) if variant == "rpp" else int(parts[1])
            result = existing_metrics_df[
                existing_metrics_df["model"] == parts[0].split("/checkpoint")[0]
            ]

            for i, row in result.iterrows():
                # print(i, row[variant], val)
                if row[variant] == val:
                    print(f"Using existing metrics for {col}")
                    metrics = row.to_dict()
                    # print(metrics)
                    break

        if metrics is None:
            print(f"Calculating metrics for {col}")
            metrics = calc_metrics(
                df["english"], df[col], sources=df["chinese"], debug=True
            )
        print(f"{col}: {metrics}")

        comet.append(metrics["comet"])
        meteor.append(metrics["meteor"])
        spbleu.append(
            metrics["spbleu"] if "spbleu" in metrics else metrics["sacrebleu"]["score"]
        )
        bleu_1.append(
            metrics["bleu_1"] if "bleu_1" in metrics else metrics["bleu_scores"]["bleu"]
        )
        rouge_l.append(
            metrics["rouge_l"]
            if "rouge_l" in metrics
            else metrics["rouge_scores"]["rougeL"]
        )

        df[["ews_score", "repetition_score", "total_repetitions", "answer_len"]] = df.apply(
            lambda x: detect_repetition_scores(x, col), axis=1
        )
        ews_score.append(df["ews_score"].mean())
        repetition_score.append(df["repetition_score"].mean())
        total_repetitions.append(df["total_repetitions"].mean())

        rr.append(df["total_repetitions"].mean() / df["answer_len"].mean())

        r, t = df[df["total_repetitions"] > 0].shape[0], df.shape[0]
        percentage_of_repeated_entries.append(100 * r / t)

        model = col.split(f"/{variant}")[0].split("/checkpoint")[0]

        new_col = f"ground_truth_tokens-{model}"
        df[new_col] = df["english"].apply(
            lambda x: len(tokenizers[model](x)["input_ids"])
        )

        new_col = f"count_chinese_characters-{col}"
        df[new_col] = df[col].apply(
            lambda x: 1 if count_chinese_characters(x) > 0 else 0
        )
        translation_completeness.append(1 - df[new_col].sum() / len(df))

        new_col = f"output_tokens-{col}"
        df[new_col] = df[col].apply(
            lambda x: (
                len(tokenizers[model](x)["input_ids"]) if isinstance(x, str) else 0
            )
        )

        num_max_output_tokens.append(
            count_entries_with_max_tokens(df[new_col], max_output_tokens)
        )
        
    metrics_df["comet"] = comet
    metrics_df["meteor"] = meteor
    metrics_df["spbleu"] = spbleu
    metrics_df["bleu_1"] = bleu_1
    metrics_df["rouge_l"] = rouge_l
    metrics_df["ews_score"] = ews_score
    metrics_df["newline_score"] = ews_score
    metrics_df["repetition_score"] = repetition_score
    metrics_df["total_repetitions"] = total_repetitions
    metrics_df["rr"] = rr
    metrics_df["rrp"] = metrics_df["rr"].apply(
        lambda x: x * 100
    )
    metrics_df["perf"] = metrics_df["comet"]
    metrics_df["rap"] = metrics_df.apply(
        lambda x: calc_adjusted_performance(x["comet"], x["rr"]), axis=1
    )

    metrics_df["translation_completeness"] = translation_completeness
    metrics_df["num_max_output_tokens"] = num_max_output_tokens
    metrics_df["percentage_of_repeated_entries"] = percentage_of_repeated_entries

    if variant != "rpp":
        metrics_df[variant] = metrics_df[variant].astype(int)

    return metrics_df


def analyze_translation_results(df, col, max_new_tokens=300, repetition_threshold=100):
    df[["ews_score", "repetition_score", "total_repetitions", "answer_len"]] = df.apply(
        lambda x: detect_repetition_scores(x, col), axis=1
    )
    rows = df.query(f"total_repetitions > {repetition_threshold}")
    print(
        f"*** Found {len(rows)} rows with total_repetitions > {repetition_threshold} for {col}"
    )

    for i in range(len(rows)):
        row = rows.iloc[i]
        print(row["chinese"])
        print("=" * 80)
        print(row["english"])
        print("=" * 80)
        output = row[col]
        print(output)
        print("=" * 80)
        detect_repetitions(output, debug=True)

    output_tokens = f"output_tokens-{col}"
    df2 = df[df[output_tokens] >= max_new_tokens][
        ["chinese", "english", col, output_tokens]
    ]

    print(
        f"\n*** Found {len(df2)} rows with output_tokens >= {max_new_tokens} for {col}"
    )
    print_row_details(df2, range(len(df2)))

    count_chinese_characters = f"count_chinese_characters-{col}"
    df3 = df[df[count_chinese_characters] > 0][
        ["chinese", "english", col, count_chinese_characters]
    ]

    print(f"\n*** Found {len(df3)} rows with incomplete translations for {col}")
    print_row_details(df3, range(len(df3)))


def plot_metrics(metrics_df, figsize=(14, 5), ylim=(0, 0.44)):
    plt.figure(figsize=figsize)
    df_melted = pd.melt(
        metrics_df, id_vars="model", value_vars=["meteor", "bleu_1", "rouge_l"]
    )

    barplot = sns.barplot(x="variable", y="value", hue="model", data=df_melted)

    # Set different hatches for each model
    hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*", "//", "\\\\"]

    # Create a dictionary to map models to hatches
    model_hatches = {
        model: hatches[i % len(hatches)]
        for i, model in enumerate(metrics_df["model"].unique())
    }

    # Apply hatches based on the model
    num_vars = len(df_melted["variable"].unique())
    for i, bar in enumerate(barplot.patches):
        model = df_melted["model"].iloc[i // num_vars]
        bar.set_hatch(model_hatches[model])

    # Manually update legend to match the bar hatches
    handles, labels = barplot.get_legend_handles_labels()
    for handle, model in zip(handles, metrics_df["model"].unique()):
        handle.set_hatch(model_hatches[model])

    barplot.set_xticklabels(["METEOR", "BLEU-1", "ROUGE-L"])
    for p in barplot.patches:
        if p.get_height() == 0:
            continue
        barplot.annotate(
            f"{p.get_height():.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            xytext=(0, 10),
            textcoords="offset points",
        )

    barplot.set(ylim=ylim, ylabel="Scores", xlabel="Metrics")
    plt.legend(bbox_to_anchor=(0.5, -0.1), loc="upper center")
    plt.show()


def plot_times(perf_df, ylim=0.421):
    # Adjusted code to put "train-time" bars in red at the bottom

    fig, ax1 = plt.subplots(figsize=(12, 10))

    color_train = "tab:red"
    color_eval = "orange"
    ax1.set_xlabel("Models")
    ax1.set_ylabel("Time (mins)")
    ax1.set_xticks(range(len(perf_df["model"])))  # Set x-ticks positions
    ax1.set_xticklabels(perf_df["model"], rotation=90)

    # Plot "train-time" first so it's at the bottom
    ax1.bar(
        perf_df["model"],
        perf_df["train-time(mins)"],
        color=color_train,
        label="train-time",
    )

    # Then, plot "eval-time" on top of "train-time"
    ax1.bar(
        perf_df["model"],
        perf_df["eval-time(mins)"],
        bottom=perf_df["train-time(mins)"],
        color=color_eval,
        label="eval-time",
    )

    ax1.tick_params(axis="y")
    ax1.legend(loc="upper left")

    if "meteor" in perf_df.columns:
        ax2 = ax1.twinx()
        color_meteor = "tab:blue"
        ax2.set_ylabel("METEOR", color=color_meteor)
        ax2.plot(
            perf_df["model"],
            perf_df["meteor"],
            color=color_meteor,
            marker="o",
            label="meteor",
        )
        ax2.tick_params(axis="y", labelcolor=color_meteor)
        ax2.legend(loc="upper right")
        ax2.set_ylim(ax2.get_ylim()[0], ylim)

    # Show numbers in bars
    for p in ax1.patches:
        height = p.get_height()
        if height == 0:  # Skip bars with height 0
            continue
        ax1.annotate(
            f"{height:.2f}",
            (p.get_x() + p.get_width() / 2.0, p.get_y() + height),
            ha="center",
            va="center",
            xytext=(0, -10),
            textcoords="offset points",
        )

    fig.tight_layout()
    plt.show()


def translate_via_openai(
    text, translation_prompt, max_tokens=None, model="gpt-4o-mini", base_url=None
):
    llm = ChatOpenAI(
        model=model,
        temperature=0,
        max_tokens=max_tokens,
        timeout=None,
        max_retries=2,
        base_url=base_url,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates Chinese to English.",
            ),
            (
                "human",
                translation_prompt,
            ),
        ]
    )

    chain = prompt | llm
    response = chain.invoke(
        {
            "input": text,
        }
    )

    return response.content


def eval_openai(num_shots, datasets, model="gpt-4o-mini", max_new_tokens=300):
    translation_prompt = get_few_shot_prompt(datasets["train"], num_shots=num_shots)
    eval_dataset = datasets["test"]
    total = len(eval_dataset)
    predictions = []

    for i in tqdm(range(total)):
        output = translate_via_openai(
            eval_dataset["chinese"][i],
            translation_prompt,
            model=model,
            max_tokens=max_new_tokens,
        )
        predictions.append(output)

    return predictions


def convert_time_to_seconds(time_str):
    # print(f"converting time_str: {time_str}")
    # Split the time string into its components
    time_parts = list(map(int, time_str.split(":")))

    # Initialize total minutes
    total_seconds = 0

    # Calculate total minutes based on the number of parts
    if len(time_parts) == 3:  # HH:MM:SS
        hours, minutes, seconds = time_parts
        total_seconds = hours * 3600 + minutes * 60 + seconds
    elif len(time_parts) == 2:  # MM:SS
        minutes, seconds = time_parts
        total_seconds = minutes * 60 + seconds
    elif len(time_parts) == 1:  # SS
        seconds = time_parts[0]
        total_seconds = seconds

    return total_seconds


def process_log_file(log_file, total_entries, variant):
    time_pattern = re.compile(r"\[(.{5,10})<00:00")
    metrics_pattern = re.compile(rf"(.*)/{variant}-(.*) metrics:")

    model = []
    shots = []
    eval_time = []

    i = 0

    with open(log_file, "r") as f:
        try:
            for line in f:
                i += 1
                matches = time_pattern.search(line)
                if matches:
                    time_pattern_matches = matches
                else:
                    matches = metrics_pattern.search(line)
                    if matches:
                        metrics_pattern_matches = matches
                        groups = metrics_pattern_matches.groups()

                        model.append(groups[0].split("/checkpoint")[0])
                        shots.append(groups[1])

                        groups = time_pattern_matches.groups()
                        time_str = groups[0]
                        eval_time.append(
                            convert_time_to_seconds(time_str) / total_entries
                        )
        except Exception as e:
            print(f"Error processing log file: {log_file} at line {i}: {line}")
            print(e)

    df = pd.DataFrame(
        {
            "model": model,
            variant: shots,
            "eval_time": eval_time,
        }
    )
    return df


def load_eval_times(logs_folder, total_entries=1133, variant="shots"):
    # Get a list of all files in the logs folder
    log_files = glob.glob(os.path.join(logs_folder, "*"))
    log_files.sort()

    time_df = pd.DataFrame({"model": [], variant: [], "eval_time": []})

    for log_file in log_files:
        print(f"Loading content of {log_file}")
        df = process_log_file(log_file, total_entries, variant)
        time_df = pd.concat([time_df, df], ignore_index=True)

    time_df[variant] = time_df[variant].apply(
        lambda x: x if variant == "rpp" else int(x)
    )
    # Keep the last occurrence of each duplicate
    return time_df.drop_duplicates(subset=["model", variant], keep="last")


def load_alpaca_data(data_path):
    alpaca_data_path = "data/alpaca_mac.json"

    if os.path.exists(alpaca_data_path):
        print("loading existing data from:", alpaca_data_path)
        data = pd.read_json(alpaca_data_path, orient="records", lines=False)
        return data

    datasets = load_translation_dataset(data_path)
    prompt_template = get_few_shot_prompt(datasets["train"], num_shots=0)

    df_train = datasets["train"].to_pandas()
    df_train["instruction"] = df_train.apply(
        lambda x: prompt_template.format(input=x["chinese"]), axis=1
    )

    df_alpaca = pd.DataFrame(
        {
            "system": [system_prompt] * len(df_train),
            "instruction": df_train["instruction"].to_list(),
            "input": [""] * len(df_train),
            "output": df_train["english"].to_list(),
        }
    )

    df_alpaca.to_json(alpaca_data_path, orient="records", lines=False, indent=2)

    return df_alpaca


def load_openai_training_data(
    data_path, openai_data_path="datasets/mac/openai-training.jsonl"
):
    if os.path.exists(openai_data_path):
        print("loading existing data from:", openai_data_path)
        data = pd.read_json(openai_data_path, orient="records", lines=True)
        return data

    datasets = load_translation_dataset(data_path)
    prompt_template = get_few_shot_prompt(datasets["train"], num_shots=0)

    df_train = datasets["train"].to_pandas()
    messages = []

    for i, row in df_train.iterrows():
        messages.append(
            [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt_template.format(input=row["chinese"]),
                },
                {
                    "role": "assistant",
                    "content": row["english"],
                },
            ]
        )

    df_openai = pd.DataFrame(
        {
            "messages": messages,
        }
    )
    df_openai.to_json(openai_data_path, orient="records", lines=True)
    return df_openai
