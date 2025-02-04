import os
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import nltk
import evaluate
import traceback

bert_score = evaluate.load("bertscore")
meteor = evaluate.load("meteor")

print(f"loading: {__file__}")

# pattern_non_word_char_repetition = re.compile(r"\s{5,}")
# pattern_text_repetitions = re.compile(r"(.{5}.*)\s*((\1)\s*)+", re.M | re.DOTALL)

# final version
pattern_non_word_char_repetition = re.compile(r"[\s\W]{5,}")
pattern_text_repetitions = re.compile(
    r"(?P<repeat>.{5}.*?)(?:[\s\W]*(?P=repeat))+", re.M | re.DOTALL | re.IGNORECASE
)
# Explanation of the Regex Pattern:
#   (?P<repeat>.{5}.*?): Captures any sequence of characters with minimal length of 5 and names this group repeat.
#     .*?: Matches zero or more characters, non-greedily (as few as possible).
#   (?:[\s\W]+(?P=repeat))+: A non-capturing group that matches one or more repetitions of:
#     [\s\W]+: One or more whitespace or non-word characters (spaces, punctuation, etc.).
#     (?P=repeat): A backreference to the named group repeat.


def del_non_word_char_repetition(text, debug=False):
    count = 0

    if isinstance(text, str):
        if debug:
            print("----detect non-word characters repetition----")
        count = len(text)
        text = pattern_non_word_char_repetition.sub("\t", text)
        count -= len(text)
        if debug and count:
            print(f"removed non-word characters repetition: {count}")
    return text, count


# final version for repetition detection
def detect_text_repetitions(text, debug=False):
    count = 0

    if isinstance(text, str):
        if debug:
            print("----detect text repetitions----")
        matches = pattern_text_repetitions.finditer(text)
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

            start, end = match.span()
            count += end - start - len(match.group(1))

    return count


def detect_repetitions(text, debug=False):
    if isinstance(text, str) is False:
        return 0, 0, 0
    text, count_non_word_char_repetition = del_non_word_char_repetition(
        text, debug=debug
    )
    count_text_repetitions = detect_text_repetitions(text, debug=debug)
    total_repetitions = count_non_word_char_repetition + count_text_repetitions

    result = (count_non_word_char_repetition, count_text_repetitions, total_repetitions)

    if debug:
        print(result)
    return result


def detect_scores(
    row, debug=False, answer_col="answer", ground_truth_col="ground_truth"
):
    newline_score, repetition_score, total_repetitions = detect_repetitions(
        row[answer_col], debug=debug
    )

    if ground_truth_col:
        ground_truth_newline_score, ground_truth_repetition_score, _ = (
            detect_repetitions(row[ground_truth_col], debug=debug)
        )

        newline_score -= ground_truth_newline_score
        if newline_score < 0:
            newline_score = 0

        repetition_score -= ground_truth_repetition_score
        if repetition_score < 0:
            repetition_score = 0

        total_repetitions = newline_score + repetition_score

    return pd.Series([newline_score, repetition_score, total_repetitions])


def load_with_newline_and_repetition_scores(result_file, force_recalculate=False):
    print(f"loading result file: {result_file}")
    df = pd.read_csv(result_file, comment="#", on_bad_lines="warn")

    if (
        force_recalculate
        or "newline_score" not in df.columns
        or "repetition_score" not in df.columns
        or "total_repetitions" not in df.columns
        or "nrr" not in df.columns
        or "rr" not in df.columns
    ):
        if (
            force_recalculate
            or "newline_score" not in df.columns
            or "repetition_score" not in df.columns
            or "total_repetitions" not in df.columns
        ):
            df[["newline_score", "repetition_score", "total_repetitions"]] = df.apply(
                detect_scores, axis=1
            )

        df["answer_len"] = df["answer"].apply(
            lambda x: len(x) if isinstance(x, str) else 0
        )

        df["nrr"] = df.apply(
            lambda x: (
                1
                if x["answer_len"] == 0
                else 1 - (x["newline_score"] + x["repetition_score"]) / x["answer_len"]
            ),
            axis=1,
        )

        df["rr"] = df["nrr"].apply(lambda x: 1 - x)

        df.to_csv(result_file, index=False)

    return df


def replace_last(source_string, old_string, new_string):
    head, _sep, tail = source_string.rpartition(old_string)
    return head + new_string + tail


def load_for_repetition_penalty(
    csv_result_file, repetition_penalty, force_recalculate=False
):
    result_file = replace_last(
        csv_result_file, ".csv", f"_RP_{repetition_penalty:.3f}.csv"
    )
    return load_with_newline_and_repetition_scores(
        result_file, force_recalculate=force_recalculate
    )


rap_penalty_functions = {
    "linear": lambda x: x,
    "quadratic": lambda x: x * x,
    "cubic": lambda x: x * x * x,
    "logarithmic": lambda x: math.log(x + 1, 2),
    "exponential": lambda x: math.exp(x - 1),
}


def calc_adjusted_performance(f, r, l=1, penalty_function="cubic"):
    n = 1 - r / l if l > 0 else 0
    return f * rap_penalty_functions[penalty_function](n)


def calculate_adjusted_performance(row):
    r = row["total_repetitions"]
    l = row["answer_len"]
    adjusted_precision = calc_adjusted_performance(row["precision"], r, l)
    adjusted_recall = calc_adjusted_performance(row["recall"], r, l)
    return pd.Series([adjusted_precision, adjusted_recall])


def load_performance_df(csv_result_file, repetition_penalty):
    result_file = replace_last(
        csv_result_file, ".csv", f"_RP_{repetition_penalty:.3f}-t2_evaluated.json"
    )
    result_file = result_file.replace("/results/", "/eval/")
    print(f"loading json file: {result_file}")
    df = pd.read_json(result_file)

    return df


def calculate_performance_score(
    csv_result_file, repetition_penalty, force_recalculate=False
):
    result_file = replace_last(
        csv_result_file, ".csv", f"_rpp_{repetition_penalty:.2f}.csv"
    )

    if os.path.exists(result_file):
        print(f"loading result file: {result_file}")
        df = load_with_newline_and_repetition_scores(
            result_file, force_recalculate=force_recalculate
        )
    else:
        print(f"re-creating result file: {result_file}")
        df = pd.DataFrame()
        force_recalculate = True

    if force_recalculate or "f2" in df.columns or "f1" not in df.columns:
        try:
            perf_df = load_performance_df(csv_result_file, repetition_penalty)
            df.drop(
                columns=[
                    "precision",
                    "recall",
                    "f1",
                    "f2",
                    "entities_in_answer",
                    "entities_in_question",
                    "word_count",
                ],
                errors="ignore",
                inplace=True,
            )

            df["id"] = perf_df["id"]
            df["question"] = perf_df["question"]
            df["answer"] = perf_df["pred_answer"]
            df["word_count"] = df["answer"].apply(
                lambda x: len(nltk.word_tokenize(x)) if isinstance(x, str) else 0
            )
            df["ground_truth"] = perf_df["ground_truth"]

            df["eval_gemini_1.0_pro"] = perf_df["eval_gemini_1.0_pro"]
            df["precision"] = perf_df["score"].apply(lambda x: x[0])
            df["recall"] = perf_df["score"].apply(lambda x: x[1])
            df["f1"] = perf_df["score"].apply(lambda x: x[2])
        except Exception as e:
            print(f"\tignored error: {e}")
            # traceback.print_exc()

        df[["newline_score", "repetition_score", "total_repetitions"]] = df.apply(
            detect_scores, axis=1
        )
        df["answer_len"] = df["answer"].apply(
            lambda x: len(x) if isinstance(x, str) else 0
        )

        df[["adjusted_precision", "adjusted_recall"]] = df.apply(
            calculate_adjusted_performance, axis=1
        )

        df.to_csv(result_file, index=False)
        print(f"performance scores saved to result file: {result_file}")

    # print(f"df len: {len(df)}")

    return df


def adjust_perf_scores_with_repetition_penalty(result, precision, recall):
    newline_score = [
        df["newline_score"].mean() for df in result["df_list_repetition_penalty"]
    ]

    repetition_score = [
        df["repetition_score"].mean() for df in result["df_list_repetition_penalty"]
    ]

    answer_len = [
        df["answer_len"].mean() for df in result["df_list_repetition_penalty"]
    ]

    precision = [
        calc_adjusted_performance(f, n + r, l)
        for f, n, r, l in zip(precision, newline_score, repetition_score, answer_len)
    ]
    recall = [
        calc_adjusted_performance(f, n + r, l)
        for f, n, r, l in zip(recall, newline_score, repetition_score, answer_len)
    ]

    return precision, recall


def plot_performance_scores(
    result,
    models=None,
    title="Performance",
):
    if models is None:
        models = result.keys()
    for model in models:
        print(f"model: {model}")
        df = result[model]["df_overall"]

        # Calculate the statistics
        precision = [
            df["precision"].mean() for df in result[model]["df_list_repetition_penalty"]
        ]
        recall = [
            df["recall"].mean() for df in result[model]["df_list_repetition_penalty"]
        ]
        f1 = [2 * (p * r) / (p + r) for p, r in zip(precision, recall)]
        best_f1 = max(f1)
        best_f1_index = f1.index(best_f1)

        precision, recall = adjust_perf_scores_with_repetition_penalty(
            result[model], precision, recall
        )
        afrp = [2 * (p * r) / (p + r) for p, r in zip(precision, recall)]

        # f1 = [df["f1"].mean() for df in result[model]["df_list_repetition_penalty"]]
        best_afrp = max(afrp)
        best_afrp_index = afrp.index(best_afrp)

        adjusted_precision = [
            df["adjusted_precision"].mean()
            for df in result[model]["df_list_repetition_penalty"]
        ]
        adjusted_recall = [
            df["adjusted_recall"].mean()
            for df in result[model]["df_list_repetition_penalty"]
        ]
        afrp2 = [
            2 * (p * r) / (p + r) for p, r in zip(adjusted_precision, adjusted_recall)
        ]
        best_afrp2 = max(afrp2)
        best_afrp2_index = afrp2.index(best_afrp2)

        repetition_penalties = list(df["repetition_penalty"])

        # line plot for precision, recall, f1
        plt.figure(figsize=(10, 6))

        plt.axvspan(
            repetition_penalties[best_f1_index] - 0.01,
            repetition_penalties[best_f1_index] + 0.01,
            alpha=0.5,
            edgecolor="none",
            facecolor="blue",
        )

        # plt.axvspan(
        #     repetition_penalties[best_afrp2_index] - 0.01,
        #     repetition_penalties[best_afrp2_index] + 0.01,
        #     alpha=0.5,
        #     edgecolor="none",
        #     facecolor="green",
        # )

        plt.axvspan(
            repetition_penalties[best_afrp_index] - 0.01,
            repetition_penalties[best_afrp_index] + 0.01,
            alpha=0.5,
            edgecolor="none",
            facecolor="orange",
        )

        plt.plot(repetition_penalties, f1, label="F1", marker="D", color="blue")
        # plt.plot(
        #     repetition_penalties,
        #     afrp2,
        #     label="Per-question RAP - F1",
        #     marker="s",
        #     color="green",
        # )
        plt.plot(
            repetition_penalties,
            afrp,
            label="RAP - F1",
            marker="o",
            color="orange",
        )
        plt.xlabel("Repetition Penalties")
        plt.ylabel("Score")
        # plt.xlim(0.99, 1.31)
        # y in percentage
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.title(f"{model} {title}")
        plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")

        plt.show()


def plot_best_afrp(
    result,
    models=None,
    title="Models with Best RAP - F1",
    ref_result=None,
):
    # Initialize lists to store the statistics
    model_names = []
    best_f1 = []
    best_afrp = []
    best_repetition_penalty = []
    best_mtr = []

    if models is None:
        models = result.keys()
    for model in models:
        print(f"model: {model}")
        df = result[model]["df_overall"]

        # Calculate the statistics
        precision = [
            df["precision"].mean() for df in result[model]["df_list_repetition_penalty"]
        ]
        recall = [
            df["recall"].mean() for df in result[model]["df_list_repetition_penalty"]
        ]
        # f1 = [df["f1"].mean() for df in result[model]["df_list_repetition_penalty"]]
        f1 = [2 * (p * r) / (p + r) for p, r in zip(precision, recall)]

        newline_score = [
            df["newline_score"].mean()
            for df in result[model]["df_list_repetition_penalty"]
        ]
        # print(f"newline_score: {newline_score}")

        repetition_score = [
            df["repetition_score"].mean()
            for df in result[model]["df_list_repetition_penalty"]
        ]
        # print(f"repetition_score: {repetition_score}")

        answer_len = [
            df["answer_len"].mean()
            for df in result[model]["df_list_repetition_penalty"]
        ]

        afrp = [
            calc_adjusted_performance(f, n + r, l)
            for f, n, r, l in zip(f1, newline_score, repetition_score, answer_len)
        ]

        best_afrp.append(max(afrp))
        best_afrp_index = afrp.index(best_afrp[-1])
        best_repetition_penalty.append(df["repetition_penalty"][best_afrp_index])

        best_f1.append(f1[best_afrp_index])
        best_mtr.append(
            newline_score[best_afrp_index] + repetition_score[best_afrp_index]
        )

        # print(
        #     f"best repetition penalty: {best_repetition_penalty[-1]}, best afrp: {best_afrp[-1]}, f1: {best_f1[-1]}"
        # )

        df = result[model]["df_list_repetition_penalty"][best_afrp_index]

        model_names.append(
            f"{model} (RP={best_repetition_penalty[-1]})"
        )  # Add the model name to the list

    if ref_result is not None:
        print("ref_result:", ref_result)
        for model in ref_result.keys():
            model_names.append(model)
            df = pd.read_csv(ref_result[model])
            # df = df[df["id"].isin(wikidata_df["id"])]

            p = df["precision"].mean()
            r = df["recall"].mean()

            f1 = 2 * p * r / (p + r) if p + r > 0 else 0
            best_f1.append(f1)
            best_afrp.append(f1)
            best_mtr.append(0)

    print("model_names:", model_names)
    # print("best_f1:", best_f1)
    # print("best_afrp:", best_afrp)

    # Create a DataFrame with the statistics
    data = pd.DataFrame(
        {
            "Model": model_names,
            "RAP - F1": best_afrp,
            "F1": best_f1,
        }
    )

    # Melt the DataFrame to a long format
    data_melted = data.melt(id_vars="Model", var_name="Metric", value_name="Score")

    # Pivot the DataFrame to a wide format
    data_pivoted = data_melted.pivot(index="Metric", columns="Model", values="Score")

    # make sure the columns are following the order of the models
    data_pivoted = data_pivoted[model_names]

    # make sure three groups in the order of precision, recall, f1
    data_pivoted = data_pivoted.reindex(["RAP - F1", "F1"])

    # Plot the statistics
    plt.figure(figsize=(15, 6))
    ax = data_pivoted.plot(kind="bar", ax=plt.gca(), width=0.9)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")

    # Set the rotation of the x-axis labels to 0 degrees
    plt.xticks(rotation=0)

    # Format the y-axis to display as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # get the max value of the y-axis
    a1 = max(best_afrp)
    a2 = max(best_f1)

    max_value = max([a1, a2]) * 1.12
    print("max_value:", max_value)

    # Set the y-axis limit up to 70%
    ax.set_ylim(0, max_value)

    # Add the values above each bar
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height() * 100:.1f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 10),
            textcoords="offset points",
            rotation=90,
        )

    plt.show()
    return data_pivoted, best_mtr


def plot_best_performance(
    result,
    models=None,
    title="Models with Best F1 Score",
    adjusted_f1=False,
    ref_result=None,
):
    # Initialize lists to store the statistics
    model_names = []
    best_precision = []
    best_recall = []
    best_f1 = []
    best_repetition_penalty = []
    best_mtr = []

    if models is None:
        models = result.keys()
    for model in models:
        print(f"model: {model}")
        df = result[model]["df_overall"]

        # Calculate the statistics
        precision = [
            df["precision"].mean() for df in result[model]["df_list_repetition_penalty"]
        ]
        recall = [
            df["recall"].mean() for df in result[model]["df_list_repetition_penalty"]
        ]
        newline_score = [
            df["newline_score"].mean()
            for df in result[model]["df_list_repetition_penalty"]
        ]

        repetition_score = [
            df["repetition_score"].mean()
            for df in result[model]["df_list_repetition_penalty"]
        ]

        if adjusted_f1:
            precision, recall = adjust_perf_scores_with_repetition_penalty(
                result[model], precision, recall
            )

        # f1 = [df["f1"].mean() for df in result[model]["df_list_repetition_penalty"]]
        f1 = [2 * (p * r) / (p + r) for p, r in zip(precision, recall)]

        best_f1.append(max(f1))
        best_f1_index = f1.index(best_f1[-1])
        best_repetition_penalty.append(df["repetition_penalty"][best_f1_index])

        best_precision.append(precision[best_f1_index])
        best_recall.append(recall[best_f1_index])
        best_mtr.append(newline_score[best_f1_index] + repetition_score[best_f1_index])

        print(
            f"best repetition penalty: {best_repetition_penalty[-1]}, best f1: {best_f1[-1]}, precision: {best_precision[-1]}, recall: {best_recall[-1]}"
        )

        df = result[model]["df_list_repetition_penalty"][best_f1_index]

        model_names.append(
            f"{model} (RP={best_repetition_penalty[-1]})"
        )  # Add the model name to the list

        # print sum for columns: newline_score, repetition_score
        print(
            f"newline_score: {df['newline_score'].sum()}, repetition_score: {df['repetition_score'].sum()}"
        )

    if ref_result is not None:
        print("ref_result:", ref_result)
        for model in ref_result.keys():
            model_names.append(model)
            df = pd.read_csv(ref_result[model])
            # df = df[df["id"].isin(wikidata_df["id"])]

            best_precision.append(df["precision"].mean())
            best_recall.append(df["recall"].mean())
            f1 = (
                2
                * (best_precision[-1] * best_recall[-1])
                / (best_precision[-1] + best_recall[-1])
            )
            # best_f1.append(df["f1"].mean())
            best_f1.append(f1)
            best_mtr.append(0)

    # Create a DataFrame with the statistics
    data = (
        pd.DataFrame(
            {
                "Model": model_names,
                "Adjusted Precision with RP": best_precision,
                "Adjusted Recall with RP": best_recall,
                "Adjusted F1 with RP": best_f1,
            }
        )
        if adjusted_f1
        else pd.DataFrame(
            {
                "Model": model_names,
                "Precision": best_precision,
                "Recall": best_recall,
                "F1": best_f1,
            }
        )
    )
    columns = list(data.columns)

    # Melt the DataFrame to a long format
    data_melted = data.melt(id_vars="Model", var_name="Metric", value_name="Score")

    # Pivot the DataFrame to a wide format
    data_pivoted = data_melted.pivot(index="Metric", columns="Model", values="Score")

    # make sure the columns are following the order of the models
    data_pivoted = data_pivoted[model_names]

    # make sure three groups in the order of precision, recall, f1
    data_pivoted = data_pivoted.reindex(columns[1:])

    # Plot the statistics
    plt.figure(figsize=(10, 6))
    ax = data_pivoted.plot(kind="bar", ax=plt.gca(), width=0.9)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")

    # Set the rotation of the x-axis labels to 0 degrees
    plt.xticks(rotation=0)

    # Format the y-axis to display as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # get the max value of the y-axis
    a1 = max(best_precision)
    a2 = max(best_recall)
    a3 = max(best_f1)

    max_value = max([a1, a2, a3]) * 1.12
    print("max_value:", max_value)

    # Set the y-axis limit up to 70%
    ax.set_ylim(0, max_value)

    # Add the values above each bar
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height() * 100:.1f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 10),
            textcoords="offset points",
            rotation=90,
        )

    plt.show()
    return data_pivoted, best_mtr


def plot_best_performance_ms_macro(
    result,
    models=None,
    title="Models with Best RAP - Performance",
    ref_result=None,
    skip_generic_prompt=False,
    include_adjusted_performance=True,
):
    # Initialize lists to store the statistics
    model_names = []
    best_f1 = []
    best_afrp = []
    best_repetition_penalty = []
    best_bleu1 = []
    best_rougeL = []
    best_mtr = []

    if models is None:
        models = result.keys()
    for model in models:
        if skip_generic_prompt and "generic prompt" in model:
            continue
        print(f"model: {model}")
        df = result[model]["df_overall"]

        # Calculate the statistics
        bleu1 = [x for x in df["bleu1"]]
        rougeL = [x for x in df["rougeL"]]
        f1 = [2 * (p * r) / (p + r) for p, r in zip(bleu1, rougeL)]

        newline_score = [
            df["newline_score"].mean()
            for df in result[model]["df_list_repetition_penalty"]
        ]
        # print(f"newline_score: {newline_score}")

        repetition_score = [
            df["repetition_score"].mean()
            for df in result[model]["df_list_repetition_penalty"]
        ]
        # print(f"repetition_score: {repetition_score}")

        answer_len = [
            df["answer_len"].mean()
            for df in result[model]["df_list_repetition_penalty"]
        ]

        afrp = [
            calc_adjusted_performance(f, n + r, l)
            for f, n, r, l in zip(f1, newline_score, repetition_score, answer_len)
        ]

        best_afrp.append(max(afrp if include_adjusted_performance else f1))
        best_afrp_index = (
            afrp.index(best_afrp[-1])
            if include_adjusted_performance
            else f1.index(best_afrp[-1])
        )
        best_repetition_penalty.append(df["repetition_penalty"][best_afrp_index])

        best_f1.append(f1[best_afrp_index])
        best_bleu1.append(bleu1[best_afrp_index])
        best_rougeL.append(rougeL[best_afrp_index])
        best_mtr.append(
            newline_score[best_afrp_index] + repetition_score[best_afrp_index]
        )

        # print(
        #     f"best repetition penalty: {best_repetition_penalty[-1]}, best afrp: {best_afrp[-1]}, f1: {best_f1[-1]}"
        # )

        df = result[model]["df_list_repetition_penalty"][best_afrp_index]

        model_names.append(
            f"{model} (RP={best_repetition_penalty[-1]})"
        )  # Add the model name to the list

    if ref_result is not None:
        print("ref_result:", ref_result)
        for model in ref_result.keys():
            model_names.append(model)
            df = pd.read_csv(ref_result[model], comment="#", on_bad_lines="warn")
            # df = df[df["id"].isin(wikidata_df["id"])]

            p = df["bleu1"][0]
            best_bleu1.append(p)

            r = df["rougeL"][0]
            best_rougeL.append(r)

            f1 = 2 * p * r / (p + r) if p + r > 0 else 0
            best_f1.append(f1)
            best_afrp.append(f1)
            best_mtr.append(0)

    # print("model_names:", model_names)
    # print("best_f1:", best_f1)
    # print("best_afrp:", best_afrp)

    # Create a DataFrame with the statistics
    data = (
        pd.DataFrame(
            {
                "Model": model_names,
                "RAP - Perf Score": best_afrp,
                "Overall Perf Score": best_f1,
            }
        )
        if include_adjusted_performance
        else pd.DataFrame(
            {
                "Model": model_names,
                "Bleu-1": best_bleu1,
                "Rouge-L": best_rougeL,
                "Overall Perf Score": best_f1,
            }
        )
    )

    # Melt the DataFrame to a long format
    data_melted = data.melt(id_vars="Model", var_name="Metric", value_name="Score")

    # Pivot the DataFrame to a wide format
    data_pivoted = data_melted.pivot(index="Metric", columns="Model", values="Score")

    # make sure the columns are following the order of the models
    data_pivoted = data_pivoted[model_names]

    columns = list(data.columns)
    data_pivoted = data_pivoted.reindex(columns[1:])

    # Plot the statistics
    plt.figure(figsize=(10, 6))
    ax = data_pivoted.plot(kind="bar", ax=plt.gca(), width=0.9)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")

    # Set the rotation of the x-axis labels to 0 degrees
    plt.xticks(rotation=0)

    # Format the y-axis to display as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # get the max value of the y-axis
    a1 = max(best_afrp)
    a2 = max(best_f1)
    a3 = max(best_bleu1)
    a4 = max(best_rougeL)

    max_value = (
        max([a1, a2] if include_adjusted_performance else [a1, a2, a3, a4]) * 1.12
    )
    print("max_value:", max_value)

    # Set the y-axis limit up to 70%
    ax.set_ylim(0, max_value)

    # Add the values above each bar
    for p in ax.patches:
        ax.annotate(
            f"{p.get_height() * 100:.1f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 10),
            textcoords="offset points",
            rotation=90,
        )

    plt.show()
    return data_pivoted, best_mtr


all_open_source_models = [
    "gemma-1.1-2b-it",
    "Phi-3-mini-128k-instruct",
    "gemma-1.1-7b-it",
    "Llama-2-7b-chat-hf",
    "Mistral-7B-Instruct-v0.2",
    "Meta-Llama-3-8B-Instruct",
    "Llama-2-13b-chat-hf",
    "Llama-2-70b-chat-hf",
    "Meta-Llama-3-70B-Instruct",
]


def load_for_repetition_penalty_ms_macro(
    csv_result_file, repetition_penalty, force_recalculate=False
):
    result_file = replace_last(
        csv_result_file, ".csv", f"_rpp_{repetition_penalty:.2f}.csv"
    )
    df = load_with_newline_and_repetition_scores(
        result_file, force_recalculate=force_recalculate
    )

    return df


# MS MACRO
def plot_performance_scores_ms_macro(
    result,
    models=None,
    title="Performance",
):
    if models is None:
        models = result.keys()
    for model in models:
        print(f"model: {model}")
        df = result[model]["df_overall"]
        # print(result[model]["df_list_repetition_penalty"][0].describe())

        # Calculate the statistics
        bleu1 = list(df["bleu1"])
        rougeL = list(df["rougeL"])
        f1 = [2 * (p * r) / (p + r) for p, r in zip(bleu1, rougeL)]
        best_f1 = max(f1)
        best_f1_index = f1.index(best_f1)

        bleu1, rougeL = adjust_perf_scores_with_repetition_penalty(
            result[model], bleu1, rougeL
        )
        afrp = [2 * (p * r) / (p + r) for p, r in zip(bleu1, rougeL)]

        # f1 = [df["f1"].mean() for df in result[model]["df_list_repetition_penalty"]]
        best_afrp = max(afrp)
        best_afrp_index = afrp.index(best_afrp)

        repetition_penalties = list(df["repetition_penalty"])

        # line plot for precision, recall, f1
        plt.figure(figsize=(10, 6))

        plt.axvspan(
            repetition_penalties[best_f1_index] - 0.01,
            repetition_penalties[best_f1_index] + 0.01,
            alpha=0.5,
            edgecolor="none",
            facecolor="blue",
        )

        plt.axvspan(
            repetition_penalties[best_afrp_index] - 0.01,
            repetition_penalties[best_afrp_index] + 0.01,
            alpha=0.5,
            edgecolor="none",
            facecolor="orange",
        )

        plt.plot(
            repetition_penalties,
            f1,
            label="Overall Perf Score",
            marker="D",
            color="blue",
        )
        plt.plot(
            repetition_penalties,
            afrp,
            label="RAP - Perf Score",
            marker="o",
            color="orange",
        )

        plt.xlabel("Repetition Penalties")
        plt.ylabel("Score")
        # plt.xlim(0.99, 1.31)
        # y in percentage
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.title(f"{model} {title}")
        plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")

        plt.show()


def plot_repetition_factors(result, groups):
    for group in groups:
        # Plot the statistics
        plt.figure(figsize=(10, 6))

        max_value = 0
        for model in result.keys():
            if not group in model.lower():
                continue
            print(f"model: {model}")
            df = result[model]["df_overall"]
            repetition_panelties = [
                repetition_penalty for repetition_penalty in df["repetition_penalty"]
            ]

            mean_score = [
                df["total_repetitions"].mean()
                for df in result[model]["df_list_repetition_penalty"]
            ]

            sns.lineplot(x=repetition_panelties, y=mean_score, label=model)

            new_max = max(mean_score)
            if new_max > max_value:
                max_value = new_max

        max_value = max_value * 1.05
        # if max_value < 1.5:
        #     max_value = 1.5
        # set ylimit
        plt.ylim(0, max_value)

        # show grid
        plt.grid(True)
        plt.xlabel("Repetition Penalties")
        plt.ylabel("Mean Total Repetitions")
        plt.title("Mean Total Repetitions vs Repetition Penalties")
        plt.legend()

        plt.show()


def plot_repetition_factors_by_group(result, group_filter=None):
    markers = ["D", "o", "s", "x"]
    colors = ["blue", "orange", "green", "red"]

    # Plot the statistics
    plt.figure(figsize=(10, 6))
    index = 0
    max_value = 0

    for model in result.keys():
        if group_filter is not None and group_filter not in model:
            continue

        print(f"model: {model}")

        df = result[model]["df_overall"]
        repetition_panelties = [
            repetition_penalty for repetition_penalty in df["repetition_penalty"]
        ]

        # Calculate the statistics
        mean_score = [
            df["total_repetitions"].mean()
            for df in result[model]["df_list_repetition_penalty"]
        ]
        if len(mean_score) != len(repetition_panelties):
            print(
                f"model: {model} has different length of repetition penalties and mean score"
            )
            print("repetition_panelties:", len(repetition_panelties))
            print("mean_score:", len(mean_score))
            continue

        new_max = max(mean_score)
        if new_max > max_value:
            max_value = new_max

        sns.lineplot(
            x=repetition_panelties,
            y=mean_score,
            label=model,
            marker=markers[index],
            color=colors[index],
        )

        index += 1

    max_value = max_value * 1.05
    # if max_value < 1.5:
    #     max_value = 1.5
    # set ylimit
    plt.ylim(0, max_value)
    max_value = 0

    plt.xlabel("Repetition Penalties")
    plt.ylabel("Mean Total Repetitions")
    plt.title("Mean Total Repetitions vs Repetition Penalties")
    plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left")

    plt.show()


ms_marco_csv_result_files = [
    "data/results/gemma-1.1-2b-it(RAG - Generic Prompt)_mm.csv",
    "data/results/gemma-1.1-2b-it(RAG - Chat Template)_mm.csv",
    "data/results/gemma-1.1-2b-it(Non-RAG)_mm.csv",
    "data/results/Phi-3-mini-128k-instruct(RAG - Generic Prompt)_mm.csv",
    "data/results/Phi-3-mini-128k-instruct(RAG - Chat Template)_mm.csv",
    "data/results/Phi-3-mini-128k-instruct(Non-RAG)_mm.csv",
    "data/results/gemma-1.1-7b-it(RAG - Generic Prompt)_mm.csv",
    "data/results/gemma-1.1-7b-it(RAG - Chat Template)_mm.csv",
    "data/results/gemma-1.1-7b-it(Non-RAG)_mm.csv",
    "data/results/Llama-2-7b-chat-hf(RAG - Generic Prompt)_mm.csv",
    "data/results/Llama-2-7b-chat-hf(RAG - Chat Template)_mm.csv",
    "data/results/Llama-2-7b-chat-hf(Non-RAG)_mm.csv",
    "data/results/Mistral-7B-Instruct-v0.2(RAG - Generic Prompt)_mm.csv",
    "data/results/Mistral-7B-Instruct-v0.2(RAG - Chat Template)_mm.csv",
    "data/results/Mistral-7B-Instruct-v0.2(Non-RAG)_mm.csv",
    "data/results/Meta-Llama-3-8B-Instruct(RAG - Generic Prompt)_mm.csv",
    "data/results/Meta-Llama-3-8B-Instruct(RAG - Chat Template)_mm.csv",
    "data/results/Meta-Llama-3-8B-Instruct(Non-RAG)_mm.csv",
    "data/results/Llama-2-13b-chat-hf(RAG - Generic Prompt)_mm.csv",
    "data/results/Llama-2-13b-chat-hf(RAG - Chat Template)_mm.csv",
    "data/results/Llama-2-13b-chat-hf(Non-RAG)_mm.csv",
    "data/results/Llama-2-70b-chat-hf(RAG - Generic Prompt)_mm.csv",
    "data/results/Llama-2-70b-chat-hf(RAG - Chat Template)_mm.csv",
    "data/results/Llama-2-70b-chat-hf(Non-RAG)_mm.csv",
    "data/results/Meta-Llama-3-70B-Instruct(RAG - Generic Prompt)_mm.csv",
    "data/results/Meta-Llama-3-70B-Instruct(RAG - Chat Template)_mm.csv",
    "data/results/Meta-Llama-3-70B-Instruct(Non-RAG)_mm.csv",
]

webqsp_csv_result_files = [
    "data/results/gemma-1.1-2b-it(RAG - Generic Prompt)_wd.csv",
    "data/results/gemma-1.1-2b-it(RAG - Chat Template)_wd.csv",
    "data/results/gemma-1.1-2b-it(Non-RAG)_wd.csv",
    "data/results/Phi-3-mini-128k-instruct(RAG - Generic Prompt)_wd.csv",
    "data/results/Phi-3-mini-128k-instruct(RAG - Chat Template)_wd.csv",
    "data/results/Phi-3-mini-128k-instruct(Non-RAG)_wd.csv",
    "data/results/gemma-1.1-7b-it(RAG - Generic Prompt)_wd.csv",
    "data/results/gemma-1.1-7b-it(RAG - Chat Template)_wd.csv",
    "data/results/gemma-1.1-7b-it(Non-RAG)_wd.csv",
    "data/results/Llama-2-7b-chat-hf(RAG - Generic Prompt)_wd.csv",
    "data/results/Llama-2-7b-chat-hf(RAG - Chat Template)_wd.csv",
    "data/results/Llama-2-7b-chat-hf(Non-RAG)_wd.csv",
    "data/results/Mistral-7B-Instruct-v0.2(RAG - Generic Prompt)_wd.csv",
    "data/results/Mistral-7B-Instruct-v0.2(RAG - Chat Template)_wd.csv",
    "data/results/Mistral-7B-Instruct-v0.2(Non-RAG)_wd.csv",
    "data/results/Meta-Llama-3-8B-Instruct(RAG - Generic Prompt)_wd.csv",
    "data/results/Meta-Llama-3-8B-Instruct(RAG - Chat Template)_wd.csv",
    "data/results/Meta-Llama-3-8B-Instruct(Non-RAG)_wd.csv",
    "data/results/Llama-2-13b-chat-hf(RAG - Generic Prompt)_wd.csv",
    "data/results/Llama-2-13b-chat-hf(RAG - Chat Template)_wd.csv",
    "data/results/Llama-2-13b-chat-hf(Non-RAG)_wd.csv",
    "data/results/Llama-2-70b-chat-hf(RAG - Generic Prompt)_wd.csv",
    "data/results/Llama-2-70b-chat-hf(RAG - Chat Template)_wd.csv",
    "data/results/Llama-2-70b-chat-hf(Non-RAG)_wd.csv",
    "data/results/Meta-Llama-3-70B-Instruct(RAG - Generic Prompt)_wd.csv",
    "data/results/Meta-Llama-3-70B-Instruct(RAG - Chat Template)_wd.csv",
    "data/results/Meta-Llama-3-70B-Instruct(Non-RAG)_wd.csv",
]


def calc_rap_scores(
    result, precision="precision", recall="recall", penalty_function="cubic"
):
    newline_score = [
        df["newline_score"].mean() for df in result["df_list_repetition_penalty"]
    ]

    repetition_score = [
        df["repetition_score"].mean() for df in result["df_list_repetition_penalty"]
    ]

    if precision in result["df_list_repetition_penalty"][0].columns:
        precision = [
            df[precision].mean() for df in result["df_list_repetition_penalty"]
        ]
        recall = [df[recall].mean() for df in result["df_list_repetition_penalty"]]
    else:
        precision = result["df_overall"][precision]
        recall = result["df_overall"][recall]

    f1 = [2 * (p * r) / (p + r) for p, r in zip(precision, recall)]

    nrr = [
        1 - (n + r) / s
        for f, n, r, s in zip(
            f1, newline_score, repetition_score, result["df_overall"]["answer_len"]
        )
    ]

    rap = [
        calc_adjusted_performance(f, 1 - n, penalty_function=penalty_function)
        for f, n in zip(f1, nrr)
    ]

    return newline_score, repetition_score, f1, rap, nrr


def get_model_name(csv_result_file):
    parts = re.split(r"[_/]", csv_result_file)
    print(f"parts: {parts}")
    model_name = parts[2]
    return model_name


def load_webqsp_result(
    csv_result_files, force_recalculate=False, save=False, penalty_function="cubic"
):
    result = {}
    for i, csv_result_file in enumerate(csv_result_files):
        try:
            df = pd.read_csv(csv_result_file)
            model_name = get_model_name(csv_result_file)
            print(f"\tmodel_name: {model_name}")

            dfs = [
                calculate_performance_score(
                    csv_result_file,
                    repetition_penalty,
                    force_recalculate=force_recalculate,
                )
                for repetition_penalty in df["repetition_penalty"]
            ]

            answer_lens = []
            for df_rpp in dfs:
                answer_lens.append(df_rpp["answer_len"].mean())
            df["answer_len"] = answer_lens

            result[model_name] = {
                "df_overall": df,
                "df_list_repetition_penalty": dfs,
                "file": csv_result_file,
            }
            newline_score, repetition_score, perf, rap, nrr = calc_rap_scores(
                result[model_name], penalty_function=penalty_function
            )
            df["newline_score"] = newline_score
            df["repetition_score"] = repetition_score
            df["total_repetitions"] = df["newline_score"] + df["repetition_score"]
            df["perf"] = perf
            df["nrr"] = nrr
            df["rap"] = rap
            df["rr"] = df["nrr"].apply(lambda x: 1 - x)
            df["rrp"] = df["rr"].apply(lambda x: x * 100)
            if save:
                df.to_csv(csv_result_file, index=False)
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()

    return result


def load_ms_marco_result(
    csv_result_files,
    force_recalculate=False,
    calc_bertscore=True,
    save=False,
    penalty_function="cubic",
):
    result = {}
    for csv_result_file in csv_result_files:
        try:
            df = pd.read_csv(csv_result_file)
            model_name = get_model_name(csv_result_file)
            print(f"\tmodel_name: {model_name}")

            dfs = [
                load_for_repetition_penalty_ms_macro(
                    csv_result_file,
                    repetition_penalty,
                    force_recalculate=force_recalculate,
                )
                for repetition_penalty in df["repetition_penalty"]
            ]

            answer_lens = []
            for df_rpp in dfs:
                answer_lens.append(df_rpp["answer_len"].mean())
            df["answer_len"] = answer_lens

            col = "bert_score" if calc_bertscore else "meteor"
            score_unavailable = col not in df.columns

            if score_unavailable:
                save = True
                bert_meteor_scores = []
                bert_score_references = None
                for df_rpp in dfs:
                    if calc_bertscore:
                        bert_meteor_score = 0

                        for i, row in df_rpp.iterrows():
                            answer = row["answer"]
                            if not isinstance(answer, str):
                                answer = ""
                            bert_meteor_score += bert_score.compute(
                                predictions=[answer],
                                references=[row["ground_truth"][0]],
                                lang="en",
                                model_type="microsoft/deberta-large-mnli",
                            )["f1"][0]
                        # get average of bertscore
                        bert_meteor_score = bert_meteor_score / len(df_rpp)

                        print(f"bert_score: {bert_meteor_score}")
                    else:
                        bert_meteor_score = meteor.compute(
                            predictions=df_rpp["answer"],
                            references=df_rpp["ground_truth"],
                        )["meteor"]

                    bert_meteor_scores.append(bert_meteor_score)

                df[col] = bert_meteor_scores

            result[model_name] = {
                "df_overall": df,
                "df_list_repetition_penalty": dfs,
                "file": csv_result_file,
            }
            newline_score, repetition_score, perf, rap, nrr = calc_rap_scores(
                result[model_name],
                precision=col,
                recall=col,
                penalty_function=penalty_function,
            )
            df["newline_score"] = newline_score
            df["repetition_score"] = repetition_score
            df["total_repetitions"] = df["newline_score"] + df["repetition_score"]
            df["perf"] = perf
            df["nrr"] = nrr
            df["rap"] = rap
            df["rr"] = df["nrr"].apply(lambda x: 1 - x)
            df["rrp"] = df["rr"].apply(lambda x: x * 100)

            if save:
                df.to_csv(csv_result_file, index=False)
        except Exception as e:
            print("An error occurred:", e)
            traceback.print_exc()
            print(f"csv_result_file: {csv_result_file}")

    return result
