import json
import os
import sys
import pandas as pd
from timeit import default_timer as timer
import nltk

sys.path.insert(0, os.getcwd())

chatting = len(sys.argv) > 1 and sys.argv[1] == "chat"

if chatting:
    os.environ["BATCH_SIZE"] = "1"

from app_modules.init import app_init
from app_modules.llm_qa_chain import QAChain
from app_modules.utils import print_llm_response, calc_metrics, detect_repetition_scores

llm_loader, qa_chain = app_init()

if chatting:
    print("Starting chat mode")
    while True:
        question = input("Please enter your question: ")
        if question.lower() == "exit":
            break
        result = qa_chain.call_chain({"question": question, "chat_history": []}, None)
        print_llm_response(result)

    sys.exit(0)

num_of_questions = 0

if len(sys.argv) > 1:
    num_of_questions = int(sys.argv[1])

# Create an empty DataFrame with column names
df = pd.DataFrame(
    columns=[
        "id",
        "question",
        "answer",
    ]
)

batch_size = int(os.getenv("BATCH_SIZE", "1"))
print(f"Batch size: {batch_size}")

questions_file_path = os.environ.get("QUESTIONS_FILE_PATH")
debug_retrieval = os.getenv("DEBUG_RETRIEVAL", "false").lower() == "true"

# Open the file for reading
print(f"Reading questions from file: {questions_file_path}")
test_data = json.loads(open(questions_file_path).read())

if isinstance(test_data, dict):
    questions = [test_data[key] for key in test_data.keys()]
    ids = [key for key in test_data.keys()]
else:
    questions = test_data
    ids = [row["id"] for row in questions]

if num_of_questions > 0:
    questions = questions[:num_of_questions]

print(f"Number of questions: {len(questions)}")

if __name__ == "__main__":
    chat_start = timer()
    index = 0

    while index < len(questions):
        batch_ids = ids[index : index + batch_size]
        batch_questions = [q["question"] for q in questions[index : index + batch_size]]

        if isinstance(qa_chain, QAChain):
            inputs = [{"question": q, "chat_history": []} for q in batch_questions]
        else:
            inputs = [{"question": q} for q in batch_questions]

        start = timer()
        result = qa_chain.call_chain(inputs, None)
        end = timer()
        print(f"Completed in {end - start:.3f}s")

        # print("result:", result)
        batch_answers = [r["answer"] for r in result]

        for id, question, answer in zip(batch_ids, batch_questions, batch_answers):
            df.loc[len(df)] = {
                "id": id,
                "question": question,
                "answer": answer,
            }

        index += batch_size

        for r in result:
            print_llm_response(r, debug_retrieval)

    chat_end = timer()
    total_time = chat_end - chat_start
    print(f"Total time used: {total_time:.3f} s")

    df2 = pd.DataFrame(
        columns=[
            "id",
            "question",
            "answer",
            "word_count",
            "ground_truth",
        ]
    )

    for i in range(len(df)):
        question = questions[i]
        answer = df["answer"][i]
        query = df["question"][i]
        id = df["id"][i]

        ground_truth = question[
            "wellFormedAnswers" if "wellFormedAnswers" in question else "answers"
        ]

        word_count = len(nltk.word_tokenize(answer))

        df2.loc[len(df2)] = {
            "id": id,
            "question": query,
            "answer": answer,
            "word_count": word_count,
            "ground_truth": ground_truth,
        }

    df2[["newline_score", "repetition_score", "total_repetitions"]] = df2[
        "answer"
    ].apply(detect_repetition_scores)

    pd.options.display.float_format = "{:.3f}".format
    print(df2.describe())

    word_count = df2["word_count"].sum()

    csv_file = (
        os.getenv("TEST_RESULTS_CSV_FILE") or f"qa_batch_{batch_size}_test_results.csv"
    )
    with open(csv_file, "w") as f:
        f.write(
            f"# RAG: {isinstance(qa_chain, QAChain)} questions: {questions_file_path}\n"
        )
        f.write(
            f"# model: {llm_loader.model_name} repetition_penalty: {llm_loader.repetition_penalty}\n"
        )

    df2.to_csv(csv_file, mode="a", index=False, header=True)
    print(f"test results saved to file: {csv_file}")

    scores = calc_metrics(df2)

    df = pd.DataFrame(
        {
            "model": [llm_loader.model_name],
            "repetition_penalty": [llm_loader.repetition_penalty],
            "word_count": [word_count],
            "inference_time": [total_time],
            "inference_speed": [word_count / total_time],
            "bleu1": [scores["bleu_scores"]["bleu"]],
            "rougeL": [scores["rouge_scores"]["rougeL"]],
        }
    )

    print(f"Number of words generated: {word_count}")
    print(f"Average generation speed: {word_count / total_time:.3f} words/s")

    csv_file = os.getenv("ALL_RESULTS_CSV_FILE") or "qa_chain_all_results.csv"
    file_existed = os.path.exists(csv_file) and os.path.getsize(csv_file) > 0
    df.to_csv(csv_file, mode="a", index=False, header=not file_existed)
    print(f"all results appended to file: {csv_file}")
