import json
import os
import sys
import evaluate
import gradio as gr
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import InferenceClient, login

found_dotenv = find_dotenv(".env")

if len(found_dotenv) == 0:
    found_dotenv = find_dotenv(".env.example")
print(f"loading env vars from: {found_dotenv}")
load_dotenv(found_dotenv, override=False)

path = os.path.dirname(found_dotenv) + "/src"
print(f"Adding {path} to sys.path")
sys.path.append(path)

from eval_modules.utils import calc_perf_scores, detect_repetitions

model_name = os.getenv("MODEL_NAME") or "microsoft/Phi-3.5-mini-instruct"
hf_token = os.getenv("HF_TOKEN")

login(token=hf_token, add_to_git_credential=True)

questions_file_path = os.getenv("QUESTIONS_FILE_PATH") or "./data/datasets/ms_macro.json"

questions = json.loads(open(questions_file_path).read())
examples = [[question["question"].strip()] for question in questions]
print(f"Loaded {len(examples)} examples")

qa_system_prompt = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
# client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
# client = InferenceClient("HuggingFaceH4/zephyr-7b-gemma-v0.1")
# client = InferenceClient("microsoft/Phi-3.5-mini-instruct")
client = InferenceClient(model_name, token=hf_token)


def chat(
    message,
    history: list[tuple[str, str]],
    system_message,
    temperature=0,
    max_tokens=256,
    top_p=0.95,
):
    chat = []
    for item in history:
        chat.append({"role": "user", "content": item[0]})
        if item[1] is not None:
            chat.append({"role": "assistant", "content": item[1]})

    index = -1
    if [message] in examples:
        index = examples.index([message])
        message = f"{qa_system_prompt}\n\n{questions[index]['context']}\n\nQuestion: {message}"
        print("RAG prompt:", message)

    chat.append({"role": "user", "content": message})

    messages = [{"role": "system", "content": system_message}]
    messages.append({"role": "user", "content": message})

    partial_text = ""

    finish_reason = None
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
        seed=42,
    ):
        finish_reason = message.choices[0].finish_reason
        # print("finish_reason:", finish_reason)

        if finish_reason is None:
            new_text = message.choices[0].delta.content
            partial_text += new_text
            yield partial_text
        else:
            break

    answer = partial_text
    (whitespace_score, repetition_score, total_repetitions) = detect_repetitions(answer)
    partial_text += "\n\nRepetition Metrics:\n"
    partial_text += f"1. EWC Repetition Score: {whitespace_score:.3f}\n"
    partial_text += f"1. Text Repetition Score: {repetition_score:.3f}\n"
    partial_text += f"1. Total Repetitions: {total_repetitions:.3f}\n"
    rr = total_repetitions / len(answer) if len(answer) > 0 else 0
    partial_text += f"1. Repetition Ratio: {rr:.3f}\n"

    if index >= 0:  # RAG
        key = (
            "wellFormedAnswers"
            if "wellFormedAnswers" in questions[index]
            else "answers"
        )
        scores = calc_perf_scores([answer], [questions[index][key]], debug=True)

        partial_text += "\n\n Performance Metrics:\n"
        partial_text += f'1. BLEU-1: {scores["bleu_scores"]["bleu"]:.3f}\n'
        partial_text += f'1. RougeL: {scores["rouge_scores"]["rougeL"]:.3f}\n'
        perf = scores["bert_scores"]["f1"][0]
        partial_text += f"1. BERT-F1: {perf:.3f}\n"
        nrr = 1 - rr
        partial_text += f"1. RAP-BERT-F1: {perf * nrr * nrr * nrr:.3f}\n"

        partial_text += f"\n\nGround truth: {questions[index][key][0]}\n"

    partial_text += f"\n\nThe text generation has ended because: {finish_reason}\n"

    yield partial_text


demo = gr.ChatInterface(
    fn=chat,
    examples=examples,
    cache_examples=False,
    additional_inputs_accordion=gr.Accordion(
        label="⚙️ Parameters", open=False, render=False
    ),
    additional_inputs=[
        gr.Textbox(value="You are a friendly Chatbot.", label="System message"),
        gr.Slider(
            minimum=0, maximum=2, step=0.1, value=0, label="Temperature", render=False
        ),
        gr.Slider(
            minimum=128,
            maximum=4096,
            step=1,
            value=512,
            label="Max new tokens",
            render=False,
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)
demo.launch()
