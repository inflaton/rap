import os
import sys
import evaluate
import gradio as gr
from huggingface_hub import InferenceClient, login
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import login

found_dotenv = find_dotenv(".env")

if len(found_dotenv) == 0:
    found_dotenv = find_dotenv(".env.example")
print(f"loading env vars from: {found_dotenv}")
load_dotenv(found_dotenv, override=False)

path = os.path.dirname(found_dotenv)
print(f"Adding {path} to sys.path")
sys.path.append(path)

from llm_toolkit.llm_utils import *
from llm_toolkit.translation_utils_v2 import *
from eval_modules.calc_repetitions import detect_repetitions

model_name = os.getenv("MODEL_NAME") or "microsoft/Phi-3.5-mini-instruct"
num_shots = int(os.getenv("NUM_SHOTS", 10))
data_path = os.getenv("DATA_PATH")
hf_token = os.getenv("HF_TOKEN")

login(token=hf_token, add_to_git_credential=True)

comet = evaluate.load("comet", config_name="Unbabel/wmt22-cometkiwi-da", gpus=1)
meteor = evaluate.load("meteor")
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")


def calc_perf_scores(prediction, source, reference, debug=False):
    if debug:
        print("prediction:", prediction)
        print("source:", source)
        print("reference:", reference)

    if reference:
        bleu_scores = bleu.compute(
            predictions=[prediction], references=[reference], max_order=1
        )
        rouge_scores = rouge.compute(predictions=[prediction], references=[reference])
        rouge_scores = rouge.compute(predictions=[prediction], references=[reference])
        meteor_scores = meteor.compute(predictions=[prediction], references=[reference])
    
    comet_metric = comet.compute(
        predictions=[prediction], sources=[source], references=[reference]
    )

    result = {"bleu_scores": bleu_scores, "rouge_scores": rouge_scores, "meteor_scores":meteor_scores, "comet_scores": comet_metric}

    if debug:
        print("result:", result)

    return result

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""
# client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")
client = InferenceClient(model_name, token=hf_token)

datasets = load_translation_dataset(data_path)
print_row_details(datasets["test"].to_pandas())
translation_prompt = get_few_shot_prompt(datasets["train"], num_shots)

examples = [[row["chinese"]] for row in datasets["test"]][:5]

def respond(
    message,
    history: list[tuple[str, str]],
    system_message,
    max_tokens,
    temperature,
    top_p,
):
    messages = [{"role": "system", "content": system_message}]
    source = message

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": translation_prompt.format(input=message)})

    partial_text = ""

    finish_reason = None
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        frequency_penalty=None,  # frequency_penalty,
        presence_penalty=None,  # presence_penalty,
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
    (whitespace_score, repetition_score, total_repetitions) = detect_repetitions(answer, debug=True)
    partial_text += "\n\nRepetition Metrics:\n"
    partial_text += f"1. EWC Repetition Score: {whitespace_score:.3f}\n"
    partial_text += f"1. Text Repetition Score: {repetition_score:.3f}\n"
    partial_text += f"1. Total Repetitions: {total_repetitions:.3f}\n"
    partial_text += (
        f"1. Repetition Ratio: {total_repetitions / len(answer):.3f}\n"
    )

    partial_text += "\n\n Performance Metrics:\n"

    if [source]  in examples:
        idx = examples.index([source])
        reference = datasets["test"]["english"][idx]
    else:
        reference = ""

    scores = calc_perf_scores(answer, source, reference, debug=True)

    partial_text += f'1. COMET: {scores["comet_scores"]["mean_score"]:.3f}\n'
    if reference:
        partial_text += f'1. METEOR: {scores["meteor_scores"]["meteor"]:.3f}\n'
        partial_text += f'1. BLEU-1: {scores["bleu_scores"]["bleu"]:.3f}\n'
        partial_text += f'1. RougeL: {scores["rouge_scores"]["rougeL"]:.3f}\n'
        partial_text += f"\n\nGround truth: {reference}\n"

    partial_text += f"\n\nThe text generation has ended because: {finish_reason}\n"

    yield partial_text

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
demo = gr.ChatInterface(
    respond,
    examples=examples,
    cache_examples=False,
    textbox=gr.Textbox(placeholder="Enter your Chinese sentence for translation"),
    additional_inputs=[
        gr.Textbox(value="You are a helpful assistant that translates Chinese to English.", label="System message"),
        gr.Slider(minimum=1, maximum=2048, value=512, step=1, label="Max new tokens"),
        gr.Slider(minimum=0.1, maximum=4.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.95,
            step=0.05,
            label="Top-p (nucleus sampling)",
        ),
    ],
)


if __name__ == "__main__":
    demo.launch()
