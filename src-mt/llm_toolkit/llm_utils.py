import os
import re
import numpy as np
import torch
import tiktoken
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)
from tqdm import tqdm


def get_template(model_name):
    model_name = model_name.lower()
    if "llama" in model_name:
        return "llama3"
    if "internlm" in model_name:
        return "intern2"
    if "glm" in model_name:
        return "glm4"
    return "chatml"


class OpenAITokenizer:

    def __init__(self, model_name):
        self.model_name = model_name
        self.encoding = tiktoken.get_encoding(model_name)

    def __call__(self, text, return_tensors="pt"):
        return {"input_ids": self.encoding.encode(text)}


def load_tokenizer(model_name):
    if "gpt" in model_name:
        return OpenAITokenizer("cl100k_base")

    return AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="left"
    )


def load_model(
    model_name,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    adapter_name_or_path=None,
    using_llama_factory=False,
):
    print(f"loading model: {model_name} with adapter: {adapter_name_or_path}")

    if using_llama_factory:
        from llamafactory.chat import ChatModel

        template = get_template(model_name)

        args = dict(
            model_name_or_path=model_name,
            adapter_name_or_path=adapter_name_or_path,  # load the saved LoRA adapters
            template=template,  # same to the one in training
            finetuning_type="lora",  # same to the one in training
            quantization_bit=4 if load_in_4bit else None,  # load 4-bit quantized model
        )
        chat_model = ChatModel(args)
        if os.getenv("RESIZE_TOKEN_EMBEDDINGS") == "true":
            chat_model.engine.model.resize_token_embeddings(
                len(chat_model.engine.tokenizer), pad_to_multiple_of=32
            )
        return chat_model.engine.model, chat_model.engine.tokenizer

    tokenizer = load_tokenizer(model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=dtype,
    )

    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        if load_in_4bit
        else AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        )
    )

    if adapter_name_or_path:
        adapter_name = model.load_adapter(adapter_name_or_path)
        model.active_adapters = adapter_name

    if not tokenizer.pad_token:
        print("Adding pad token to tokenizer for model: ", model_name)
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=32)

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def check_gpu():
    # torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
    is_cuda = torch.cuda.is_available()

    # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
    if is_cuda:
        device = torch.device("cuda")
        print("CUDA is available, we have found ", torch.cuda.device_count(), " GPU(s)")
        print(torch.cuda.get_device_name(0))
        print("CUDA version: " + torch.version.cuda)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS is available")
    else:
        device = torch.device("cpu")
        print("GPU/MPS not available, CPU used")
    return device


def test_model(model, tokenizer, prompt, device="cuda"):
    inputs = tokenizer(
        [prompt],
        return_tensors="pt",
    ).to(device)

    text_streamer = TextStreamer(tokenizer)

    _ = model.generate(
        **inputs, max_new_tokens=2048, streamer=text_streamer, use_cache=True
    )


def extract_answer(text, debug=False):
    if text:
        # Remove the begin and end tokens
        text = re.sub(
            r".*?(assistant|\[/INST\]).+?\b",
            "",
            text,
            flags=re.DOTALL | re.MULTILINE,
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

        text = text.split("。")[0].strip()
        if debug:
            print("--------\nstep 4:", text)

        text = re.sub(
            r"^Response:.+?\b",
            "",
            text,
            flags=re.DOTALL | re.MULTILINE,
        )
        if debug:
            print("--------\nstep 5:", text)

    return text


def eval_model(
    model,
    tokenizer,
    eval_dataset,
    device="cuda",
    max_new_tokens=2048,
    repetition_penalty=1.0,
    do_sample=True,
    top_p=0.95,
    top_k=0,  # select from top 0 tokens (because zero, relies on top_p)
    temperature=0.01,
    batch_size=1,
):
    total = len(eval_dataset)
    predictions = []

    model.eval()

    with torch.no_grad():
        for i in tqdm(range(0, total, batch_size)):  # Iterate in batches
            batch_end = min(i + batch_size, total)  # Ensure not to exceed dataset
            batch_prompts = eval_dataset["prompt"][i:batch_end]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,  # Ensure all inputs in the batch have the same length
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                use_cache=False,
            )
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            decoded_output = tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )  # Skip special tokens for clean output
            if i == 0:
                print("Batch output:", decoded_output)
            predictions.extend(decoded_output)

    return predictions


def evaluate_model_with_repetition_penalty(
    model,
    tokenizer,
    model_name,
    dataset,
    on_repetition_penalty_step_completed,
    start_repetition_penalty=1.0,
    end_repetition_penalty=1.3,
    step_repetition_penalty=0.02,
    batch_size=1,
    max_new_tokens=2048,
    device="cuda",
):
    print(f"Evaluating model: {model_name} on {device}")

    for repetition_penalty in np.arange(
        start_repetition_penalty,
        end_repetition_penalty + step_repetition_penalty / 2,
        step_repetition_penalty,
    ):
        # round to 2 decimal places
        repetition_penalty = round(repetition_penalty, 2)
        print(f"*** Evaluating with repetition_penalty: {repetition_penalty}")
        predictions = eval_model(
            model,
            tokenizer,
            dataset,
            device=device,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
        )

        model_name_with_rp = f"{model_name}/rpp-{repetition_penalty:.2f}"

        try:
            on_repetition_penalty_step_completed(
                model_name_with_rp,
                predictions,
            )
        except Exception as e:
            print(e)


def save_model(
    model,
    tokenizer,
    include_gguf=True,
    include_merged=True,
    publish=True,
):
    try:
        token = os.getenv("HF_TOKEN") or None
        model_name = os.getenv("MODEL_NAME")

        save_method = "lora"
        quantization_method = "q5_k_m"

        model_names = get_model_names(
            model_name, save_method=save_method, quantization_method=quantization_method
        )

        model.save_pretrained(model_names["local"])
        tokenizer.save_pretrained(model_names["local"])

        if publish:
            model.push_to_hub(
                model_names["hub"],
                token=token,
            )
            tokenizer.push_to_hub(
                model_names["hub"],
                token=token,
            )

        if include_merged:
            model.save_pretrained_merged(
                model_names["local"] + "-merged", tokenizer, save_method=save_method
            )
            if publish:
                model.push_to_hub_merged(
                    model_names["hub"] + "-merged",
                    tokenizer,
                    save_method="lora",
                    token="",
                )

        if include_gguf:
            model.save_pretrained_gguf(
                model_names["local-gguf"],
                tokenizer,
                quantization_method=quantization_method,
            )

            if publish:
                model.push_to_hub_gguf(
                    model_names["hub-gguf"],
                    tokenizer,
                    quantization_method=quantization_method,
                    token=token,
                )
    except Exception as e:
        print(e)


def print_row_details(df, indices=[0]):
    for index in indices:
        for col in df.columns:
            print("-" * 50)
            print(f"{col}: {df[col].iloc[index]}")
