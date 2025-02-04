import json
import os
import re

from langchain.chains import ConversationChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.base import Chain

from app_modules.llm_inference import LLMInference, get_system_prompt_and_user_message
from app_modules.utils import CustomizedConversationSummaryBufferMemory
from langchain.chains import LLMChain
from langchain.globals import get_debug

chat_history_enabled = os.getenv("CHAT_HISTORY_ENABLED", "false").lower() == "true"
B_INST, E_INST = "[INST]", "[/INST]"


def create_llama_2_prompt_template():
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    system_prompt, user_message = get_system_prompt_and_user_message()

    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    prompt_template = B_INST + SYSTEM_PROMPT + user_message + E_INST
    return prompt_template


def create_llama_3_prompt_template():
    system_prompt, user_message = get_system_prompt_and_user_message()
    prompt_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{ system_prompt }<|eot_id|><|start_header_id|>user<|end_header_id|>
{ user_message }<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    return prompt_template


def create_phi_3_prompt_template():
    system_prompt, user_message = get_system_prompt_and_user_message()
    prompt_template = f"""<|system|>
{ system_prompt }<|end|>
<|user|>
{ user_message }<|end|>
<|assistant|>
"""

    return prompt_template


def create_orca_2_prompt_template():
    system_prompt, user_message = get_system_prompt_and_user_message(orca=False)

    prompt_template = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant"
    return prompt_template


def create_mistral_prompt_template():
    system_prompt, user_message = get_system_prompt_and_user_message()

    prompt_template = B_INST + system_prompt + "\n\n" + user_message + E_INST
    return prompt_template


def create_gemma_prompt_template():
    return "<start_of_turn>user\n{input}<end_of_turn>\n<start_of_turn>model\n"


def create_prompt_template(model_name):
    print(f"creating prompt template for model: {model_name}")
    if re.search(r"llama-?2", model_name, re.IGNORECASE):
        return create_llama_2_prompt_template()
    elif re.search(r"llama-?3", model_name, re.IGNORECASE):
        return create_llama_3_prompt_template()
    elif re.search(r"phi-?3", model_name, re.IGNORECASE):
        return create_phi_3_prompt_template()
    elif model_name.lower().startswith("orca"):
        return create_orca_2_prompt_template()
    elif model_name.lower().startswith("mistral"):
        return create_mistral_prompt_template()
    elif model_name.lower().startswith("gemma"):
        return create_gemma_prompt_template()

    return (
        """You are a chatbot having a conversation with a human.
{history}
Human: {input}
Chatbot:"""
        if chat_history_enabled
        else """You are a chatbot having a conversation with a human.
Human: {input}
Chatbot:"""
    )


class ChatChain(LLMInference):
    def __init__(self, llm_loader):
        super().__init__(llm_loader)

    def create_chain(self) -> Chain:
        template = create_prompt_template(self.llm_loader.model_name)
        print(f"template: {template}")

        if chat_history_enabled:
            prompt = PromptTemplate(
                input_variables=["history", "input"], template=template
            )
            memory = CustomizedConversationSummaryBufferMemory(
                llm=self.llm_loader.llm, max_token_limit=1024, return_messages=False
            )

            llm_chain = ConversationChain(
                llm=self.llm_loader.llm,
                prompt=prompt,
                verbose=False,
                memory=memory,
            )
        else:
            prompt = PromptTemplate(input_variables=["input"], template=template)
            llm_chain = LLMChain(llm=self.llm_loader.llm, prompt=prompt)

        return llm_chain

    def _process_inputs(self, inputs):
        if not isinstance(inputs, list):
            inputs = {"input": inputs["question"]}
        elif self.llm_loader.llm_model_type == "huggingface":
            inputs = [self.apply_chat_template(input["question"]) for input in inputs]
        else:
            inputs = [{"input": i["question"]} for i in inputs]

        if get_debug():
            print("_process_inputs:", json.dumps(inputs, indent=4))

        return inputs
