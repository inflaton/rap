import abc
import json
import os
import re
import time
import urllib
from queue import Queue
from threading import Thread
from typing import List, Optional
from urllib.parse import quote, urlparse, urlunparse

from langchain.chains.base import Chain

from app_modules.llm_loader import LLMLoader, TextIteratorStreamer
from app_modules.utils import remove_extra_spaces

chat_history_enabled = os.getenv("CHAT_HISTORY_ENABLED", "false").lower() == "true"


def get_system_prompt_and_user_message(orca=False):
    # system_prompt = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
    system_prompt = (
        "You are Orca, an AI language model created by Microsoft. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
        if orca
        else "You are a chatbot having a conversation with a human."
    )

    user_message = "{input}"

    if chat_history_enabled:
        user_message = "Chat History:\n\n{history} \n\n" + user_message
        system_prompt += " Read the chat history to get context."

    return system_prompt, user_message


class LLMInference(metaclass=abc.ABCMeta):
    def __init__(self, llm_loader):
        self.llm_loader = llm_loader
        self.chain = None
        self.pattern = re.compile(r"\s*<.+>$")

    @abc.abstractmethod
    def create_chain(self) -> Chain:
        pass

    def get_chain(self) -> Chain:
        if self.chain is None:
            self.chain = self.create_chain()

        return self.chain

    def reset(self) -> None:
        self.chain = None

    def _process_inputs(self, inputs):
        return inputs

    def _normalize_result(self, result):
        # print(f"_normalize_result: {result}")
        if isinstance(result, list):
            result = result[0]

        key = "text" if "text" in result else "generated_text"
        if key in result:
            result["answer"] = result[key]
            del result[key]

        result["answer"] = self.pattern.sub("", result["answer"])
        return result

    def _process_results(self, results):
        if isinstance(results, list):
            return [self._normalize_result(result) for result in results]

        return self._normalize_result(results)

    def _run_batch(self, chain, inputs):
        if self.llm_loader.llm_model_type == "huggingface":
            results = self.llm_loader.llm.pipeline(inputs)
        else:
            results = chain.batch(inputs)

        return results

    def run_chain(self, chain, inputs, callbacks: Optional[List] = []):
        inputs = self._process_inputs(inputs)

        # check if inputs is an array
        if isinstance(inputs, list):
            results = self._run_batch(chain, inputs)
        else:
            results = chain.invoke(inputs, {"callbacks": callbacks})

        return self._process_results(results)

    def call_chain(
        self,
        inputs,
        streaming_handler,
        q: Queue = None,
        testing: bool = False,
    ):
        print(json.dumps(inputs, indent=4))
        if self.llm_loader.huggingfaceStreamingEnabled():
            self.llm_loader.lock.acquire()

        try:
            if self.llm_loader.huggingfaceStreamingEnabled():
                self.llm_loader.streamer.reset(q)

            chain = self.get_chain()
            result = (
                self._run_chain_with_streaming_handler(
                    chain, inputs, streaming_handler, testing
                )
                if streaming_handler is not None
                else self.run_chain(chain, inputs)
            )

            if "answer" in result:
                result["answer"] = remove_extra_spaces(result["answer"])

            return result
        finally:
            if self.llm_loader.huggingfaceStreamingEnabled():
                self.llm_loader.lock.release()

    def _execute_chain(self, chain, inputs, q, sh):
        q.put(self.run_chain(chain, inputs, callbacks=[sh]))

    def _run_chain_with_streaming_handler(
        self, chain, inputs, streaming_handler, testing
    ):
        que = Queue()

        t = Thread(
            target=self._execute_chain,
            args=(chain, inputs, que, streaming_handler),
        )
        t.start()

        if self.llm_loader.huggingfaceStreamingEnabled():
            count = (
                2
                if "chat_history" in inputs and len(inputs.get("chat_history")) > 0
                else 1
            )

            while count > 0:
                try:
                    for token in self.llm_loader.streamer:
                        if not testing:
                            streaming_handler.on_llm_new_token(token)

                    self.llm_loader.streamer.reset()
                    count -= 1
                except Exception:
                    if not testing:
                        print("nothing generated yet - retry in 0.5s")
                    time.sleep(0.5)

        t.join()
        return que.get()

    def apply_chat_template(self, user_message):
        result = (
            []
            if re.search(r"gemma|mistral", self.llm_loader.model_name, re.IGNORECASE)
            else [
                {
                    "role": "system",
                    "content": get_system_prompt_and_user_message()[0],
                }
            ]
        )
        result.append(
            {
                "role": "user",
                "content": user_message,
            }
        )
        return result
