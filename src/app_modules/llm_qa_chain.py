import json
import os
from typing import List
import pandas as pd
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain
from app_modules.llm_inference import LLMInference
from app_modules.utils import CustomizedConversationSummaryBufferMemory

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.globals import get_debug

retrieve_from_questions_file = os.getenv("RETRIEVER_TYPE") == "questions_file"
apply_chat_template_for_rag = os.getenv("APPLY_CHAT_TEMPLATE_FOR_RAG") == "true"

print(f"retrieve_from_questions_file: {retrieve_from_questions_file}", flush=True)
print(f"apply_chat_template_for_rag: {apply_chat_template_for_rag}", flush=True)

if retrieve_from_questions_file:
    questions_file_path = os.getenv("QUESTIONS_FILE_PATH")
    questions_df = pd.read_json(questions_file_path)
    print(f"Questions file loaded: {questions_file_path}", flush=True)


class DatasetRetriever(BaseRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Get documents relevant to a query.
        Args:
            query: String to find relevant documents for
            run_manager: The callbacks handler to use
        Returns:
            List of relevant documents
        """
        docs = []
        df = questions_df

        # find the query in the df
        filtered = df[df["question"].str.lower() == query.lower()]

        # iterate over the filtered df
        for i in range(len(filtered)):
            docs.append(
                Document(
                    page_content=filtered.iloc[i]["context"],
                    metadata={"source": filtered.iloc[i]["id"]},
                )
            )

        if not docs:
            print(f"No documents found for query: {query}", flush=True)

        return docs


class QAChain(LLMInference):
    def __init__(self, vectorstore, llm_loader):
        super().__init__(llm_loader)
        self.vectorstore = vectorstore

    def create_chain(self) -> Chain:
        if retrieve_from_questions_file:
            retriever = DatasetRetriever()
        else:
            retriever = self.vectorstore.as_retriever(
                search_kwargs=self.llm_loader.search_kwargs
            )

        if os.environ.get("CHAT_HISTORY_ENABLED") == "true":
            memory = CustomizedConversationSummaryBufferMemory(
                llm=self.llm_loader.llm,
                output_key="answer",
                memory_key="chat_history",
                max_token_limit=1024,
                return_messages=True,
            )
            qa = ConversationalRetrievalChain.from_llm(
                self.llm_loader.llm,
                memory=memory,
                chain_type="stuff",
                retriever=retriever,
                get_chat_history=lambda h: h,
                return_source_documents=True,
            )
        else:
            qa = ConversationalRetrievalChain.from_llm(
                self.llm_loader.llm,
                retriever=retriever,
                max_tokens_limit=8192,  # self.llm_loader.max_tokens_limit,
                return_source_documents=True,
            )

        return qa

    def _process_inputs(self, inputs):
        if isinstance(inputs, list) and self.llm_loader.llm_model_type == "huggingface":
            inputs = [self.get_prompt(i) for i in inputs]

        if get_debug():
            print("_process_inputs:", json.dumps(inputs, indent=4))

        return inputs

    def get_prompt(self, inputs):
        qa_system_prompt = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer."

        df = questions_df
        query = inputs["question"]

        # find the query in the df
        filtered = df[df["question"].str.lower() == query.lower()]

        context = filtered.iloc[0]["context"] if len(filtered) > 0 else ""

        if apply_chat_template_for_rag:
            return self.apply_chat_template(
                f"{qa_system_prompt}\n\n{context}\n\nQuestion: {query}"
            )
        else:
            return f"{qa_system_prompt}\n\n{context}\n\nQuestion: {query}\n\nHelpful Answer:"
