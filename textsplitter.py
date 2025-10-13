from dotenv import load_dotenv
from langchain.text_splitter import TextSplitter
from typing import List, Any
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import StrOutputParser
import re
import tiktoken
import os
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class LLMTextSplitter(TextSplitter):
    def __init__(self, model_name: str = 'gpt-4o', prompt_type: str = 'wide', count_tokens: bool = False, encoding_name: str = 'cl100k_base', **kwargs: Any):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.prompt_type = prompt_type
        self.count_tokens = count_tokens
        self.encoding_name = encoding_name
        self.model = ChatOpenAI(model_name=self.model_name)
        self.output_paser = StrOutputParser()

    
        wide_topic_template = "Split the text according to the broad topics it deals with and add >>> <<< around each chunk: {text}"
        granular_topic_template = "Split the text into detailed, granular topics and add >>> <<< around each chunk: {text}"

        if prompt_type == "wide":
            self.prompt_template = ChatPromptTemplate.from_template(wide_topic_template)
        elif prompt_type == "granular":
            self.prompt_template = ChatPromptTemplate.from_template(granular_topic_template)
        else:
            raise ValueError("Invalid prompt type specified. Choose 'wide' or 'granular'.")

        self.chain = self.prompt_template | self.model | self.output_paser
    
    def split_text(self, text: str) -> List[str]:
        if self.count_tokens:
            token_count = self.num_tokens_from_string(text)
            print(f"Token count of input text: {token_count}")

        response = self.chain.invoke({"text": text})
        return self.format_chunks(response)

    def format_chunks(self, text: str) -> List[str]:
        pattern = r">>>\s*([\s\S]*?)\s*<<<"
        chunks = re.findall(pattern, text, re.DOTALL)
        formatted_chunks = [chunk.strip() for chunk in chunks]
        return formatted_chunks

    def num_tokens_from_string(self, string: str) -> int:
        encoding = tiktoken.get_encoding(self.encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens