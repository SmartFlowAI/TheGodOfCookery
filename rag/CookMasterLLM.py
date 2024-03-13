from functools import partial
from langchain.llms.base import LLM
from typing import Any, List, Optional, Iterator
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import GenerationChunk
from modelscope import AutoModelForCausalLM, AutoTokenizer


class CookMasterLLM(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None

    def __init__(self, model, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs: Any):
        # 重写调用函数
        response, history = self.model.chat(self.tokenizer, prompt, history=[])
        return response

    def _stream(self,
                prompt: str,
                stop: List[str] = None,
                run_manager: CallbackManagerForLLMRun = None,
                **kwargs: Any) -> Iterator[GenerationChunk]:
        event_obj = self.get_custom_event_object(**kwargs)
        text_callback = partial(event_obj.on_llm_new_token)
        index = 0

        for i, (resp, _) in enumerate(self.model.stream_chat(self.tokenizer, prompt, history=[])):
            text_callback(resp[index:])
            generation = GenerationChunk(text=resp[index:])
            index = len(resp)
            yield generation

    def stream_chat(self, query):
        index = 0
        for i, (resp, _) in enumerate(
                self.model.stream_chat(self.tokenizer, query, history=[])):
            print(resp[index:], end='')
            index = len(resp)

    @property
    def _llm_type(self) -> str:
        return "InternLM2"
