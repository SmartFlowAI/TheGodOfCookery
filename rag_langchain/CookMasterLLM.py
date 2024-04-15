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
        response, history = self.model.chat(self.tokenizer, prompt, history=[],
                                            meta_instruction="您是一个厨师，熟悉很多菜的制作方法。用户会问你哪些菜怎么制作，您可以用自己的专业知识答复他。回答的内容一般包含两块：这道菜需要哪些食材，这道菜具体是怎么做出来的。如果用户没有问菜谱相关的问题，就提醒他对菜谱的相关问题进行提问。")
        return response

    # 这个函数用于重载langchain的流式输出，实现难度太大，已放弃
    def _stream(self,
                prompt: str,
                stop: List[str] = None,
                run_manager: CallbackManagerForLLMRun = None,
                **kwargs: Any) -> Iterator[GenerationChunk]:
        event_obj = self.get_custom_event_object(**kwargs)
        text_callback = partial(event_obj.on_llm_new_token)
        index = 0

        for i, (resp, _) in enumerate(self.model.stream_chat(self.tokenizer, prompt, history=[],
                                                             meta_instruction="您是一个厨师，熟悉很多菜的制作方法。用户会问你哪些菜怎么制作，您可以用自己的专业知识答复他。回答的内容一般包含两块：这道菜需要哪些食材，这道菜具体是怎么做出来的。如果用户没有问菜谱相关的问题，就提醒他对菜谱的相关问题进行提问。")):
            text_callback(resp[index:])
            generation = GenerationChunk(text=resp[index:])
            index = len(resp)
            yield generation

    # 可以使用这个函数直接在命令行中进行流式对话
    def stream_chat(self, query):
        index = 0
        for i, (resp, _) in enumerate(
                self.model.stream_chat(self.tokenizer, query, history=[])):
            print(resp[index:], end='')
            index = len(resp)

    @property
    def _llm_type(self) -> str:
        return "InternLM2"
