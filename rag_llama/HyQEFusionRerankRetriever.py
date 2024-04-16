from typing import List
from BCEmbedding.models import RerankerModel
from llama_index.core import QueryBundle
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore


class HyQEFusionRerankRetriever(BaseRetriever):
    def __init__(self,
                 retrievers: List[BaseRetriever],
                 bce_reranker_config: dict,
                 verbose: bool = False,
                 ):
        super().__init__(verbose=verbose)
        self.retrievers = retrievers
        self.reranker = RerankerModel(model_name_or_path=bce_reranker_config["model"],
                                      use_fp16=bce_reranker_config["use_fp16"],
                                      device=bce_reranker_config["device"])
        self.top_n = bce_reranker_config["top_n"]
        self.verbose = verbose

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        # 取出query_bundle中的待检索字符串query_str
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        query_str = query_bundle.query_str

        # 依次调用各个子检索器的retrieve方法，将检索结果合并
        nodes = []
        for retriever in self.retrievers:
            nodes.extend(retriever.retrieve(query_str))
        if len(nodes) == 0:
            return []

        # 从检索结果中提取出待重排序的passages
        passages = []
        valid_nodes = []
        invalid_nodes = []
        for node in nodes:
            passage = node.node.get_content()
            if isinstance(passage, str) and len(passage) > 0:
                passages.append(passage.replace('\n', ' '))
                valid_nodes.append(node)
            else:
                invalid_nodes.append(node)

        # 调用reranker对passages进行重排序
        rerank_result = self.reranker.rerank(query_str, passages)

        # 根据rerank_result对nodes进行重排序
        new_nodes = []
        for score, nid in zip(rerank_result['rerank_scores'], rerank_result['rerank_ids']):
            node = valid_nodes[nid]
            node.score = score
            new_nodes.append(node)
        for node in invalid_nodes:
            node.score = 0
            new_nodes.append(node)

        assert len(new_nodes) == len(nodes)

        # 取出前top_n个
        new_nodes = new_nodes[:self.top_n]

        # 取出metadata中的caipu字段，放入node的text中
        for i in range(len(new_nodes)):
            if "caipu" in new_nodes[i].node.metadata:
                new_nodes[i].node.text = new_nodes[i].node.metadata["caipu"]
                new_nodes[i].node.metadata = {}

        return new_nodes
