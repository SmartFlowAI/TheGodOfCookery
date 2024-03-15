from typing import Any, List

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain.retrievers.document_compressors.base import (
    BaseDocumentCompressor,
)

# copy from langchain.retrievers.ContextualCompressionRetriever
# 可以替换假设问题为原始菜谱的Retriever
class HyQEContextualCompressionRetriever(BaseRetriever):
    """Retriever that wraps a base retriever and compresses the results."""

    base_compressor: BaseDocumentCompressor
    """Compressor for compressing retrieved documents."""

    base_retriever: BaseRetriever
    """Base Retriever to use for getting relevant documents."""

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            Sequence of relevant documents
        """
        docs = self.base_retriever.get_relevant_documents(
            query, callbacks=run_manager.get_child(), **kwargs
        )
        if docs:
            for i in range(len(docs)):
                # 取出metadata里存放的菜谱，并用菜谱替换掉page_content里保存的假设问题
                if "caipu" in docs[i].metadata:
                    docs[i].page_content = docs[i].metadata["caipu"]
                    docs[i].metadata = {}
            compressed_docs = self.base_compressor.compress_documents(
                docs, query, callbacks=run_manager.get_child()
            )
            return list(compressed_docs)
        else:
            return []

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> List[Document]:
        """Get documents relevant for a query.

        Args:
            query: string to find relevant documents for

        Returns:
            List of relevant documents
        """
        docs = await self.base_retriever.aget_relevant_documents(
            query, callbacks=run_manager.get_child(), **kwargs
        )
        if docs:
            for i in range(len(docs)):
                # 取出metadata里存放的菜谱，并用菜谱替换掉page_content里保存的假设问题
                if "caipu" in docs[i].metadata:
                    docs[i].page_content = docs[i].metadata["caipu"]
                    docs[i].metadata = {}
            compressed_docs = await self.base_compressor.acompress_documents(
                docs, query, callbacks=run_manager.get_child()
            )
            return list(compressed_docs)
        else:
            return []
