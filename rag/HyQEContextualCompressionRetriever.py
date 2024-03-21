# copy from langchain.retrievers.ContextualCompressionRetriever

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
            compressed_docs = self.base_compressor.compress_documents(
                docs, query, callbacks=run_manager.get_child()
            )
            compressed_docs = list(compressed_docs)
            for i in range(len(compressed_docs)):
                if "caipu" in compressed_docs[i].metadata:
                    # 取出metadata中的caipu字段，放入page_content中
                    compressed_docs[i].page_content = compressed_docs[i].metadata["caipu"]
                    compressed_docs[i].metadata = {}
            return compressed_docs
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
            compressed_docs = await self.base_compressor.acompress_documents(
                docs, query, callbacks=run_manager.get_child()
            )
            compressed_docs = list(compressed_docs)
            for i in range(len(compressed_docs)):
                if "caipu" in compressed_docs[i].metadata:
                    # 取出metadata中的caipu字段，放入page_content中
                    compressed_docs[i].page_content = compressed_docs[i].metadata["caipu"]
                    compressed_docs[i].metadata = {}
            return compressed_docs
        else:
            return []
