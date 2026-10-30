from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from app.ingestion.parser import ParsedPDF

HIDDEN_FROM_EMBEDDING = ["source", "page", "type", "document_id", "table_index"]


def build_nodes(parsed: ParsedPDF, document_id: int) -> list[TextNode]:
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    nodes = splitter.get_nodes_from_documents(parsed.text_docs)

    for table in parsed.table_docs:
        nodes.append(TextNode(text=table.text, metadata=dict(table.metadata)))

    for node in nodes:
        node.metadata["document_id"] = document_id
        node.excluded_embed_metadata_keys = HIDDEN_FROM_EMBEDDING

    return nodes