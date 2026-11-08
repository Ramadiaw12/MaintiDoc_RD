# Le chunker doit transformer chaque description en chunk de type image, qui garde le chemin de l’image en métadonnée
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from app.ingestion.parser import ParsedPDF

HIDDEN_FROM_EMBEDDING = ["source", "page", "type", "document_id", "table_index", "file_path"]

# Ici nous avons la fonction build_nodes qui prend en entrée un objet ParsedPDF et un identifiant de document.
# Elle utilise un SentenceSplitter pour découper les documents textuels en nœuds de texte, puis ajoute les documents tabulaires et les images avec leurs métadonnées correspondantes.
def build_nodes(parsed: ParsedPDF, document_id: int) -> list[TextNode]:
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    nodes = splitter.get_nodes_from_documents(parsed.text_docs)

    # Alors là, la boucles for table in parsed.table_docs: parcourt chaque document tabulaire extrait du PDF et crée un nœud de texte pour chaque tableau, en utilisant le texte du tableau et les métadonnées correspondantes. Les nœuds de texte sont ensuite ajoutés à la liste nodes
    for table in parsed.table_docs:
        nodes.append(TextNode(text=table.text, metadata=dict(table.metadata)))

    for img in parsed.images:
        if img.get("caption"):
            nodes.append(TextNode(
                text=img["caption"],
                metadata={k: img[k] for k in ("source", "page", "type", "file_path")},
            ))

    for node in nodes:
        node.metadata["document_id"] = document_id
        node.excluded_embed_metadata_keys = HIDDEN_FROM_EMBEDDING
    return nodes
