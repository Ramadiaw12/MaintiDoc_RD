# Chunker découpe les documents textuels et tabulaires extraits d'un PDF en nœuds de texte, en utilisant un séparateur de phrases pour créer des nœuds de taille appropriée. 
# Il ajoute également des métadonnées aux nœuds, y compris l'identifiant du document et les clés de métadonnées à exclure de l'intégration.
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from app.ingestion.parser import ParsedPDF

# Les clés de métadonnées à exclure de l'intégration sont définies dans la liste HIDDEN_FROM_EMBEDDING, qui comprend les clés "source", "page", "type", "document_id" et "table_index". Ces clés sont utilisées pour stocker des informations sur l'origine du document, le numéro de page, le type de contenu (texte ou tableau), l'identifiant du document et l'index du tableau, mais elles ne sont pas pertinentes pour l'intégration des nœuds de texte.
HIDDEN_FROM_EMBEDDING = ["source", "page", "type", "document_id", "table_index"]

# La fonction build_nodes prend un objet ParsedPDF et un identifiant de document, et retourne une liste de nœuds de texte. 
# Elle utilise un séparateur de phrases pour découper les documents textuels en nœuds, ajoute les documents tabulaires en tant que nœuds de texte, et ajoute des métadonnées aux nœuds, y compris l'identifiant du document et les clés de métadonnées à exclure de l'intégration.
def build_nodes(parsed: ParsedPDF, document_id: int) -> list[TextNode]:
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    nodes = splitter.get_nodes_from_documents(parsed.text_docs)

    for table in parsed.table_docs:
        nodes.append(TextNode(text=table.text, metadata=dict(table.metadata)))

    for node in nodes:
        node.metadata["document_id"] = document_id
        node.excluded_embed_metadata_keys = HIDDEN_FROM_EMBEDDING

    return nodes