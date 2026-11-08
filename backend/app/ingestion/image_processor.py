import base64
from functools import lru_cache
from pathlib import Path

import httpx
from PIL import Image
from sentence_transformers import SentenceTransformer

from app.core.config import settings

CAPTION_PROMPT = (
    "Describe this figure from a technical document in detail. "
    "Mention the text, labels, boxes and arrows you can see."
)


@lru_cache(maxsize=1)
def get_clip_model() -> SentenceTransformer:
    return SentenceTransformer("clip-ViT-B-32")


def embed_images(paths: list[str]) -> list[list[float]]:
    images = [Image.open(p).convert("RGB") for p in paths]
    return get_clip_model().encode(images, normalize_embeddings=True).tolist()

# fontion qui prend une chaîne de caractères en entrée et retourne un vecteur d'intégration en utilisant le modèle CLIP mis en cache. La fonction utilise la méthode encode du modèle pour générer le vecteur d'intégration, puis normalise les intégrations avant de les convertir en liste de flottants.
def embed_text_clip(query: str) -> list[float]:
    return get_clip_model().encode([query], normalize_embeddings=True)[0].tolist()

# Fonction qui prend le chemin d'une image en entrée et retourne une description textuelle de celle-ci en utilisant le modèle de vision par ordinateur Ollama
def describe_image(path: str) -> str:
    image_b64 = base64.b64encode(Path(path).read_bytes()).decode()
    response = httpx.post(
        f"{settings.ollama_base_url}/api/generate",
        json={
            "model": settings.vision_model,
            "prompt": CAPTION_PROMPT,
            "images": [image_b64],
            "stream": False,
        },
        timeout=300,
    )
    response.raise_for_status()
    return response.json()["response"].strip()

# Alors là , la fonction process_images prend une liste d'images (sous forme de dictionnaires contenant des informations sur chaque image) et effectue plusieurs opérations sur chacune d'elles. Elle commence par vérifier si la liste d'images est vide, auquel cas elle retourne simplement la liste vide. 
# Ensuite, elle utilise la fonction embed_images pour générer des vecteurs d'intégration pour chaque image en utilisant le modèle CLIP mis en cache. Pour chaque image, elle ajoute le vecteur d'intégration au dictionnaire correspondant sous la clé "clip_embedding". Ensuite, elle tente de générer une description textuelle de l'image en appelant la fonction describe_image.
# Si la description est réussie, elle est ajoutée au dictionnaire de l'image sous la clé "caption". 
# Si une erreur se produit lors de la génération de la description, un message d'erreur est affiché et la valeur de "caption" est définie sur None. Enfin, un message est imprimé pour chaque image indiquant le numéro de page et les 80 premiers caractères de la description (ou une chaîne vide si aucune description n'a été générée). La fonction retourne finalement la liste mise à jour des images avec leurs vecteurs d'intégration et descriptions.
def process_images(images: list[dict]) -> list[dict]:
    if not images:
        return images

    vectors = embed_images([img["file_path"] for img in images])
    for img, vector in zip(images, vectors):
        img["clip_embedding"] = vector
        try:
            img["caption"] = describe_image(img["file_path"])
        except Exception as error:
            print(f"  Description impossible (page {img['page']}): {error}")
            img["caption"] = None
        print(f"  page {img['page']} : {(img['caption'] or '')[:80]}")
    return images
