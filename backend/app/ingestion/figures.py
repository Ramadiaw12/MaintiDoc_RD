# File pour extraire les figures vectorielles d'un PDF et les enregistrer en tant qu'images PNG.
from pathlib import Path

import pymupdf

MIN_FIGURE_SIZE = 100  # points : on ignore les petits traits décoratifs
PADDING = 8            # marge ajoutée autour de la figure (pour les légendes)


# Alors nous avons la fonction extract_vector_figures qui prend en entrée une page d'un PDF, le numéro de la page, un répertoire pour stocker les images extraites, une liste de rectangles représentant les tableaux sur la page, et un dictionnaire de métadonnées de base 
# La fonction parcourt les dessins regroupés sur la page, filtre ceux qui sont trop petits ou qui occupent presque toute la page, ou qui chevauchent significativement un tableau. Pour chaque dessin valide, elle crée un clip avec une marge autour du dessin, génère une image à partir de ce clip, l'enregistre dans le répertoire spécifié avec un nom de fichier unique basé sur le numéro de page et l'index du dessin, et ajoute les métadonnées correspondantes à la liste des figures extraites 
# La fonction retourne finalement cette liste de figures.
def extract_vector_figures(page, page_number, images_dir, table_rects, base_meta):
    figures = []
    page_area = page.rect.get_area()

    # Parcourt chaque dessin regroupé sur la page, en commençant par l'index 1. Pour chaque dessin, il vérifie si sa largeur ou sa hauteur est inférieure à MIN_FIGURE_SIZE, s'il occupe plus de 90 % de la surface de la page, ou s'il chevauche significativement un tableau (plus de 50 % de sa surface). Si l'une de ces conditions est remplie, le dessin est ignoré. Sinon, un clip est créé autour du dessin avec une marge définie par PADDING, et une image est générée à partir de ce clip avec une résolution de 150 dpi
    for index, rect in enumerate(page.cluster_drawings(), start=1):
        if rect.width < MIN_FIGURE_SIZE or rect.height < MIN_FIGURE_SIZE:
            continue
        if rect.get_area() > 0.9 * page_area:           # cadre de page entière
            continue
        if any((rect & t).get_area() > 0.5 * rect.get_area() for t in table_rects):
            continue                                    # ce sont les traits d'un tableau
        # Crée un clip avec une marge autour du dessin, génère une image à partir de ce clip, l'enregistre dans le répertoire spécifié avec un nom de fichier unique basé sur le numéro de page et l'index du dessin, et ajoute les métadonnées correspondantes à la liste des figures extraites    
        clip = pymupdf.Rect(
            rect.x0 - PADDING, rect.y0 - PADDING, rect.x1 + PADDING, rect.y1 + PADDING
        ) & page.rect
        pix = page.get_pixmap(clip=clip, dpi=150)

        # Crée un nom de fichier unique pour l'image en utilisant le numéro de page et l'index du dessin, et en ajoutant l'extension appropriée. Le nom de fichier est ensuite utilisé pour créer un chemin complet vers le répertoire des images, et l'image est enregistrée sur le disque à cet emplacement
        path = Path(images_dir) / f"p{page_number}_fig{index}.png"
        pix.save(path)
        figures.append(
            {**base_meta, "type": "image", "file_path": str(path),
             "width": pix.width, "height": pix.height}
        )
    return figures