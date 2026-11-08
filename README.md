# MaintiDoc_RD

# Pipeline du pdf :
              DOCUMENT PDF
                   |
          +--------+--------+
          |                 |
     Extraction texte   Extraction images
          |                 |
       Découpage          Images
       en chunks            |
          |             Prétraitement
          |                 |
     Embeddings          CLIP Image
     textuels               |
          |            Vecteurs (512)
          |                 |
          v                 v
       PGVECTOR          PGVECTOR
          |                 |
          +--------+--------+
                   |
             Recherche
             multimodale
                   |
              MAINTIDOC



# Le lien entre figures.py et parser.py
parse_pdf()  (parser.py)
 └─ pour chaque page :
     ├─ texte           → Document (type "text")
     ├─ tableaux        → Document (type "table")     ┐
     ├─ images raster   → fichier PNG/JPEG + infos    │ trouve les images EMBARQUÉES
     └─ extract_vector_figures()  (figures.py)         │ trouve les schémas DESSINÉS
          reçoit : la page, son numéro, le dossier,    │
                   les positions des tableaux,         │
                   les métadonnées de base ────────────┘
          renvoie : une liste de dicts (même format que les images raster)