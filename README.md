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