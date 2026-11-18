# MaintiDoc_RD

# **Partie I: RAG Agentique Multi-MOdal**
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



## 🏗️ Architecture technique de la partie :

Cette première partie de MaintiDoc repose sur une architecture **RAG Agentique multimodale**, combinant l'extraction documentaire, la recherche vectorielle et un modèle de langage local pour fournir des réponses contextualisées, vérifiables et accompagnées de leurs sources.

### 1. Pipeline d'ingestion documentaire

Les documents PDF sont traités afin d'en extraire les informations exploitables par le système :

```text
PDF
 │
 ▼
PyMuPDF — Extraction documentaire
 │
 ├── Texte
 ├── Tableaux
 └── Images → descriptions textuelles
 │
 ▼
Segmentation (Chunking)
 │
 ▼
BGE-small — Génération des embeddings
(384 dimensions)
 │
 ▼
PostgreSQL + pgvector
(Docker · Port 5433)
 │
 ▼
Index vectoriel
```

### 2. Pipeline de recherche et de génération

Lorsqu'un technicien pose une question, MaintiDoc recherche les passages documentaires les plus pertinents avant de générer une réponse.

```text
Question utilisateur
        │
        ▼
Génération de l'embedding
        │
        ▼
Recherche vectorielle
(PostgreSQL + pgvector)
        │
        ▼
Récupération des passages pertinents
        │
        ▼
Filtrage par seuil de pertinence
        │
        ▼
Qwen — Ollama (LLM local)
        │
        ├── Contexte documentaire suffisant
        │        │
        │        ▼
        │   Réponse contextualisée
        │   avec citations [n]
        │
        └── Contexte insuffisant
                 │
                 ▼
             Refus de répondre
             sans preuve documentaire
        │
        ▼
Réponse + sources associées
```

### 3. Architecture applicative

Le système est organisé autour de deux composants principaux :

| Composant | Technologie | Responsabilité |
|---|---|---|
| Backend | FastAPI | API, orchestration RAG et logique métier |
| Frontend | Streamlit | Interface conversationnelle et gestion documentaire |
| Base de données | PostgreSQL + pgvector | Stockage documentaire et recherche vectorielle |
| Embeddings | BGE-small | Transformation du texte en vecteurs de 384 dimensions |
| LLM | Qwen via Ollama | Génération des réponses |
| Conteneurisation | Docker | Gestion des services et de l'environnement |

### 4. Mode agent et stratégie de repli

MaintiDoc intègre également un **mode agent basé sur LangChain**, conçu pour enrichir les capacités d'interaction avec les composants du système.

```text
              Question utilisateur
                       │
                       ▼
                FastAPI Backend
                       │
                       ▼
                Mode de traitement
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       Mode Agent            RAG classique
       (LangChain)           (Fallback)
             │                   │
             └─────────┬─────────┘
                       ▼
              Réponse contextualisée
                       │
                       ▼
                Sources vérifiables
                       │
                       ▼
                  Streamlit
```


---

**Principe fondamental :** MaintiDoc privilégie les réponses fondées sur les documents techniques indexés. Lorsqu'aucune information suffisamment pertinente n'est retrouvée, le système doit éviter de produire une réponse non vérifiée.