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


## ⚠️ Limites techniques & axes d'amélioration

> **Engineering transparency:** MaintiDoc est un projet en évolution. Cette section documente les limites identifiées lors des premiers tests, les compromis techniques actuels et les améliorations à envisager.

### 1. 🖥️ Ressources matérielles & performances

L'exécution locale de plusieurs composants IA sollicite fortement les ressources disponibles.

| Indicateur | État observé |
|---|---|
| RAM physique | 15 Go |
| RAM utilisée | Environ 13 Go |
| Swap | Saturé lors de la dernière mesure |
| Inférence | CPU uniquement |
| Temps de réponse de l'agent | 62 secondes sous saturation |

**Problèmes identifiés :**
- La pression mémoire ralentit l'ensemble du pipeline.
- Plusieurs scripts Python peuvent charger leurs propres instances de PyTorch (environ 1 Go chacune selon les observations).
- Les timeouts de 5 à 6 minutes sont principalement liés aux contraintes matérielles.

**Pistes d'amélioration :**
- Optimiser le chargement et la réutilisation des modèles.
- Réduire les composants actifs simultanément.
- Explorer la quantification des modèles.
- Envisager une infrastructure disposant de davantage de mémoire ou d'un GPU.

<details>
<summary>📊 Performance cible envisagée</summary>

Une réponse entre **15 et 35 secondes** sur CPU constitue une estimation de travail, et non une performance déjà validée.

</details>

---

### 2. 🧠 Modèle de langage — Qwen 3B

Le modèle local Qwen 3B présente certaines limites en matière de fiabilité et de génération.

**Comportements observés :**
- Génération occasionnelle de citations inexistantes (ex. : `[6]`).
- Incohérence linguistique entre la question et la réponse.
- Sélection d'outils parfois imprévisible en mode agent.
- Capacité limitée à exploiter des contextes documentaires volumineux.

**Configuration actuelle :**

| Paramètre | Valeur |
|---|---|
| Modèle | Qwen 3B |
| Contexte | 3 072 tokens |
| `top_k` | 3 |
| Inférence | Ollama (local) |

**Pistes d'amélioration :**
- Renforcer la validation des citations côté backend.
- Contraindre le modèle à répondre exclusivement à partir du contexte récupéré.
- Améliorer la gestion de la langue.
- Étudier des modèles plus performants selon les ressources disponibles.

---

### 3. 🔎 Recherche vectorielle & embeddings

La qualité des réponses dépend directement de la pertinence des passages récupérés.

**Limite identifiée :**

Le modèle d'embedding actuel, `bge-small-en`, est principalement destiné à l'anglais. Il n'est donc pas optimal pour interroger des manuels techniques rédigés en français.

| Élément | Configuration actuelle |
|---|---|
| Embeddings | BGE-small-en |
| Dimensions | 384 |
| Stockage vectoriel | PostgreSQL + pgvector |
| Scores observés | 0.61 – 0.78 |
| Filtrage | `MIN_SCORE` |

**Pistes d'amélioration :**
- Migrer vers un modèle d'embedding multilingue.
- Calibrer `MIN_SCORE` sur un jeu de questions et de passages pertinents.
- Évaluer la qualité du classement des résultats (`top_k`).
- Tester une stratégie de recherche hybride (vectorielle + lexicale).

---

### 4. 📄 Extraction & parsing documentaire

Le pipeline d'ingestion a été testé sur un PDF numérique relativement propre. La robustesse sur des documents industriels complexes reste à évaluer.

**Cas non encore validés :**

- PDF scannés nécessitant une reconnaissance OCR.
- Documents avec plusieurs colonnes.
- Tableaux répartis sur plusieurs pages.
- En-têtes et pieds de page répétés.
- Structures documentaires complexes.

**Point particulier :**

Le module `figures.py`, destiné à l'extraction des schémas vectoriels, est intégré au pipeline, mais n'a pas encore permis de valider l'extraction de figures sur le document testé.

**Pistes d'amélioration :**
- Intégrer un pipeline OCR.
- Améliorer la détection des tableaux et leur continuité entre les pages.
- Tester le parsing sur de véritables manuels de maintenance.
- Valider l'extraction des schémas techniques vectoriels.

---

### 5. 🖼️ Compréhension multimodale

La compréhension des images et des schémas techniques représente un défi supplémentaire.

| Composant | Limite identifiée |
|---|---|
| Moondream | Descriptions parfois imprécises sur les schémas denses |
| CLIP | Difficulté à distinguer des schémas visuellement similaires |
| Affichage des images | Dépend actuellement de chemins de fichiers locaux |

**Pistes d'amélioration :**
- Tester des modèles de vision plus adaptés aux documents techniques.
- Améliorer la représentation des images et des schémas.
- Stocker les fichiers dans un emplacement accessible au backend et au frontend.
- Utiliser des références persistantes plutôt que des chemins locaux.

---

### 6. 🧪 Évaluation & fiabilité des réponses

La validation du système doit aller au-delà de quelques démonstrations fonctionnelles.

**Limite du jeu de test actuel :**

Le document *Attention Is All You Need* est largement connu des modèles de langage. Qwen peut donc restituer des connaissances préalables sans nécessairement exploiter les passages récupérés.

**Pour une évaluation plus représentative :**
- Utiliser des manuels techniques inconnus du modèle.
- Construire un jeu de questions-réponses avec des réponses de référence.
- Vérifier la fidélité des réponses aux sources.
- Mesurer la précision de la recherche documentaire.
- Tester les cas où aucune réponse pertinente n'existe.

**État actuel :** les validations sont principalement manuelles. Une suite de tests automatisés avec `pytest` reste à mettre en place.

---

### 7. ⚙️ Architecture logicielle & exploitation

Certaines fonctionnalités nécessaires à une utilisation en production ne sont pas encore implémentées.

| Fonctionnalité | État actuel |
|---|---|
| Indexation documentaire | Synchrone |
| Streaming des réponses | Non implémenté |
| Authentification | Non implémentée |
| Gestion multi-utilisateur | Non implémentée |
| Tests automatisés | À développer |
| Déploiement distant des images | À améliorer |

**Évolutions envisagées :**
- Mettre en place une indexation asynchrone.
- Ajouter le streaming des réponses.
- Introduire une authentification et une gestion des utilisateurs.
- Développer des tests unitaires et d'intégration.
- Préparer une architecture adaptée au déploiement distant.

---

### 8. 🔐 Sécurité & gestion des secrets

La protection des informations sensibles est indispensable, même pendant le développement.

**Actions de sécurité :**
- Régénérer toute clé API précédemment exposée.
- Exclure `.env` du dépôt Git.
- Utiliser des variables d'environnement pour les secrets.
- Ajouter un fichier `.env.example` sans aucune valeur sensible.
- Éviter d'exposer les informations de connexion dans les logs.

---

### 🛣️ Roadmap technique

| Priorité | Amélioration | Objectif |
|:---:|---|---|
| 🔴 Haute | Embeddings multilingues | Améliorer la recherche en français |
| 🔴 Haute | Validation des citations | Renforcer la fiabilité des réponses |
| 🔴 Haute | Optimisation mémoire | Réduire les ralentissements |
| 🟠 Moyenne | OCR et parsing avancé | Prendre en charge les PDF complexes |
| 🟠 Moyenne | Évaluation automatisée | Mesurer objectivement les performances |
| 🟠 Moyenne | Vision avancée | Mieux comprendre les schémas techniques |
| 🟢 Future | Indexation asynchrone | Améliorer l'expérience utilisateur |
| 🟢 Future | Authentification | Préparer le multi-utilisateur |

---

> **Notre approche :** privilégier la fiabilité documentaire, la traçabilité des réponses et la maîtrise des ressources avant d'élargir les capacités de l'assistant.