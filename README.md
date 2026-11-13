# Assistant documentaire RAG

Application web en français permettant d’interroger le contenu de documents PDF. Elle combine une recherche sémantique avec génération augmentée par récupération (RAG) afin de produire des réponses fondées sur les passages retrouvés dans les documents.

## Fonctionnalités

- Import de plusieurs fichiers PDF depuis une interface Gradio.
- Extraction du texte des documents avec LlamaIndex.
- Création d’embeddings multilingues en local avec Hugging Face.
- Recherche des passages pertinents avec FAISS.
- Génération de réponses en français avec LangChain et l’API OpenAI.
- Affichage des sources retrouvées, notamment du nom du document et de la page lorsque ces informations sont disponibles.
- Instruction au modèle de signaler lorsqu’il ne trouve pas la réponse dans les extraits fournis.

## Architecture

1. **Gradio** reçoit les PDF et les questions de l’utilisateur.
2. **LlamaIndex** charge le texte des documents et construit l’index de recherche.
3. **Hugging Face** génère localement les représentations vectorielles du texte.
4. **FAISS** retrouve les extraits les plus proches de la question.
5. **LangChain** prépare le prompt et transmet la question ainsi que les extraits retenus au modèle OpenAI.
6. L’application affiche la réponse et les sources retrouvées.

## Technologies

- Python 3.11
- Gradio
- LlamaIndex
- LangChain
- FAISS
- Hugging Face Sentence Transformers
- OpenAI API

## Installation

### Prérequis

- Python 3.11
- Conda
- Git
- Une clé API OpenAI

### 1. Cloner le dépôt

```powershell
git clone https://github.com/Abir992/assistant-documents-rag.git
cd assistant-documents-rag
