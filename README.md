# Assistant documentaire RAG

Application Gradio qui permet d'importer des PDF, de retrouver les passages
pertinents avec LlamaIndex et FAISS, puis d'obtenir une réponse générée avec
LangChain.

## Lancer le projet

1. Créer un environnement Python.
2. Installer FAISS et les dépendances.
3. Copier `.env.example` vers `.env` et renseigner `OPENAI_API_KEY`.
4. Lancer `python app.py`.