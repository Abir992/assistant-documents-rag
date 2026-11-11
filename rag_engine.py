import os

import faiss
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

load_dotenv()

# Ce modèle multilingue produit des vecteurs de 384 dimensions.
EMBEDDING_DIM = 384
_embedding_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

_index = None

_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Réponds en français uniquement à partir des extraits fournis. "
        "Si les documents ne permettent pas de répondre, dis-le clairement. "
        "N'invente aucune information."
    ),
    (
        "human",
        "Extraits des documents :\n{context}\n\nQuestion : {question}"
    ),
])


def index_documents(file_paths: list[str]) -> str:
    """Lit les PDF fournis et crée un index FAISS."""
    global _index

    if not file_paths:
        raise ValueError("Ajoute au moins un fichier PDF.")

    documents = SimpleDirectoryReader(input_files=file_paths).load_data()

    if not documents:
        raise ValueError("Aucun texte lisible n'a été trouvé dans ces PDF.")

    faiss_index = faiss.IndexFlatL2(EMBEDDING_DIM)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    _index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=_embedding_model,
    )

    return f"{len(file_paths)} PDF indexé(s), contenant {len(documents)} page(s)."


def answer_question(question: str) -> tuple[str, str]:
    """Retrouve les passages utiles et demande une réponse au modèle."""
    if _index is None:
        raise RuntimeError("Importe et indexe d'abord au moins un PDF.")

    if not question or not question.strip():
        raise ValueError("Écris une question.")

    retrieved_nodes = _index.as_retriever(similarity_top_k=4).retrieve(question)

    if not retrieved_nodes:
        return "Je n'ai pas trouvé de passage pertinent.", ""

    context_parts = []
    source_names = []

    for item in retrieved_nodes:
        node = item.node
        metadata = node.metadata or {}
        filename = metadata.get("file_name", "Document")
        page = metadata.get("page_label", "?")

        context_parts.append(
            f"[Source : {filename}, page {page}]\n{node.get_content()}"
        )
        source_names.append(f"- {filename}, page {page}")

    context = "\n\n".join(context_parts)

    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    )
    response = (_prompt | llm).invoke({
        "context": context,
        "question": question,
    })

    unique_sources = list(dict.fromkeys(source_names))
    return str(response.content), "\n".join(unique_sources)