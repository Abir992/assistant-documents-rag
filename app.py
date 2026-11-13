import gradio as gr

from rag_engine import answer_question, index_documents


def index_action(files):
    if not files:
        return "Ajoute au moins un PDF."

    try:
        return index_documents(files)
    except Exception as error:
        return f"Erreur pendant l’indexation : {error}"


def question_action(question):
    print(f"[Gradio] Clic reçu. Question : {question!r}", flush=True)
    try:
        answer, sources = answer_question(question)
        print("[Gradio] Réponse reçue de answer_question.", flush=True)
        return answer, sources
    except Exception as error:
        print(f"[Gradio] Erreur : {error!r}", flush=True)
        return f"Erreur : {error}", ""
    

with gr.Blocks(title="Assistant documentaire RAG") as demo:
    gr.Markdown("# Assistant documentaire")
    gr.Markdown("Importe des PDF, puis pose une question sur leur contenu.")

    pdf_files = gr.File(
        label="Choisir des PDF",
        file_count="multiple",
        file_types=[".pdf"],
        type="filepath",
    )
    index_button = gr.Button("Indexer les documents")
    status = gr.Textbox(label="État de l’indexation", interactive=False)

    index_button.click(
        fn=index_action,
        inputs=pdf_files,
        outputs=status,
    )

    question = gr.Textbox(
        label="Ta question",
        placeholder="Que disent les documents sur… ?",
    )
    ask_button = gr.Button("Poser la question")
    answer = gr.Markdown(label="Réponse")
    sources = gr.Markdown(label="Sources retrouvées")

    ask_button.click(
        fn=question_action,
        inputs=question,
        outputs=[answer, sources],
    )

demo.launch()