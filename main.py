import requests
import faiss
import numpy as np
import PyPDF2

from pydantic import BaseModel
from ollama import chat

###############################################################################
# 1. CONFIG & GLOBALS
###############################################################################
OLLAMA_URL = "http://localhost:11434/api"
GM_MODEL = "llama3.2"  # Game Master's model
WATSON_MODEL = "tinyllama"  # Watson's model
PDF_PATH = "RAG/cano.pdf"

CHUNK_SIZE = 500
OVERLAP_SIZE = 50


###############################################################################
# 2. SCHEMA DEFINITIONS
###############################################################################
class GMGameState(BaseModel):
    # The “master” representation of the game. GM knows everything (including culprit).
    narration: str
    real_culprit: str
    suspected_murderer: str | None
    discovered_clues: list[str]
    interviewed_suspects: list[str]
    finished: bool


class WatsonMessage(BaseModel):
    # Watson only returns the text of his replies (or any small extras).
    # He does NOT know the real murderer. This is just an example schema.
    watson_reply: str


###############################################################################
# 3. PDF -> TEXT -> CHUNKS (for RAG)
###############################################################################
def pdf_to_chunks(pdf_path, chunk_size=CHUNK_SIZE, overlap=OVERLAP_SIZE):
    text_chunks = []
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        num_pages = len(reader.pages)
        for page_idx in range(num_pages):
            page_text = reader.pages[page_idx].extract_text()
            if not page_text:
                continue
            page_text = page_text.strip().replace('\n', ' ')
            start_idx = 0
            while start_idx < len(page_text):
                end_idx = start_idx + chunk_size
                chunk = page_text[start_idx:end_idx]
                text_chunks.append(chunk)
                start_idx += (chunk_size - overlap)
    return text_chunks


###############################################################################
# 4. EMBEDDING & FAISS INDEX
###############################################################################
def get_embeddings(texts, model_name):
    embeddings = []
    for t in texts:
        payload = {"text": t, "model": model_name}
        resp = requests.post(f"{OLLAMA_URL}/embed", json=payload)
        resp.raise_for_status()
        data = resp.json()
        embeddings.append(data["embeddings"])
    return embeddings


def build_faiss_index(chunks, model_name):
    chunk_embeddings = get_embeddings(chunks, model_name)
    emb_array = np.array(chunk_embeddings, dtype=np.float32)
    dimension = emb_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(emb_array)
    return index, chunks


def retrieve_top_chunks(query, index, chunk_texts, model_name, k=2):
    q_embedding = get_embeddings([query], model_name)[0]
    q_embedding = np.array(q_embedding, dtype=np.float32).reshape(1, -1)
    distances, indices = index.search(q_embedding, k)
    return [chunk_texts[i] for i in indices[0]]


###############################################################################
# 5. OLLAMA CALLS FOR EACH LLM
###############################################################################
def call_game_master_llm(game_state: GMGameState, context: str, user_query: str) -> GMGameState:
    """
    Calls the GM LLM with a prompt that returns a JSON representation of the updated GMGameState.
    """
    system_content = (
        "You are the Game Master of a Sherlock Holmes murder mystery. "
        "You know the real culprit but keep it hidden from the user. "
        "The user is Sherlock Holmes. You must maintain the game logic, clues, and keep track of progress. "
        "You will respond ONLY with valid JSON that conforms to the GMGameState schema: "
        "(narration, real_culprit, suspected_murderer, discovered_clues, interviewed_suspects, finished). "
        "Do not speak for Sherlock or Watson. Do not reveal the real murderer unless the game is finished. "
        "If the user suspects X, ensure X is not the real murderer. "
        "No text outside JSON. Maintain an immersive narrative in the 'narration' field. "
        "If 'finished' is true, the game ends."
    )

    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "assistant",
            "content": (
                f"Current GMGameState JSON:\n{game_state.json()}\n\n"
                f"Context from RAG:\n{context}"
            )
        },
        {"role": "user", "content": user_query}
    ]

    response = chat(
        messages=messages,
        model=GM_MODEL,
        format=GMGameState.model_json_schema(),  # Enforce GMGameState
        temperature=0.7,
    )

    try:
        updated_state = GMGameState.model_validate_json(response.message.content)
        return updated_state
    except Exception as e:
        print("Error parsing GM LLM response into GMGameState:", e)
        print("Response content was:", response.message.content)
        return game_state  # fallback to old state


def call_watson_llm(conversation_history: str) -> WatsonMessage:
    """
    Calls the Watson LLM, which does NOT know the murderer.
    conversation_history is everything the user and Watson said so far in this mini-session.
    Returns WatsonMessage with watson_reply.
    """
    system_content = (
        "You are Dr. John Watson, loyal companion to Sherlock Holmes. "
        "You do NOT know who the murderer is. "
        "You should be supportive and observant, offering insights but never overshadowing Sherlock. "
        "Respond ONLY in valid JSON that conforms to WatsonMessage schema: (watson_reply). "
        "No text outside JSON. Maintain an immersive style. "
    )

    messages = [
        {"role": "system", "content": system_content},
        {
            "role": "assistant",
            "content": (
                "Below is the conversation so far between you (Watson) and Sherlock:\n"
                f"{conversation_history}"
            )
        },
        {
            "role": "user",
            "content": "Please respond as Watson in valid JSON."
        },
    ]

    response = chat(
        messages=messages,
        model=WATSON_MODEL,
        format=WatsonMessage.model_json_schema(),  # Enforce WatsonMessage
        temperature=0.7,
    )

    try:
        watson_update = WatsonMessage.model_validate_json(response.message.content)
        return watson_update
    except Exception as e:
        print("Error parsing Watson LLM response into WatsonMessage:", e)
        print("Response content was:", response.message.content)
        return WatsonMessage(watson_reply="I'm sorry, something went wrong.")


###############################################################################
# 6. MAIN LOOP
###############################################################################
def main():
    # 6A. Build the knowledge base
    print("Loading PDF and building knowledge base for GM RAG indexing...")
    holmes_chunks = pdf_to_chunks(PDF_PATH)
    gm_index, gm_chunk_texts = build_faiss_index(holmes_chunks, GM_MODEL)
    print(f"Total Sherlock chunks: {len(holmes_chunks)}")

    # 6B. Initialize GMGameState.
    gm_state = GMGameState(
        narration="A dark and stormy night falls upon Baker Street...",
        real_culprit="Professor Moriarty",  # The GM keeps it hidden
        suspected_murderer=None,
        discovered_clues=[],
        interviewed_suspects=[],
        finished=False
    )

    print("\nWelcome to the Sherlock Holmes Mystery!")
    print("Type 'exit' to quit.\n")

    # 6C. Primary game loop
    while True:
        if gm_state.finished:
            print("The game has concluded. Goodbye!")
            break

        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting. Goodbye!")
            break

        # 6D. If user says "Watson!" -> enter Watson sub-loop
        if user_input.startswith("Watson!"):
            watson_convo_log = ""
            watson_convo_log += f"Sherlock (User): {user_input}\n"

            while True:
                # Call Watson LLM
                watson_resp = call_watson_llm(watson_convo_log)

                # Print Watson's response
                print("\nWatson:\n")
                print("\033[94m" + watson_resp.watson_reply + "\033[0m")

                # Get next user line
                next_input = input("You (still talking to Watson?): ")
                if next_input.lower() == "exit":
                    print("Exiting. Goodbye!")
                    return

                # If user no longer says "Watson!", we exit sub-loop
                if not next_input.startswith("Watson!"):
                    # -- Pass the entire Watson conversation to the GM now --
                    # so the GM knows what was discussed
                    # We'll append it to the RAG context in a separate call:
                    conversation_summary = (
                            "Sherlock and Watson had the following conversation:\n"
                            + watson_convo_log
                            + f"\nWatson's latest reply:\n{watson_resp.watson_reply}"
                    )
                    # We can feed this to the GM so it stays updated
                    gm_state = call_game_master_llm(
                        gm_state,
                        conversation_summary,
                        "Update the GM with Watson conversation."
                    )

                    user_input = next_input  # carry on to main GM logic
                    break

                # Otherwise, continue Watson sub-loop
                watson_convo_log += f"Sherlock (User): {next_input}\n"

            # Past the while loop, we continue below with user_input = next_input

        # 6E. If we reach here, the user is talking to the GM (normal flow):
        top_chunks = retrieve_top_chunks(user_input, gm_index, gm_chunk_texts, GM_MODEL, k=2)
        rag_context = "\n---\n".join(top_chunks)

        # Update GM state via LLM
        gm_state = call_game_master_llm(gm_state, rag_context, user_input)

        # Display GM's new narration
        print("\nGame Master:\n")
        print(gm_state.narration)
        print("\nCurrent GM State:", gm_state.dict(), "\n")


if __name__ == "__main__":
    main()
