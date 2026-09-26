import os
import sys
import argparse
import logging
import re
import requests
import subprocess
from flask import Flask, request, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(
    description="Lancer ChatPhone en spécifiant l'URL de l'API de recherche (optionnel)."
)
parser.add_argument(
    "ngrok_url", nargs="?",
    help="URL de l'API de recherche (locale ou tunnel Ngrok). Par défaut : http://127.0.0.1:8000"
)
args = parser.parse_args()

NGROK_URL = (args.ngrok_url or os.getenv("SEARCH_API_URL") or os.getenv("NGROK_URL")
             or "http://127.0.0.1:8000").rstrip("/")

OLLAMA_MODEL = "qwen3:0.6b"

GREETING_KEYWORDS = ["salut", "bonjour", "hello", "hi", "hey", "coucou", "yo"]
THANKS_KEYWORDS   = ["merci", "thanks", "thank you", "remercie", "cool merci"]

def is_greeting(q): return any(w in q.lower() for w in GREETING_KEYWORDS)
def is_thanks(q):   return any(w in q.lower() for w in THANKS_KEYWORDS)

def retrieve(query: str, k: int = 20):
    logging.info(f"Retrieving {k} candidates for '{query}' from {NGROK_URL}")
    try:
        resp = requests.post(
            f"{NGROK_URL}/search",
            json={"text": query, "k": k}, timeout=15
        )
        resp.raise_for_status()
        items = resp.json().get("results", [])
        logging.info(f"→ {len(items)} items retrieved.")
        return items
    except Exception as e:
        logging.error(f"Error in retrieve(): {e}")
        return []

def generate_with_ollama(prompt: str) -> str:
    cmd = ["ollama", "run", OLLAMA_MODEL, prompt]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        out, err = proc.communicate(timeout=90)
        if proc.returncode != 0:
            logging.error(f"Ollama rc={proc.returncode}\n{err}")
            return "Désolé, une erreur est survenue lors de la génération."
        return out.strip()
    except Exception as e:
        logging.error(f"Ollama exception: {e}")
        return "Désolé, une erreur interne s'est produite."

def strip_think(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def generate(query: str, infos: list) -> str:
    if infos:
        ctx_lines = []
        for i, itm in enumerate(infos[:5], start=1):
            brand = itm.get("brand", "").strip()
            model = itm.get("model", "").strip()
            review = itm.get("review", "").replace("\n", " ").strip()
            snippet = review[:200] + "…" if len(review) > 200 else review
            ctx_lines.append(f"{i}. {brand} {model} — «{snippet}»")
        context_str = "\n".join(ctx_lines)
    else:
        context_str = "Aucune information récupérée."

    prompt = (
        "Vous êtes ChatPhone, un assistant IA expert en smartphones.\n\n"
        "Contexte (extraits d'avis sur plusieurs modèles) :\n"
        f"{context_str}\n\n"
        "Instructions IMPÉRATIVES pour répondre :\n"
        "1. Si la question NE CONCERNE PAS la recherche ou l'évaluation ou la recommandation de smartphones, "
        "répondez EXACTEMENT :\n"
        "   \"Je suis un assistant spécialisé dans la recommandation de smartphones, "
        "je ne suis pas capable de répondre à ce type de question.\"\n"
        "2. SINON (la question concerne les smartphones) :\n"
        "   a) Identifiez la ou les caractéristiques demandées (ex. «bonne caméra», "
        "«256GB de stockage», «16GB RAM», etc.).\n"
        "   b) Parcourez les extraits ci-dessus et choisissez le modèle le plus adapté.\n"
        "   c) Répondez EXACTEMENT en une seule phrase, au format :\n"
        "      \"Je recommande le [Marque Modèle] car [justification courte basée sur le contexte].\"\n"
        "   d) Ne citez pas textuellement les avis, ne mentionnez qu'un seul modèle et n'ajoutez "
        "aucune autre phrase.\n"
        "3. Si aucun modèle ne correspond aux critères, répondez EXACTEMENT :\n"
        "   \"Je n'ai pas suffisamment d'informations dans ma base de données pour "
        "identifier un modèle correspondant précisément à votre demande.\"\n\n"
        f"Question de l'utilisateur : \"{query}\"\n"
        "Réponse :"
    )

    raw = generate_with_ollama(prompt)
    return strip_think(raw)


app = Flask(__name__)
CORS(app)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json() or {}
    query = data.get('query','').strip()
    if not query:
        return jsonify({'response': "Veuillez poser une question."}), 400

    if is_greeting(query):
        return jsonify({'response': "Bonjour ! Je suis ChatPhone. Comment puis-je vous aider ?"})
    if is_thanks(query):
        return jsonify({'response': "Avec plaisir ! N'hésitez pas si vous avez d'autres questions."})

    infos  = retrieve(query, k=20)
    answer = generate(query, infos)
    return jsonify({'response': answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)