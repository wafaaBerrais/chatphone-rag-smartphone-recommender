import requests
import subprocess

NGROK_URL = "https://a0ce-34-139-239-126.ngrok-free.app"

def retrieve(query: str, k: int = 5):
    resp = requests.post(f"{NGROK_URL}/search",
                         json={"text": query, "k": k})
    resp.raise_for_status()
    return resp.json()["results"]

def generate_with_ollama(prompt: str) -> str:
    cmd = [
        "ollama", "run", "qwen3:0.6b",
        prompt
    ]
    proc = subprocess.Popen(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True)
    out, err = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Ollama error (rc={proc.returncode}):\n{err}")
    return out.strip()

def generate(query: str, infos: list) -> str:
    first      = infos[0]
    phone_desc = f"{first['brand']} {first['model']}"

    excerpts = [
        info['review'].replace("\n", " ")[:150] + "…"
        for info in infos[:3]
    ]
    context = "\n".join(f"- {e}" for e in excerpts)

    prompt = (
        f"Vous êtes un assistant expert en smartphones.\n"
        f"Ces extraits proviennent du {phone_desc} :\n{context}\n\n"
        f"Question : {query}\n"
        "En une seule réponse synthétique, commencez par « Le "
        f"{phone_desc} se distingue par » et expliquez pourquoi c'est "
        "une excellente option photo, sans recopier textuellement les extraits.\n\n"
        "Réponse :"
    )

    return generate_with_ollama(prompt)

if __name__=="__main__":
    q     = "Quel smartphone pour photo ?"
    infos = retrieve(q, k=5)
    print("→ infos RAG :", infos, "\n")
    answer = generate(q, infos)
    print("=== RÉPONSE DU BOT ===\n", answer)