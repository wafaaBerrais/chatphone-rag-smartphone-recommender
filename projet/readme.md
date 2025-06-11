## présentation des fichiers :

### app.py : 
notre backend où on recoit les reponses de api resultats de recherche et on commence la generation avec le llm

### CHATEPHONEtemplate.html :
le frontend de systeme 

### pretraitement_dataset.ipynb :
contient juste un sauvgarde de mon code de notebook colab
donc voici le lien pour executer : https://colab.research.google.com/drive/1L4VfFfvPBrtPR6GIToEEUidAWTBPKmC2#scrollTo=BSSa0pDS6Uba&uniqifier=1

### rep dataset : 
contient juste la base de donnees d'amazon avant aucun traitement 

## note
- vous trouver les fichier que je pravaille avec dans mon notebook dans mon drive exactement le dossier tmp et vous avez l'acces a ce dossier 

## Comment faire pour tester ou executer
- télécharger ollama et qwen3:0.6b avec "ollama pull qwen3:0.6b"
- activer l'envirenement "source venv/bin/activate"
- passer a notebook colab executer d'abord les pip des installation necessaire si une installation necessite le redemarage redemerer et apres continuer à installer ce que il ya apres cette derniere 
apres vous aller au dernier code "API" vous le lancer il va vous donner une resultat qui resemble à ca 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(

modules.json: 100%
 349/349 [00:00<00:00, 13.1kB/s]
config_sentence_transformers.json: 100%
 116/116 [00:00<00:00, 4.04kB/s]
README.md: 100%
 10.5k/10.5k [00:00<00:00, 556kB/s]
sentence_bert_config.json: 100%
 53.0/53.0 [00:00<00:00, 1.11kB/s]
config.json: 100%
 612/612 [00:00<00:00, 21.3kB/s]
model.safetensors: 100%
 90.9M/90.9M [00:02<00:00, 45.6MB/s]
tokenizer_config.json: 100%
 350/350 [00:00<00:00, 11.3kB/s]
vocab.txt: 100%
 232k/232k [00:00<00:00, 2.65MB/s]
tokenizer.json: 100%
 466k/466k [00:00<00:00, 5.50MB/s]
special_tokens_map.json: 100%
 112/112 [00:00<00:00, 1.49kB/s]
config.json: 100%
 190/190 [00:00<00:00, 5.00kB/s]

API accessible via : https://fbd2-34-106-155-228.ngrok-free.app     <------------------------------------------- vous recuperer cet url 


INFO:     Started server process [33535]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit) 

- apres avoir recuperer l'URL "https://fbd2-34-106-155-228.ngrok-free.app"
- lancer dans le terminale de votre local "python3 app.py https://fbd2-34-106-155-228.ngrok-free.app // n'oublie pas de remplacer l'url par celle recuperer apres l'execusion de l'api
- apres vous lancer le CHATPHONEtemplate.html dans chrome ou autre navigateur (il suffait de l'ouvrir) puis vous commencer a discuter avec l'agent de dialogue

## note important concernat ollama "qwen"
- si vous auriez cette erreur ou reponse de chatbot "Désolé, une erreur est survenue lors de la génération." assurez vous que votre connexion est bonne et vous avez environ 1.6 GB de ram libre pour que ollama tourne 


- pour les pip instal necessaire a l'execusion de l'API commencer par ca 
!pip install sentence-transformers faiss-cpu
- puis ca 
!pip install --upgrade protobuf tensorflow
!pip uninstall -y torch torchvision torchaudio sentence-transformers
!pip install --no-cache-dir torch sentence-transformers
!pip install --no-cache-dir torchvision timm
!pip install --no-cache-dir vaderSentiment
!pip install sentence-transformers faiss-cpu pandas fastapi uvicorn pyngrok
!pip install pyngrok
!ngrok authtoken ***REMOVED***
!pip install --quiet fastapi uvicorn nest_asyncio
- puis ca 
!pip uninstall -y sentence-transformers transformers huggingface-hub
!pip install sentence-transformers==2.2.2 transformers==4.30.2 huggingface-hub==0.15.1
- puis
!pip install pyngrok
!ngrok authtoken ***REMOVED***

- SI VOUS CONSTATEZ OU RENCONTREZ DES ERREUR TECHNIQUE OU DIFFICULTÉ D'INSTALATION OU DE CONFIGURATION N'HÉSITEZ PAS DE ME CONTACTER
