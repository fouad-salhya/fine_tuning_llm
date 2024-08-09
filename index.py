# Importer les bibliothèques nécessaires
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Télécharger le tokenizer et le modèle pré-entraîné
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Configurer l'appareil pour utiliser le CPU
device = torch.device("cpu")
model.to(device)

# Charger le jeu de données tweet_eval pour l'analyse des sentiments
dataset = load_dataset('tweet_eval', 'sentiment')

# Prétraitement : Tokenisation des données
def tokenize_function(data):
    return tokenizer(data['text'], padding="max_length", truncation=True)

# Appliquer la tokenisation au jeu de données
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Diviser les données en ensembles d'entraînement et de test
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir='./results',          # Répertoire de sortie pour les checkpoints du modèle
    num_train_epochs=1,              # Nombre d'époques (réduites pour le test rapide)
    per_device_train_batch_size=8,   # Taille du batch pour l'entraînement
    per_device_eval_batch_size=16,   # Taille du batch pour l'évaluation
    warmup_steps=500,                # Nombre d'étapes de warmup
    weight_decay=0.01,               # Taux de décroissance des poids (regularization)
    logging_dir='./logs',            # Répertoire pour les logs TensorBoard
    logging_steps=10,
    evaluation_strategy="epoch"      # Évaluation à chaque époque
)

# Définir le Trainer
trainer = Trainer(
    model=model,                         # Le modèle à entraîner
    args=training_args,                  # Arguments d'entraînement
    train_dataset=train_dataset,         # Jeu de données d'entraînement
    eval_dataset=test_dataset            # Jeu de données de validation (évaluation)
)

# Entraîner le modèle
trainer.train()

# Évaluer le modèle
eval_results = trainer.evaluate()
print(f"Résultats de l'évaluation : {eval_results}")

# Sauvegarder le modèle entraîné
trainer.save_model("trained_bert_sentiment_model")
tokenizer.save_pretrained("trained_bert_sentiment_model")

print("L'entraînement est terminé et le modèle est sauvegardé.")
