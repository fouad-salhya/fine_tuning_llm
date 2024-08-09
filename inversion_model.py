import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Télécharger le tokenizer et le modèle pré-entraîné
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=4)

# Configurer l'appareil pour utiliser le CPU
device = torch.device("cpu")
model.to(device)

# Charger le jeu de données AG News
dataset = load_dataset('ag_news')

# Prétraitement : Tokenisation des données
def tokenize_function(data):
    return tokenizer(data['text'], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Utilisation du dataset complet pour l'entraînement et le test
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# Définir les arguments d'entraînement
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Configuration du Trainer pour gérer l'entraînement
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Entraînement du modèle
trainer.train()

# Évaluation du modèle
eval_results = trainer.evaluate()
print(f"Résultats de l'évaluation : {eval_results}")

# Sauvegarde du modèle entraîné et du tokenizer
trainer.save_model("trained_distilbert_agnews_model")
tokenizer.save_pretrained("trained_distilbert_agnews_model")

print("L'entraînement est terminé et le modèle est sauvegardé.")
