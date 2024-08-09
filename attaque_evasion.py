import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Charger le modèle et le tokenizer sauvegardés
model = DistilBertForSequenceClassification.from_pretrained("trained_distilbert_agnews_model")
tokenizer = DistilBertTokenizer.from_pretrained("trained_distilbert_agnews_model")

# Configurer l'appareil pour utiliser le CPU
device = torch.device("cpu")
model.to(device)

# Passer le modèle en mode évaluation
model.eval()

# Fonction pour appliquer une attaque par évasion sur un exemple de texte
def evade_attack(example_text, epsilon=0.1):
    inputs = tokenizer(example_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(device)
    inputs_embeds = model.distilbert.embeddings(inputs.input_ids)

    # Perturber les embeddings avec une petite valeur epsilon
    perturbed_embeds = inputs_embeds + epsilon * torch.sign(torch.randn_like(inputs_embeds))

    with torch.no_grad():
        logits = model(inputs_embeds=perturbed_embeds).logits

    predicted_label = torch.argmax(logits, dim=1).item()
    return predicted_label

# Fonction interactive pour l'attaque par évasion
def interactive_evade_attack():
    while True:
        example_text = input("Entrez un texte pour l'attaque par évasion (ou tapez 'exit' pour quitter) : ")
        if example_text.lower() == 'exit':
            break
        
        original_label = evade_attack(example_text, epsilon=0.0)  # Sans perturbation
        attacked_label = evade_attack(example_text, epsilon=0.1)  # Avec perturbation
        
        print(f"Label original: {original_label}, Label après attaque: {attacked_label}\n")

# Lancer l'attaque par évasion interactive
interactive_evade_attack()

print("Fin de la session d'attaque par évasion.")
