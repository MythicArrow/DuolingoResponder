
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import MT5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')

# Define the PPO algorithm components
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(1024, 1)  # Adjust size to match mT5-XXL hidden size

    def forward(self, x):
        return self.fc(x)

class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.fc = nn.Linear(1024, 1)  # Adjust size to match mT5-XXL hidden size

    def forward(self, x):
        return self.fc(x)

class PPO:
    def __init__(self, model, tokenizer, policy_net, value_net, clip_epsilon=0.2, gamma=0.99, lr=3e-4):
        self.model = model
        self.tokenizer = tokenizer
        self.policy_net = policy_net
        self.value_net = value_net
        self.clip_epsilon = clip_epsilon
        self.gamma = gamma
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.optimizer_value = optim.Adam(self.value_net.parameters(), lr=lr)

    def compute_rewards(self, predicted_translation, reference_translation):
        reward = self.compute_combined_reward(predicted_translation, reference_translation)
        return reward

    def compute_combined_reward(self, predicted, reference):
        bleu = self.compute_bleu_reward(predicted, reference)
        rouge = self.compute_rouge_reward(predicted, reference)
        meteor = self.compute_meteor_reward(predicted, reference)
        return ((bleu*2) + (rouge*1.5) + (meteor * 2.5) / 3

    def compute_bleu_reward(self, predicted, reference):
        predicted_tokens = nltk.word_tokenize(predicted)
        reference_tokens = [nltk.word_tokenize(reference)]
        return sentence_bleu(reference_tokens, predicted_tokens)

    def compute_rouge_reward(self, predicted, reference):
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        score = scorer.score(reference, predicted)
        return (score['rouge1'].fmeasure + score['rouge2'].fmeasure + score['rougeL'].fmeasure) / 3

    def compute_meteor_reward(self, predicted, reference):
        return meteor_score([reference], predicted)

    def ppo_update(self, states, actions, rewards, old_log_probs, epsilon=0.2):
        for _ in range(10):  # Number of epochs
            new_log_probs = self.policy_net(states)
            values = self.value_net(states)
            advantages = rewards - values.detach()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surrogate_loss = torch.min(ratio * advantages, torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantages)
            value_loss = torch.mean((rewards - values) ** 2)
            self.optimizer_policy.zero_grad()
            surrogate_loss.mean().backward()
            self.optimizer_policy.step()
            self.optimizer_value.zero_grad()
            value_loss.backward()
            self.optimizer_value.step()

    def train_step(self, sentence, reference_translation):
        inputs = self.tokenize_input(sentence, "English", "French")  # Example: English to French
        predicted_translation = self.translate_sentence(sentence, "English", "French")
        reward = self.compute_rewards(predicted_translation, reference_translation)
        states = torch.tensor([inputs], dtype=torch.float32)
        actions = torch.tensor([predicted_translation], dtype=torch.float32)
        old_log_probs = torch.tensor([0.0], dtype=torch.float32)  # Placeholder
        self.ppo_update(states, actions, torch.tensor([reward], dtype=torch.float32), old_log_probs)

    def tokenize_input(self, sentence, source_lang, target_lang):
        input_text = f"translate {source_lang} to {target_lang}: {sentence}"
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        return inputs

    def translate_sentence(self, sentence, source_lang, target_lang):
        inputs = self.tokenize_input(sentence, source_lang, target_lang)
        outputs = self.model.generate(inputs)
        translation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translation

# Load mT5-XXL model and tokenizer
model = MT5ForConditionalGeneration.from_pretrained('google/mt5-xxl')
tokenizer = T5Tokenizer.from_pretrained('google/mt5-xxl')

# Initialize PPO
policy_net = PolicyNetwork()
value_net = ValueNetwork()
ppo = PPO(model, tokenizer, policy_net, value_net)

# Example training step
sentence = "The weather is nice today."
reference_translation = "Il fait beau aujourd'hui."  # Example reference
ppo.train_step(sentence, reference_translation)

def get_human_feedback():
    feedback = input("Was this translation correct? (yes/no): ").strip().lower()
    return 1 if feedback == "yes" else -1
