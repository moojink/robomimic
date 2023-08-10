import clip
import torch
from transformers import AutoTokenizer, AutoModel
from torch import nn

class SentenceEncoder(nn.Module):
    def __init__(self, sentence_encoder: str):
        """CLIP-based or DistilBERT-based sentence encoder for language commands. Language embeddings are frozen."""
        super(SentenceEncoder, self).__init__()
        if sentence_encoder == 'clip':
            self.device = 'cuda'
            self.model, _ = clip.load("ViT-L/14@336px", device=self.device) # ViT-L/14@336px has highest performance among CLIP models; embedding size == 768
        elif sentence_encoder == 'distilbert':
            self.device = 'cuda'
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = AutoModel.from_pretrained("distilbert-base-uncased") # embedding size == 768
        else:
            raise ValueError(f'Invalid sentence_encoder: {sentence_encoder}')
        self.sentence_encoder = sentence_encoder
        self.features_dim = 768 # for both CLIP ViT-L/14@336px and DistilBERT

    def forward(self, language_command):
        """Takes a list of text strings and outputs a single sentence embedding."""
        with torch.no_grad():
            if self.sentence_encoder == 'clip':
                tokens = clip.tokenize(language_command).to(self.device)
                language_embed = self.model.encode_text(tokens)
                language_embed = language_embed.float()
            else:
                tokens = self.tokenizer(language_command, return_tensors='pt', max_length=5, padding="max_length", truncation=True)
                input_ids = tokens['input_ids'].to(self.device)
                attention_mask = tokens['attention_mask'].to(self.device)
                language_embed = self.model(input_ids, attention_mask).last_hidden_state # shape: (batch_size, max_sentence_length, embed_size=768)
                language_embed = language_embed.mean(1) # average across dimension 1, i.e., average across all word embeddings to produce a sentence embedding
        return language_embed
