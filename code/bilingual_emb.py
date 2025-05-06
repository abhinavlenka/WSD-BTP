import os
import numpy as np
from gensim.models import KeyedVectors
from tqdm import tqdm
from fastText_multilingual.fasttext import FastVector

from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# ─── Static Embeddings Loader ─────────────────────────────────────────
def load_w2v_with_tqdm(fname, vector_size=300):
    """Load a .vec file into a KeyedVectors with a tqdm progress bar."""
    with open(fname, "r", encoding="utf-8", errors="ignore") as f:
        total = sum(1 for _ in f) - 1

    kv = KeyedVectors(vector_size=vector_size)
    words, vecs = [], []

    with open(fname, "r", encoding="utf-8", errors="ignore") as f:
        next(f)
        for line in tqdm(f, total=total, desc=f"Loading {os.path.basename(fname)}"):
            parts = line.rstrip().split()
            words.append(parts[0])
            vecs.append(np.array(parts[1:], dtype="float32"))

    kv.add_vectors(words, vecs)
    return kv

# ─── MBERT Contextual Embedder ─────────────────────────────────────────
class MBertEmbedder:
    def __init__(self, model_name="bert-base-multilingual-cased", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def embed_text(self, text):
        """Return a fixed-size embedding for a given text (word or sentence)."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=32)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Mean-pool last hidden state
        last_hidden = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden]
        attention_mask = inputs['attention_mask'].squeeze(0).unsqueeze(-1)
        masked = last_hidden * attention_mask
        summed = masked.sum(dim=0)
        counts = attention_mask.sum(dim=0)
        return (summed / counts).cpu().numpy()

    def most_similar(self, query, candidates, topn=10):
        """Compute cosine similarity between query and list of candidates."""
        q_vec = self.embed_text(query).reshape(1, -1)
        cand_vecs = np.vstack([self.embed_text(c) for c in candidates])
        sims = cosine_similarity(q_vec, cand_vecs).flatten()
        idx = np.argsort(-sims)[:topn]
        return [(candidates[i], float(sims[i])) for i in idx]

# ─── Paths ────────────────────────────────────────────
english_embedding_path = "cc.en.300.vec"
frisian_embedding_path = "cc.fy.300.vec"

# ─── Load Static Models ───────────────────────────────────
en_model = load_w2v_with_tqdm(english_embedding_path)
fy_model = load_w2v_with_tqdm(frisian_embedding_path)

# ─── Initialize MBERT Embedder ────────────────────────────────────
embedder = MBertEmbedder()

# ─── Example Usage ───────────────────────────────────────
print("\nStatic English similar to 'cat':", en_model.most_similar('cat', topn=5))
print("Static Frisian similar to 'kat':", fy_model.most_similar('kat', topn=5))

# ─── Contextual Similarity ─────────────────────────────
candidates = ['dog', 'feline', 'kitten', 'mouse', 'pet']
print("\nMBERT most similar to 'cat':", embedder.most_similar('cat', candidates))

# ─── Cross-Lingual Contextual ──────────────────────────
# Compare English 'cat' with Frisian words by embedding each Frisian candidate
fy_candidates = ['kat', 'hûn', 'húsdier', 'mûs', 'hynder']
print("\nMBERT cross-lingual similar to 'cat':", embedder.most_similar('cat', fy_candidates))

# ─── (Optional) FastText Translation ─────────────────
en_dictionary = FastVector(vector_file=english_embedding_path)
fy_dictionary = FastVector(vector_file=frisian_embedding_path)

print("\nFrisian translation of English 'cat':", fy_dictionary.translate_nearest_neighbour(en_dictionary['cat']))
print("\nEnglish translation of Frisian 'kat':", en_dictionary.translate_nearest_neighbour(fy_dictionary['kat']))