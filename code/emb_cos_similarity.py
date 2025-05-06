from nltk.corpus import wordnet as wn
from collections import defaultdict
from googletrans import Translator
from simalign import SentenceAligner
from numpy import dot
from numpy.linalg import norm
import gensim
from tqdm import tqdm  # progress bar


def load_fasttext_with_tqdm(path):
    """
    Load a FastText .vec file into a gensim KeyedVectors
    with a tqdm progress bar.
    """
    # 1. Count lines (to give tqdm a total)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        header = f.readline().split()
        total_lines = int(header[0])  # vocab size
        vector_size = int(header[1])

    # 2. Initialize empty KeyedVectors
    kv = gensim.models.KeyedVectors(vector_size=vector_size)

    # 3. Read & collect all vectors with tqdm
    words = []
    vecs  = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)  # skip header
        for line in tqdm(f, total=total_lines, desc=f"Loading {path}"):
            parts = line.rstrip().split(' ')
            words.append(parts[0])
            vecs.append([float(x) for x in parts[1:]])

    # 4. Add to KeyedVectors in one go
    kv.add_vectors(words, vecs)
    return kv


# ------------------------------------------------------------------
# Replace your old loading block with:

print("Initializing FastText embeddings…")
eng_model = load_fasttext_with_tqdm("cc.en.300.vec")
fry_model = load_fasttext_with_tqdm("cc.fy.300.vec")
print("  ↳ FastText embeddings ready.\n")
# ------------------------------------------------------------------

# …then the rest of your code remains the same:
#   cosine_sim, translation, tokenization, SimAlign, BWE filtering, etc.

# Load FastText embeddings
print("Loading FastText embeddings...")
eng_model = gensim.models.KeyedVectors.load_word2vec_format("cc.en.300.vec", binary=False)
fry_model = gensim.models.KeyedVectors.load_word2vec_format("cc.fy.300.vec", binary=False)
print("  ↳ Embeddings loaded.\n")

def cosine_sim(v1, v2):
    return dot(v1, v2) / (norm(v1) * norm(v2))

# Input sentence
fry_sentence = "2021 betekent nog een jaar heel hard werken voor m'n vrouw en mij."

# Translation
translator = Translator()
translated = translator.translate(fry_sentence, dest='en')
eng_sentence = translated.text
print(f"\nTranslation: {eng_sentence}")

# Tokenization
fry_words = [w.strip(".'") for w in fry_sentence.split()]
eng_words = eng_sentence.replace('.', '').split()

# Alignment
aligner = SentenceAligner(model="bert", token_type="word", matching_methods="mai")
result = aligner.get_word_aligns(fry_words, eng_words)

# BWE-based filtering of alignments (with tqdm)
threshold = 0.01
filtered_alignments = []
print("\nFiltering alignments using BWE similarity:")
for i, j in tqdm(result['itermax'], desc="Alignments processed"):
    if i >= len(fry_words) or j >= len(eng_words):
        continue
    fw, ew = fry_words[i], eng_words[j]
    if fw in fry_model and ew in eng_model:
        sim = cosine_sim(fry_model[fw], eng_model[ew])
        if sim >= threshold:
            filtered_alignments.append((fw, ew))

# WordNet integration (with tqdm)
dutch_wn = defaultdict(lambda: {
    'synsets': [],
    'translations': []
})

print("\nFetching WordNet synsets:")
for fw, ew in tqdm(filtered_alignments, desc="Word pairs processed"):
    eng_synsets = wn.synsets(ew)
    entry = dutch_wn[fw]
    entry['translations'].append(ew)
    for syn in eng_synsets:
        entry['synsets'].append({
            'pos':        syn.pos(),
            'definition': syn.definition(),
            'lemma':      syn.lemmas()[0].name()
        })

# Output results
for word, data in dutch_wn.items():
    print(f"\n** {word.upper()} **")
    print("Translations:", ", ".join(data['translations']))
    for idx, s in enumerate(data['synsets'], 1):
        print(f"\nSense {idx}:")
        print(f"  POS       : {s['pos']}")
        print(f"  Definition: {s['definition']}")
        print(f"  Lemma     : {s['lemma']}")
