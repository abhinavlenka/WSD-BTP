from nltk.corpus import wordnet as wn
from collections import defaultdict
from googletrans import Translator
from simalign import SentenceAligner

# Input sentence
# f = open("fry_news_2020_10K-sentences.txt","r")
# for fry_sentence in f.readlines()[:5]:
#     # print(fry_sentence)
# 
fry_sentence = "2021 betekent nog een jaar heel hard werken voor m'n vrouw en mij."

# 2. Translation
translator = Translator()
translated = translator.translate(fry_sentence, dest='en')
eng_sentence = translated.text
print(f"\nTranslation: {eng_sentence}")

# 3. Tokenization   
fry_words = [w.strip(".'") for w in fry_sentence.split()]
eng_words = eng_sentence.replace('.', '').split()

# 4. Alignment
aligner = SentenceAligner(model="bert", token_type="word", matching_methods="mai")
result = aligner.get_word_aligns(fry_words, eng_words)

# Formatting aligned pairs
alignments = [
    (fry_words[i], eng_words[j]) 
    for i, j in result['itermax']
    if i < len(fry_words) and j < len(eng_words)
]

print("\Fry ↔ English Alignment:")
for dutch, english in alignments:
    print(f"{dutch:<15} ↔ {english}")

# Create Dutch WordNet structure
dutch_wn = defaultdict(lambda: {
    'synsets': [],
    'lemmas': [],
    'translations': []
})

for dutch_word, eng_word in alignments:
    # Get all WordNet synsets for English word
    eng_synsets = wn.synsets(eng_word)
    
    # Create Dutch entry with English connections
    entry = dutch_wn[dutch_word]
    entry['translations'].append(eng_word)
    
    for synset in eng_synsets:
        entry['synsets'].append({
            'pos': synset.pos(),
            'definition': synset.definition(),
            # 'examples': synset.examples(),
            'english_lemma': synset.lemmas()[0].name()
        })

# Print WordNet-style entries
for word, data in dutch_wn.items():
    print(f"\n** {word.upper()} **")
    print(f"Translations: {', '.join(data['translations'])}")
    
    for i, synset in enumerate(data['synsets'], 1):
        print(f"\nSense {i}:")
        print(f"POS: {synset['pos'].upper()}")
        print(f"Definition: {synset['definition']}")
        # print(f"Examples: {synset['examples']}")
        print(f"Linked to: {synset['english_lemma']} (English)")
# print((dutch_wn.items()))