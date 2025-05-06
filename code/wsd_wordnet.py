import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import random

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None

def generate_training_examples(word, pos_tag):
    pos = get_wordnet_pos(pos_tag)
    if not pos:
        return []

    examples = []
    senses = wn.synsets(word, pos=pos)
    for syn in senses:
        sense_id = syn.name()
        for ex in syn.examples():
            examples.append((ex, sense_id))
    return examples


def train_wsd_classifier(word, pos_tag):
    data = generate_training_examples(word, pos_tag)
    if not data:
        print(f"No examples found for '{word}' with POS '{pos_tag}'")
        return None, None

    random.shuffle(data)
    texts, labels = zip(*data)

    clf = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2))),
        ('nb', MultinomialNB())
    ])
    clf.fit(texts, labels)
    return clf, wn.synsets(word, pos=get_wordnet_pos(pos_tag))

def disambiguate(clf, context, senses):
    probs = clf.predict_proba([context])[0]
    results = list(zip(clf.classes_, probs))
    results.sort(key=lambda x: -x[1])
    print("Predicted senses:")
    for sense, prob in results:
        syn = next((s for s in senses if s.name() == sense), None)
        print(f"{sense} ({syn.definition() if syn else 'Unknown'}): {prob * 100:.2f}%")


word = "bat"
pos = "NN"  # You can use nltk.pos_tag to auto-detect
clf, senses = train_wsd_classifier(word, pos)

test_context = "He went to play with his bat."
print(test_context)
disambiguate(clf, test_context, senses)
print("-------------------------------------------------")


# print()
word = "novel"
pos = "JJ"  # You can use nltk.pos_tag to auto-detect
clf, senses = train_wsd_classifier(word, pos)

test_context = "He had a novel idea ."
print(test_context)

disambiguate(clf, test_context, senses)
 
print("-------------------------------------------------")

#Testing


f = open("test_wsd.txt","r")
for text in f.readlines():
    word = "bank"
    pos = "NN" 
    clf, senses = train_wsd_classifier(word, pos)
    print(text)
    disambiguate(clf, text, senses)
    print("-------------------------------------------------")
