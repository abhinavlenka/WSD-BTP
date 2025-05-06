import gensim
import nltk
from nltk.corpus import wordnet as wn
from pprint import pprint
import sys

# Ensure WordNet is available
nltk.download('wordnet')
nltk.download('omw-1.4')

# Attempt to import googletrans for translation/detection
# Requires `pip install googletrans==4.0.0-rc1`
try:
    from googletrans import Translator, constants
    translator = Translator()
except ImportError:
    translator = None
    sys.stderr.write(
        "[Warning] googletrans not installed or unsupported version. Frisian translation/detection disabled.\n"
    )

# Detect input language: English vs Frisian
# Falls back to WordNet presence if googletrans unavailable

def detect_language(word):
    if translator:
        try:
            detection = translator.detect(word)
            lang = detection.lang if hasattr(detection, 'lang') else detection[0].lang
            if lang == 'fy':
                return 'frisian'
            if lang == 'en':
                return 'english'
        except Exception as e:
            sys.stderr.write(f"[detect_language] translator.detect error: {e}\n")
    if wn.synsets(word):
        return 'english'
    return 'frisian'

# Lookup English senses via WordNet

def lookup_english(word):
    senses = wn.synsets(word)
    entries = []
    for s in senses:
        entries.append({
            'sense_key': s.name(),
            'definition': s.definition(),
            'examples': s.examples(),
            'synonyms': [l.name().replace('_', ' ') for l in s.lemmas()]
        })
    return entries

# Translate text between languages

def translate_text(text, src, dest):
    if not translator:
        return None
    try:
        result = translator.translate(text, src=src, dest=dest)
        return result.text
    except Exception as e:
        sys.stderr.write(f"[translate_text] translation error: {e}\n")
        return None

# Main lookup function: returns both English & Frisian senses

def lookup(word):
    lang = detect_language(word)
    # Pivot to English for WordNet lookup
    if lang == 'english':
        eng_word = word
    else:
        eng_word = translate_text(word, src='fy', dest='en') or word

    eng_entries = lookup_english(eng_word)

    fy_entries = []
    for entry in eng_entries:
        # Translate definition
        def_fy = translate_text(entry['definition'], src='en', dest='fy') or entry['definition']
        # Translate examples
        ex_fy = []
        for ex in entry.get('examples', []):
            ex_trans = translate_text(ex, src='en', dest='fy') or ex
            ex_fy.append(ex_trans)
        # Translate synonyms
        syns_fy = []
        for syn in entry['synonyms']:
            syn_trans = translate_text(syn, src='en', dest='fy') or syn
            syns_fy.append(syn_trans)
        fy_entries.append({
            'sense_key': entry['sense_key'],
            'definition': def_fy,
            'examples': ex_fy,
            'synonyms': syns_fy
        })

    return {
        'input_language': lang,
        'pivot_word': eng_word,
        'english_senses': eng_entries,
        'frisian_senses': fy_entries
    }

if __name__ == '__main__':
    test_words = ['house']
    for w in test_words:
        print(f"Lookup '{w}':")
        result = lookup(w)
        pprint(result)
        print('-' * 60)
