import unicodedata
import re
import spacy

nlp = spacy.load('ja_ginza')

def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
	)
 
def normalizeString(s):
	s = s.lower().strip()
	s = unicodeToAscii(s)
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	s = re.sub(r"\s+", r" ", s).strip()
	return s

def normalizeJapaneseString(s):
	doc = nlp(s)

	ret = ''
	for sent in doc.sents:
		for token in sent:
			ret += token.orth_ + ' '

	return ret
