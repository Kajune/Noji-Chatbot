import unicodedata
import re

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
