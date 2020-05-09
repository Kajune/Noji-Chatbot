import os
import json

from . import utils

def loadConvAI2Dataset(path):
	data = json.load(open(os.path.join(path, 'summer_wild_evaluation_dialogs.json'), 'r'))

	pairs = []
	for line in data:
		for i in range(len(line['dialog'])-1):
			pairs.append([line['dialog'][i]['text'], line['dialog'][i+1]['text']])

	pairs = [[utils.normalizeString(s) for s in l] for l in pairs]

	return pairs

