import os
import json
import glob
import re

from . import utils

def sanitizeText(text):
	text = re.sub(r'（.+）', '', text)
	text = re.sub(r'＜.+＞', '', text)

	return text

def loadNUCCDataset(path):
	fileList = glob.glob(os.path.join(path, '*.txt'))
	datafile = os.path.join(path, 'formated_lines.txt')

	pairs = []

	if not os.path.exists(datafile):
		for file in fileList:
			sentenceList = []
			text = ''
			for line in open(file, 'r', encoding='utf-8'):
				if line[0] == '＠' or line[0] == '％':
					continue
				line = line.replace('\n', '')
				match = re.match(r'[FM]\d\d\d：', line)
				if match is not None and len(text) > 0:
					sentenceList.append(sanitizeText(text))
					text = line[match.end():]
				else:
					text += line
		
			if len(text) > 0:
				sentenceList.append(sanitizeText(text))

			for i in range(len(sentenceList) - 1):
				if not '＊＊＊' in sentenceList[i] and not '＊＊＊' in sentenceList[i+1]:
					pairs.append([sentenceList[i], sentenceList[i+1]])

		writer = open(datafile, 'w', encoding='utf-8')
		for i, l in enumerate(pairs):
			print('\r%d/%d' % (i, len(pairs)), end='')
			pairs[i] = [utils.normalizeJapaneseString(s) for s in l]
			writer.write('%s\t%s\n' % (pairs[i][0], pairs[i][1]))

	else:
		for line in open(datafile, 'r', encoding='utf-8'):
			pairs.append(line.replace('\n', '').split('\t'))

	return pairs
