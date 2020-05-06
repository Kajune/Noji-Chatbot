import os
import codecs
import csv

from . import common
from . import utils

def loadLines(fileName, fields):
	lines = {}
	with open(fileName, 'r', encoding='iso-8859-1') as f:
		for line in f:
			values = line.split(" +++$+++ ")
			# Extract fields
			lineObj = {}
			for i, field in enumerate(fields):
				lineObj[field] = values[i]
			lines[lineObj['lineID']] = lineObj
	return lines

def loadConversations(fileName, lines, fields):
	conversations = []
	with open(fileName, 'r', encoding='iso-8859-1') as f:
		for line in f:
			values = line.split(" +++$+++ ")
			# Extract fields
			convObj = {}
			for i, field in enumerate(fields):
				convObj[field] = values[i]
			# Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
			lineIds = eval(convObj["utteranceIDs"])
			# Reassemble lines
			convObj["lines"] = []
			for lineId in lineIds:
				convObj["lines"].append(lines[lineId])
			conversations.append(convObj)
	return conversations

def extractSentencePairs(conversations):
	qa_pairs = []
	for conversation in conversations:
		# Iterate over all the lines of the conversation
		for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
			inputLine = conversation["lines"][i]["text"].strip()
			targetLine = conversation["lines"][i+1]["text"].strip()
			# Filter wrong samples (if one of the lists is empty)
			if inputLine and targetLine:
				qa_pairs.append([inputLine, targetLine])
	return qa_pairs

class CornellDataset(common.TextDataset):
	def __init__(self, path, max_length=10, min_count=3):
		datafile = os.path.join(path, 'formatted_movie_lines.txt')
		
		if not os.path.exists(datafile):
			MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
			MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

			lines = loadLines(os.path.join(path, 'movie_lines.txt'), MOVIE_LINES_FIELDS)
			conversations = loadConversations(os.path.join(path, "movie_conversations.txt"), lines, MOVIE_CONVERSATIONS_FIELDS)

			delimiter = str(codecs.decode('\t', "unicode_escape"))
			with open(datafile, 'w', encoding='utf-8') as outputfile:
				writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
				for pair in extractSentencePairs(conversations):
					writer.writerow(pair)

		super().__init__(datafile, max_length, min_count, utils.normalizeString)

