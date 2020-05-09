import os
import codecs

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

def loadCornellDataset(path):
	MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
	MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

	lines = loadLines(os.path.join(path, 'movie_lines.txt'), MOVIE_LINES_FIELDS)
	conversations = loadConversations(os.path.join(path, "movie_conversations.txt"), lines, MOVIE_CONVERSATIONS_FIELDS)

	pairs = []
	for conversation in conversations:
		for i in range(len(conversation["lines"]) - 1):
			inputLine = conversation["lines"][i]["text"].strip()
			targetLine = conversation["lines"][i+1]["text"].strip()
			if inputLine and targetLine:
				pairs.append([inputLine, targetLine])

	pairs = [[utils.normalizeString(s) for s in l] for l in pairs]

	return pairs

