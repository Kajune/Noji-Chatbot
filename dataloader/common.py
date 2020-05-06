import math
import random
import itertools
import torch

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
 
class Voc:
	def __init__(self):
		self.trimmed = False
		self.word2index = {}
		self.word2count = {}
		self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
		self.num_words = 3  # Count SOS, EOS, PAD
 
	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord(word)
 
	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.num_words
			self.word2count[word] = 1
			self.index2word[self.num_words] = word
			self.num_words += 1
		else:
			self.word2count[word] += 1
 
	# Remove words below a certain count threshold
	def trim(self, min_count):
		if self.trimmed:
			return
		self.trimmed = True
 
		keep_words = []
 
		for k, v in self.word2count.items():
			if v >= min_count:
				keep_words.append(k)
 
		print('keep_words {} / {} = {:.4f}'.format(
			len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
		))
 
		# Reinitialize dictionaries
		self.word2index = {}
		self.word2count = {}
		self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
		self.num_words = 3 # Count default tokens
 
		for word in keep_words:
			self.addWord(word)

	def indicesFromSentence(self, sentence):
		return [self.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def readVocs(datafile, sanitize):
	print("Reading lines...")
	# Read the file and split into lines
	lines = open(datafile, encoding='utf-8').\
		read().strip().split('\n')
	# Split every line into pairs and normalize
	pairs = [[sanitize(s) for s in l.split('\t')] for l in lines]
	voc = Voc()
	return voc, pairs

def filterPair(p, max_length):
	return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length 

def filterPairs(pairs, max_length):
	return [pair for pair in pairs if filterPair(pair, max_length)]

def trimRareWords(voc, pairs, MIN_COUNT):
	# Trim words used under the MIN_COUNT from the voc
	voc.trim(MIN_COUNT)
	# Filter out pairs with trimmed words
	keep_pairs = []
	for pair in pairs:
		input_sentence = pair[0]
		output_sentence = pair[1]
		keep_input = True
		keep_output = True
		# Check input sentence
		for word in input_sentence.split(' '):
			if word not in voc.word2index:
				keep_input = False
				break
		# Check output sentence
		for word in output_sentence.split(' '):
			if word not in voc.word2index:
				keep_output = False
				break
 
		# Only keep pairs that do not contain trimmed word(s) in their input or output sentence
		if keep_input and keep_output:
			keep_pairs.append(pair)
 
	print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs), len(keep_pairs) / len(pairs)))
	return keep_pairs

def loadPrepareData(datafile, max_length, min_count, sanitize):
	print("Start preparing training data ...")
	voc, pairs = readVocs(datafile, sanitize)
	print("Read {!s} sentence pairs".format(len(pairs)))
	pairs = filterPairs(pairs, max_length)
	print("Trimmed to {!s} sentence pairs".format(len(pairs)))
	print("Counting words...")
	for pair in pairs:
		voc.addSentence(pair[0])
		voc.addSentence(pair[1])
	print("Counted words:", voc.num_words)

	pairs = trimRareWords(voc, pairs, min_count)

	return voc, pairs
  
def zeroPadding(l, fillvalue=PAD_token):
	return list(itertools.zip_longest(*l, fillvalue=fillvalue))
 
def binaryMatrix(l, value=PAD_token):
	m = []
	for i, seq in enumerate(l):
		m.append([])
		for token in seq:
			if token == PAD_token:
				m[i].append(0)
			else:
				m[i].append(1)
	return m
 
def inputVar(l, voc):
	indexes_batch = [voc.indicesFromSentence(sentence) for sentence in l]
	lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
	padList = zeroPadding(indexes_batch)
	padVar = torch.LongTensor(padList)
	return padVar, lengths
 
def outputVar(l, voc):
	indexes_batch = [voc.indicesFromSentence(sentence) for sentence in l]
	max_target_len = max([len(indexes) for indexes in indexes_batch])
	padList = zeroPadding(indexes_batch)
	mask = binaryMatrix(padList)
	mask = torch.BoolTensor(mask)
	padVar = torch.LongTensor(padList)
	return padVar, mask, max_target_len
 
def batch2TrainData(voc, pair_batch):
	pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
	input_batch, output_batch = [], []
	for pair in pair_batch:
		input_batch.append(pair[0])
		output_batch.append(pair[1])
	inp, lengths = inputVar(input_batch, voc)
	output, mask, max_target_len = outputVar(output_batch, voc)
	return inp, lengths, output, mask, max_target_len

class TextDataset():
	def __init__(self, datafile, max_length, min_count, sanitize):
		self.voc, self.pairs = loadPrepareData(datafile, max_length, min_count, sanitize)

	def __len__(self):
		return len(self.pairs)

	def __getitem__(self, idx):
		return self.pairs[idx]

	def getVoc(self):
		return self.voc

class TextDataloader():
	def __init__(self, dataset, batch_size, shuffle=True):
		self.dataset = dataset
		self.batch_size = batch_size

		self.indices = [i for i in range(len(self.dataset))]
		if shuffle:
			random.shuffle(self.indices)

	def __len__(self):
		return math.ceil(len(self.indices) / self.batch_size)

	def __iter__(self):
		for i in range(self.__len__()):
			indices = self.indices[i * self.batch_size : (i + 1) * self.batch_size]
			batch = []
			for ind in indices:
				batch.append(self.dataset[ind])
			trainData = batch2TrainData(self.dataset.getVoc(), batch)
			yield trainData
