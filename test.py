import torch
import argparse

from dataloader.cornell import *
from dataloader.convai2 import *
from dataloader.nucc import *
from dataloader.common import TextDataloader, PAD_token, SOS_token, EOS_token
from dataloader import utils
from model.seq2seq import Seq2SeqModel

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--iteration', type=int, default=10)
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-l', '--load', type=str)
parser.add_argument('-e', '--eval', action='store_true')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = []
#dataset.extend(loadCornellDataset('data/cornell movie-dialogs corpus'))
#dataset.extend(loadConvAI2Dataset('data/ConvAI2'))
dataset.extend(loadNUCCDataset('data/nucc'))

dataloader = TextDataloader(dataset, max_length=32, min_count=3, batch_size=args.batch_size, shuffle=True)
voc = dataloader.getVoc()

model = Seq2SeqModel(device, SOS_token, voc.num_words).to(device)

if args.load:
	model.load_state_dict(torch.load(args.load))

if not args.eval:
	model.train()

	for epoch in range(args.iteration):
		for i, data in enumerate(dataloader):
			inputs, lengths, targets, mask, max_target_len = data
			inputs = inputs.to(device)
			lengths = lengths.to(device)
			targets = targets.to(device)
			mask = mask.to(device)

			print_loss = model.optimize(inputs, lengths, targets, mask, max_target_len)

			if i % 10 == 0:
				print('[Epoch: %d, %d/%d] loss: %f' % (epoch, i, len(dataloader), print_loss))

		torch.save(model.state_dict(), 'weights/%03d.pth' % (epoch))

model.eval()

def evaluate(model, voc, sentence, max_length=10):
	indexes_batch = [voc.indicesFromSentence(sentence)]
	lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
	input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
	input_batch = input_batch.to(device)
	lengths = lengths.to(device)
	tokens, scores = model.evaluate(input_batch, lengths, max_length)
	decoded_words = [voc.index2word[token.item()] for token in tokens]
	return decoded_words 
 
def evaluateInput(model, voc):
	input_sentence = ''
	while(1):
		try:
			input_sentence = input('> ')
			if input_sentence == 'q' or input_sentence == 'quit': break
			input_sentence = utils.normalizeJapaneseString(input_sentence)
			output_words = evaluate(model, voc, input_sentence)
			output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
			print('Bot:', ' '.join(output_words))
 
		except KeyError:
			print("Error: Encountered unknown word.")

evaluateInput(model, voc)
