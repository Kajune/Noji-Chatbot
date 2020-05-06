import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

class EncoderRNN(nn.Module):
	def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
		super(EncoderRNN, self).__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size
		self.embedding = embedding
 
		# Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
		#   because our input size is a word embedding with number of features == hidden_size
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
						  dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
 
	def forward(self, input_seq, input_lengths, hidden=None):
		# Convert word indexes to embeddings
		embedded = self.embedding(input_seq)
		# Pack padded batch of sequences for RNN module
		packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
		# Forward pass through GRU
		outputs, hidden = self.gru(packed, hidden)
		# Unpack padding
		outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
		# Sum bidirectional GRU outputs
		outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
		# Return output and final hidden state
		return outputs, hidden

# Luong attention layer
class Attn(nn.Module):
	def __init__(self, method, hidden_size):
		super(Attn, self).__init__()
		self.method = method
		if self.method not in ['dot', 'general', 'concat']:
			raise ValueError(self.method, "is not an appropriate attention method.")
		self.hidden_size = hidden_size
		if self.method == 'general':
			self.attn = nn.Linear(self.hidden_size, hidden_size)
		elif self.method == 'concat':
			self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
			self.v = nn.Parameter(torch.FloatTensor(hidden_size))
 
	def dot_score(self, hidden, encoder_output):
		return torch.sum(hidden * encoder_output, dim=2)
 
	def general_score(self, hidden, encoder_output):
		energy = self.attn(encoder_output)
		return torch.sum(hidden * energy, dim=2)
 
	def concat_score(self, hidden, encoder_output):
		energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
		return torch.sum(self.v * energy, dim=2)
 
	def forward(self, hidden, encoder_outputs):
		# Calculate the attention weights (energies) based on the given method
		if self.method == 'general':
			attn_energies = self.general_score(hidden, encoder_outputs)
		elif self.method == 'concat':
			attn_energies = self.concat_score(hidden, encoder_outputs)
		elif self.method == 'dot':
			attn_energies = self.dot_score(hidden, encoder_outputs)
 
		# Transpose max_length and batch_size dimensions
		attn_energies = attn_energies.t()
 
		# Return the softmax normalized probability scores (with added dimension)
		return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
	def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
		super(LuongAttnDecoderRNN, self).__init__()
 
		# Keep for reference
		self.attn_model = attn_model
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.n_layers = n_layers
		self.dropout = dropout
 
		# Define layers
		self.embedding = embedding
		self.embedding_dropout = nn.Dropout(dropout)
		self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
		self.concat = nn.Linear(hidden_size * 2, hidden_size)
		self.out = nn.Linear(hidden_size, output_size)
 
		self.attn = Attn(attn_model, hidden_size)
 
	def forward(self, input_step, last_hidden, encoder_outputs):
		# Note: we run this one step (word) at a time
		# Get embedding of current input word
		embedded = self.embedding(input_step)
		embedded = self.embedding_dropout(embedded)
		# Forward through unidirectional GRU
		rnn_output, hidden = self.gru(embedded, last_hidden)
		# Calculate attention weights from the current GRU output
		attn_weights = self.attn(rnn_output, encoder_outputs)
		# Multiply attention weights to encoder outputs to get new "weighted sum" context vector
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
		# Concatenate weighted context vector and GRU output using Luong eq. 5
		rnn_output = rnn_output.squeeze(0)
		context = context.squeeze(1)
		concat_input = torch.cat((rnn_output, context), 1)
		concat_output = torch.tanh(self.concat(concat_input))
		# Predict next word using Luong eq. 6
		output = self.out(concat_output)
		output = F.softmax(output, dim=1)
		# Return output and final hidden state
		return output, hidden

def maskNLLLoss(inp, target, mask):
	nTotal = mask.sum()
	crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
	loss = crossEntropy.masked_select(mask).mean()
	return loss, nTotal.item()

class Seq2SeqModel(nn.Module):
	def __init__(self, 
		device,
		SOS_token,
		num_words, 
		attn_model='dot', 
		hidden_size=500, 
		encoder_n_layers=2, decoder_n_layers=2, 
		dropout=0.1,
		learning_rate=0.0001,
		decoder_learning_ratio=5.0):

		super().__init__()

		self.device = device
		self.SOS_token = SOS_token

		embedding = nn.Embedding(num_words, hidden_size)
		self.encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
		self.decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, num_words, decoder_n_layers, dropout)

		self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
		self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

	def optimize(self, inputs, lengths, targets, mask, max_target_len,
		teacher_forcing_ratio=0.5, clip=50.0):
		self.encoder_optimizer.zero_grad()
		self.decoder_optimizer.zero_grad()

		encoder_outputs, encoder_hidden = self.encoder(inputs, lengths)

		decoder_input = torch.LongTensor([[self.SOS_token for _ in range(inputs.shape[1])]])
		decoder_input = decoder_input.to(self.device)

		decoder_hidden = encoder_hidden[:self.decoder.n_layers]
		use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

		loss = 0
		print_losses = []
		n_totals = 0

		if use_teacher_forcing:
			for t in range(max_target_len):
				decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
				decoder_input = targets[t].view(1, -1)
				mask_loss, nTotal = maskNLLLoss(decoder_output, targets[t], mask[t])
				loss += mask_loss
				print_losses.append(mask_loss.item() * nTotal)
				n_totals += nTotal
		else:
			for t in range(max_target_len):
				decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
				_, topi = decoder_output.topk(1)
				decoder_input = torch.LongTensor([[topi[i][0] for i in range(inputs.shape[1])]])
				decoder_input = decoder_input.to(self.device)
				mask_loss, nTotal = maskNLLLoss(decoder_output, targets[t], mask[t])
				loss += mask_loss
				print_losses.append(mask_loss.item() * nTotal)
				n_totals += nTotal

		loss.backward()
	 
		_ = nn.utils.clip_grad_norm_(self.encoder.parameters(), clip)
		_ = nn.utils.clip_grad_norm_(self.decoder.parameters(), clip)

		self.encoder_optimizer.step()
		self.decoder_optimizer.step()

		return sum(print_losses) / n_totals

	def evaluate(self, input_seq, input_length, max_length):
		encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
		decoder_hidden = encoder_hidden[:self.decoder.n_layers]
		decoder_input = torch.ones(1, 1, device=self.device, dtype=torch.long) * self.SOS_token
		all_tokens = torch.zeros([0], device=self.device, dtype=torch.long)
		all_scores = torch.zeros([0], device=self.device)
		for _ in range(max_length):
			decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
			decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
			all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
			all_scores = torch.cat((all_scores, decoder_scores), dim=0)
			decoder_input = torch.unsqueeze(decoder_input, 0)
		return all_tokens, all_scores