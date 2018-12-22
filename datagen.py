import re
import os
import unicodedata
import numpy as np

import torch

MAX_LENGTH = 10

class DataGen(object):
	def __init__(self,
		     data_path = 'fra-eng/fra.txt',
		     batch_size = 32, 
		     ratio = 0.75,
		     input_vocab_path = 'input_vocab.txt',
		     target_vocab_path = 'target_vocab.txt'):
		self.data_path = data_path
		self.input_vocab_path = input_vocab_path
		self.target_vocab_path = target_vocab_path
		self.batch_size = batch_size
		self.ratio = ratio
		
		self.train_data = None
		self.val_data = None

		self.input_length = None
		self.output_length = None
		self.input_size = None
		self.target_size = None

		self.data_size = None
		self.train_size = None
		self.val_size = None
		
		self.SOS = "<SOS>"
		self.EOS = "<EOS>"

	def tokenize(self, text):
		def unicode_to_ascii(text):
		    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

		text = unicode_to_ascii(text.lower().strip())
		text = re.sub(r"([.!?])", r" \1 ", text)
		text = re.sub(r"[^a-zA-Z.!?]+", r" ", text)
		text = re.sub(r' +', ' ', text)

		return [word for word in text.split(' ')]



	def load_data(self):
		select_prefixes = (
				   "i am ", "i m ",
				   "he is", "he s ",
				   "she is", "she s",
				   "you are", "you re ",
				   "we are", "we re ",
				   "they are", "they re "
				  )

		input_texts  = []
		decoder_input_texts = []
		target_texts = []

		self.input_size = 0
		self.target_size = 0
		self.word2idx_source = {}
		self.idx2word_source = {}
		self.word2idx_target = {self.SOS : 1, self.EOS : 2}
		self.idx2word_target = {1 : self.SOS, 2 : self.EOS}
		self.input_size = 1						# 0 is reserved for padding
		self.target_size = len(self.idx2word_target) + 1

		max_input_length = 0
		max_target_length = 0
		print('Loading data from path {} ...'.format(self.data_path))
		with open(self.data_path) as fp:
			for line in fp:
				input_text, target_text = line.strip().split('\t')
				input_text = self.tokenize(input_text)
				target_text = self.tokenize(target_text)

				source_sentence = ' '.join(input_text)
				if len(input_text) > MAX_LENGTH or \
				   len(target_text) > MAX_LENGTH - 1 or \
				   not source_sentence.startswith(select_prefixes):
					continue

				for word in input_text:
					if word not in self.word2idx_source:
						self.word2idx_source[word] = self.input_size
						self.idx2word_source[self.input_size] = word
						self.input_size += 1

				for word in target_text:
					if word not in self.word2idx_target:
						self.word2idx_target[word] = self.target_size
						self.idx2word_target[self.target_size] = word
						self.target_size += 1

				if len(input_text) > max_input_length:
					max_input_length = len(input_text)
				if len(target_text) > max_target_length:
					max_target_length = len(target_text)

				input_texts.append(input_text)
				target_texts.append(target_text)

			self.data_size = len(input_texts)
			self.train_size = int(self.data_size * self.ratio)
			self.val_size = self.data_size - self.train_size

		with open(self.input_vocab_path, 'w') as fp:
			input_vocab = self.word2idx_source.keys()
			print('Input vocab size {}'.format(len(input_vocab)))
			for word in input_vocab:
				fp.write(word + '\n')

		with open(self.target_vocab_path, 'w') as fp:
			target_vocab = self.word2idx_target.keys()
			print('Target vocab size {}'.format(len(target_vocab)))
			for word in target_vocab:
				fp.write(word + '\n')

		self.input_length = min(MAX_LENGTH, max_input_length)
		self.target_length = min(MAX_LENGTH, max_target_length + 1)	# 1 for SOS or EOS case	
		self.train_data = [[], []]					# [Input text, target text]
		self.val_data = [[], []]					# [Input text, target text]
		
		for data_idx, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
			if data_idx < self.train_size:
				data = self.train_data
			else:
				data = self.val_data
			x = [self.word2idx_source[word] for word in input_text]
			y = [self.word2idx_target[word] for word in target_text]

			data[0].append(x)					
			data[1].append(y)

		print('Data load complete! Total data length is {}.'.format(self.data_size))

	def pad(self, data, fixed = True, maxlen = None):
		if fixed == False:
			maxlen = max([len(features) for features in data])
		elif maxlen is None:
			maxlen = MAX_LENGTH
        
		paddings = [[0] * (maxlen - len(features)) for features in data]      
		data = [feat_list[:maxlen] + padding for feat_list, padding in zip(data, paddings)]
		return data

	def get_batch(self, mode = 'train'):
		if mode == 'train':
			data = self.train_data
		else:
			data = self.val_data
	
		batch_index = 0
		while True:
			x = []							# Input text sequence
			decoder_inp = []					# Teacher forcing text sequence
			y = []							# Target text sequence

			for pos in range(self.batch_size):
				data_idx = batch_index * self.batch_size + pos

				x.append(data[0][data_idx][:self.input_length])
				decoder_inp.append([self.word2idx_target[self.SOS]] + data[1][data_idx][:self.target_length - 1])
				y.append(data[1][data_idx][:self.target_length - 1] + [self.word2idx_target[self.EOS]])

				if data_idx == len(data):
					batch_index = 0
					if mode == 'train':
						idx_list = list(range(len(data[0])))
						np.random.shuffle(idx_list)
						data = [[data_item[idx] for idx in idx_list] for data_item in data]
				

			x_arr = np.array(self.pad(x, fixed = True, maxlen = self.input_length))
			decoder_inp_arr = np.array(self.pad(decoder_inp, fixed = True, maxlen = self.target_length))
			y_arr = np.array(self.pad(y, fixed = True, maxlen = self.target_length))

			yield [torch.from_numpy(x_arr).long(),
			       torch.from_numpy(decoder_inp_arr).long(),
			       torch.from_numpy(y_arr).long()]
