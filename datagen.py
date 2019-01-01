import re
import os
import unicodedata
import numpy as np
from collections import Counter

import torch

MAX_LENGTH = 50

class DataGen(object):
	def __init__(self,
		     data_path = 'fra-eng/fra.txt',
		     batch_size = 32, 
		     ratio = 0.75,
		     max_vocab_len = 5000,
		     input_vocab_path = 'input_vocab.txt',
		     target_vocab_path = 'target_vocab.txt'):
		self.data_path = data_path
		self.input_vocab_path = input_vocab_path
		self.target_vocab_path = target_vocab_path
		self.batch_size = batch_size
		self.ratio = ratio
		self.max_vocab_len = max_vocab_len
		
		self.train_data = None
		self.val_data = None

		self.input_length = None
		self.target_length = None
		self.input_size = None
		self.target_size = None

		self.data_size = None
		self.train_size = None
		self.val_size = None

		self.SOS = "<SOS>"
		self.EOS = "<EOS>"
		self.UNK = "UNK"

	def init_data(self, mode = 'train'):
		if mode != 'train' and os.path.exists(self.input_vocab_path) and os.path.exists(self.target_vocab_path):
			self.word2idx_source, self.idx2word_source = self.load_vocab(self.input_vocab_path, start_index = 1)
			self.word2idx_target, self.idx2word_target = self.load_vocab(self.input_vocab_path, start_index = 3)
			self.word2idx_target.update({self.SOS : 1, self.EOS : 2})
			self.idx2word_target.update({1 : self.SOS, 2 : self.EOS})

			self.input_size = len(self.word2idx_source)
			self.target_size = len(self.word2idx_target)
		else:	
			self.load_data()

	def tokenize(self, text):
		def unicode_to_ascii(text):
		    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

		text = unicode_to_ascii(text.lower().strip())
		text = re.sub(r"([.!|?])", r" \1 ", text)
		text = re.sub(r' +', ' ', text)
		word_list = [word for word in re.split(r"([-.\"',:? !\$#@~()*&\^%;\[\]/\\\+<>\n=])", text) \
			     if word != '' and word != ' ' and word != '\n']

		return word_list

	def create_vocab(self, 
			 text_list,
			 start_index = 1,
			 max_count = None,
			 save_path = None):
		counter = Counter([word for text in text_list for word in text])    
		top_words_with_counts = counter.most_common(max_count)

		vocab = [word for word, _ in top_words_with_counts]
		if max_count is not None:
			vocab = vocab[:max_count - 1]
		vocab += [self.UNK]

		word_to_idx = {word:index + start_index for index, word in enumerate(vocab)}		# 0 for pad
		idx_to_word = {index + start_index: word for index, word in enumerate(vocab)}

		if save_path is not None:
			with open(save_path, "w") as fp:
				fp.write("\n".join(vocab))

		return (word_to_idx, idx_to_word)

	def load_vocab(self, vocab_path, start_index = 1):
		vocab_list = []
		if os.path.exists(vocab_path):
			with open(vocab_path) as fp:
				vocab_list = [line.strip() for line in fp]

		word_to_idx = {word:index + start_index for index, word in enumerate(vocab_list)}	# 0 for pad
		idx_to_word = {index + start_index: word for index, word in enumerate(vocab_list)}
		return (word_to_idx, idx_to_word)

	def encode_source_text(self, text):
		return [self.word2idx_source.get(word, self.word2idx_source[self.UNK]) for word in text]

	def decode_source_text(self, text):
		return [self.idx2word_source[index] for index in text]

	def encode_target_text(self, text):
		return [self.word2idx_target.get(word, self.word2idx_target[self.UNK]) for word in text]

	def decode_target_text(self, text):
		return [self.idx2word_target[index] for index in text]

	def load_data(self, update_vocab = True, switch_input = True):
		select_prefixes = (
				   "i am ", "i m ",
				   "he is", "he s ",
				   "she is", "she s",
				   "you are", "you re ",
				   "we are", "we re ",
				   "they are", "they re "
				  )
		print('Loading data from path {} ...'.format(self.data_path))
		max_input_length = 0
		max_target_length = 0
		input_text_list = []
		target_text_list = []
		with open(self.data_path) as fp:
			for line in fp:
				input_text, target_text = line.strip().split('\t')

				if switch_input:
					input_text, target_text = target_text, input_text

				input_text = self.tokenize(input_text)
				target_text = self.tokenize(target_text)

				source_sentence = ' '.join(input_text)
				if len(input_text) > MAX_LENGTH or \
				   len(target_text) > MAX_LENGTH - 1: # or \
				   #not source_sentence.startswith(select_prefixes):
					continue

				if len(input_text) > max_input_length:
					max_input_length = len(input_text)
				if len(target_text) > max_target_length:
					max_target_length = len(target_text)

				input_text_list.append(input_text)
				target_text_list.append(target_text)

			self.data_size = len(input_text_list)
			self.train_size = int(self.data_size * self.ratio)
			self.val_size = self.data_size - self.train_size


		if not update_vocab and os.path.exists(self.input_vocab_path):
			self.word2idx_source, self.idx2word_source = self.load_vocab(self.input_vocab_path, start_index = 1)
		else:
			self.word2idx_source, self.idx2word_source = self.create_vocab(input_text_list, 
										       start_index = 1,			# padding
										       max_count = self.max_vocab_len - 1,
										       save_path = self.input_vocab_path)
		if not update_vocab and os.path.exists(self.target_vocab_path):
			self.word2idx_target, self.idx2word_target = self.load_vocab(self.input_vocab_path, start_index = 3)
		else:
			self.word2idx_target, self.idx2word_target = self.create_vocab(target_text_list, 
										       start_index = 3,			# padding, SOS, EOS
										       max_count = self.max_vocab_len - 3,
										       save_path = self.target_vocab_path)
		
		self.word2idx_target.update({self.SOS : 1, self.EOS : 2})
		self.idx2word_target.update({1 : self.SOS, 2 : self.EOS})

		self.input_size = len(self.word2idx_source) +  1
		self.target_size = len(self.word2idx_target) + 1

		self.input_length = min(MAX_LENGTH, max_input_length)
		self.target_length = min(MAX_LENGTH, max_target_length + 1)	# 1 for SOS or EOS case	
		self.train_data = [[], []]					# [Input text, target text]
		self.val_data = [[], []]					# [Input text, target text]
		
		for data_idx, (input_text, target_text) in enumerate(zip(input_text_list, target_text_list)):
			if data_idx < self.train_size:
				data = self.train_data
			else:
				data = self.val_data
			x = self.encode_source_text(input_text)
			y = self.encode_target_text(target_text)

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
			y = []							# Target text sequene

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

			device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

			yield [torch.tensor(x_arr, dtype = torch.long, device = device),
			       torch.tensor(decoder_inp_arr, dtype = torch.long, device = device),
			       torch.tensor(y_arr, dtype = torch.long, device = device)]
