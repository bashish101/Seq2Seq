import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from datagen import DataGen
from model import Seq2SeqAttnNet

HIDDEN_SIZE = 256
def train(batch_size, max_sequence_length, data_path, save_path, resume_flag = False):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	datagen = DataGen(data_path = data_path, 
			  batch_size = batch_size,
			  max_seq_len = max_sequence_length)
	datagen.init_data()

	input_size = datagen.input_size		# Input vocab size
	hidden_size = HIDDEN_SIZE
	output_size = datagen.target_size	# Target vocab size
	input_length = datagen.input_length
	output_length = datagen.target_length

	print('Input vocab size: ', input_size)
	print('Target vocab size: ', output_size)
	print('Input text length: ', input_length)
	print('Target text length: ', output_length)

	model = Seq2SeqAttnNet(input_size, 
			       hidden_size, 
			       output_size, 
			       input_length, 
			       output_length).to(device)

	criterion = nn.NLLLoss()
	encoder_opt = optim.Adamax(model.encoder.parameters())
	decoder_opt = optim.Adamax(model.decoder.parameters())

	def checkpoint(model, epoch, chk_path = 'seq2seq_chk.pth'):
		torch.save(model.state_dict(), chk_path)

	print (model)
	print ('Model built successfully...')
	model_parameters = filter(lambda p: p.requires_grad, model.parameters())
	print('Total params: {}'.format(sum([np.prod(p.size()) for p in model_parameters])))
	
	train_steps = datagen.train_size // batch_size
	val_steps = datagen.val_size // batch_size
	epochs = 1000

	if resume_flag and os.path.exists('seq2seq_chk.pth'):
		model.load_state_dict(torch.load('seq2seq_chk.pth'))


	train_datagen = datagen.get_batch(mode = 'train')
	val_datagen = datagen.get_batch(mode = 'val')
	for epoch in range(epochs):
		train_loss = 0
		model.set_mode('train')
		for step_idx in range(train_steps):
			x, decoder_inp, y = next(train_datagen)

			model.set_decoder_inp(decoder_inp)

			pred, _ = model(x)
			loss = criterion(pred, y)

			encoder_opt.zero_grad()
			decoder_opt.zero_grad()
			train_loss += loss.item()
			loss.backward()		
			encoder_opt.step()	
			decoder_opt.step()
			# print("===> Step {} : Loss: {:.4f}".format(step_idx, 
			#					   loss.item()))

		print ("===> Epoch {} Complete: Avg. Training Loss: {:.4f}".format(epoch, 
										   train_loss / train_steps))
		val_loss = 0
		model.set_mode('val')
		with torch.no_grad():
			for step_idx in range(val_steps):
				x, _, y = next(val_datagen)

				pred, _ = model(x)
				loss = criterion(pred, y)

				val_loss += loss.item()
		print ("===> Epoch {} Complete: Avg. validation Loss: {:.4f}".format(epoch, 
									  	     val_loss / val_steps))

		checkpoint(model, epoch, save_path)
	torch.save(model, save_path)

def test(save_path, max_sequence_length = 50, data_path = None, weights_only = False):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	datagen = DataGen(data_path = data_path, 
		  	 batch_size = batch_size,
			 max_seq_len = max_sequence_length)

	if not weights_only:
		datagen.init_data(mode = 'test')
		model = torch.load(save_path)
	else:
		datagen.init_data()

		input_size = datagen.input_size		# Input vocab size
		hidden_size = HIDDEN_SIZE
		output_size = datagen.target_size	# Target vocab size
		input_length = datagen.input_length
		output_length = datagen.target_length

		model = Seq2SeqAttnNet(input_size, 
				       hidden_size, 
				       output_size, 
				       input_length, 
				       output_length).to(device)

		model.load_state_dict(torch.load(save_path))


	model.set_mode('test')	

	inp_text = input('Enter text in english:')
	inp_text = datagen.tokenize(inp_text)
	inp_text = datagen.encode_source_text(inp_text)

	inp_text = torch.tensor(np.array([inp_text]), dtype = torch.long, device = device)

	encoder_out, hidden = model.encoder(inp_text)
	
	dec_word_list = []
	attn_wts_list = []
	prev_word = datagen.encode_target_text([datagen.SOS])
	prev_word = torch.tensor(np.array([prev_word]), dtype = torch.long, device = device)
	
	count = 0
	dec_word = None
	while prev_word != datagen.word2idx_target[datagen.EOS] and count < output_length:
		x, hidden, attn_wts = model.decoder(encoder_out, hidden, prev_word, False)

		top_v, top_i = x.squeeze().topk(1)
		prev_word = top_i
		
		attn_wts_list.append(attn_wts)	
		dec_word_list.append(prev_word)

		prev_word = prev_word.unsqueeze(1)
		count += 1

	out_text_enc = np.array([x for x in dec_word_list if x != 0])
	out_text = datagen.decode_target_text(out_text_enc)
	out_text = ' '.join(out_text)
	print(out_text)
	
if __name__  == '__main__':
	parser = argparse.ArgumentParser(description = 'Seq2SeqAttnNet Parameters')
	
	parser.add_argument('-m',
			    '--exec_mode',
			    default = 'train',
			    help = 'Execution mode',
			    choices = ['train', 'val', 'test'])
	parser.add_argument('-b',
			    '--batch_size',
			    type = int,
			    default = 32)
	parser.add_argument('-l',
			    '--max_sequence_length',
			    type = int,
			    default = 50)
	parser.add_argument('-d',
			    '--data_path',
			    default = 'rig-veda/sans_eng.txt')
	parser.add_argument('-s',
			    '--save_path',
			    default = 'seq2seq_chk.pth')

	arguments = parser.parse_args()
	mode = arguments.exec_mode
	batch_size = arguments.batch_size
	max_sequence_length = arguments.max_sequence_length
	data_path = arguments.data_path
	save_path = arguments.save_path

	if mode == 'train':
		train(batch_size, max_sequence_length, data_path, save_path, resume_flag = False)
	else:
		test(save_path, max_sequence_length, data_path, weights_only = True)	
