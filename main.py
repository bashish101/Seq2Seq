import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from datagen import DataGen
from model import Seq2SeqAttnNet

def train(batch_size, data_path, save_path, resume_flag = False):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	datagen = DataGen(data_path = data_path, 
			  batch_size = batch_size)
	datagen.load_data()

	input_size = datagen.input_size		# Input vocab size
	hidden_size = 256
	output_size = datagen.target_size	# Target vocab size
	input_length = datagen.input_length
	output_length = datagen.target_length

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

	if resume_flag:
		model.load_state_dict(torch.load(save_path))


	train_datagen = datagen.get_batch(mode = 'train')
	val_datagen = datagen.get_batch(mode = 'val')
	for epoch in range(epochs):
		train_loss = 0
		model.set_mode('train')
		for batch_idx in range(train_steps):
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
			print(loss.item())


		print ("===> Epoch {} Complete: Avg. Training Loss: {:.4f}".format(epoch, 
										   train_loss / train_steps))
		val_loss = 0
		model.set_mode('val')
		with torch.no_grad():
			for batch_idx in range(val_steps):
				x, _, y = next(val_datagen)

				pred, _ = model(x)
				loss = criterion(pred, y)

				val_loss += loss.item()
		print ("===> Epoch {} Complete: Avg. validation Loss: {:.4f}".format(epoch, 
									  	     val_loss / val_steps))

		checkpoint(model, epoch, save_path)

def test(batch_size, data_path, save_path, count):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	pass
	
	
if __name__  == '__main__':
	parser = argparse.ArgumentParser(description = 'Seq2SeqAttnNet Parameters')
	
	parser.add_argument('-m',
			    '--exec_mode',
			    default = 'train',
			    help = 'Execution mode',
			    choices = ['train', 'test'])
	parser.add_argument('-b',
			    '--batch_size',
			    default = 32)
	parser.add_argument('-c',
			    '--count',
			    default = 12)
	parser.add_argument('-d',
			    '--data_path',
			    default = 'fra-eng/fra.txt')
	parser.add_argument('-s',
			    '--save_path',
			    default = 'seq2seq_chk.pth')

	arguments = parser.parse_args()
	mode = arguments.exec_mode
	batch_size = arguments.batch_size
	count = arguments.count
	data_path = arguments.data_path
	save_path = arguments.save_path

	if mode == 'train':
		train(batch_size, data_path, save_path, resume_flag = True)
	else:
		test(batch_size, data_path, save_path, count)	
