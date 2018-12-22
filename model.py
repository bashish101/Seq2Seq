import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderModule(nn.Module):
	def __init__(self, input_size, hidden_size, batch_first = True):
		super(EncoderModule, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.batch_first = batch_first

		self.embedding = nn.Embedding(input_size, hidden_size)
		self.rnn = nn.GRU(hidden_size, hidden_size, batch_first = batch_first)

	def forward(self, x, hidden = None):
		if hidden is  None:
			hidden = self.init_hidden(x.shape)

		x = self.embedding(x)
		x, hidden = self.rnn(x)
		return x, hidden
		
	def init_hidden(self, input_shape):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if self.batch_first:
			return torch.zeros(input_shape[0], 1, self.hidden_size, device = device)
		else:
			return torch.zeros(input_shape[1], 1, self.hidden_size, device = device)
		

class AttnDecoderModule(nn.Module):
	def __init__(self, 
		     hidden_size, 
		     output_size, 
		     batch_first = True, 
		     dropout_pb = 0.1, 
		     max_input_length = 50, 
		     max_output_length = 50):
		super(AttnDecoderModule, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.batch_first = True
		self.dropout_pb = dropout_pb
		self.max_input_length = max_input_length
		self.max_output_length = max_output_length
		
		self.embedding = nn.Embedding(output_size, hidden_size)

		self.attn_fc = nn.Linear(hidden_size, 1)
		self.attn_fc1 = nn.Linear(hidden_size * 2, hidden_size, bias = False)
		self.attn_fc2 = nn.Linear(hidden_size, hidden_size, bias = False)

		self.attn = nn.Linear(hidden_size * 2, max_input_length)

		self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)

		self.dropout = nn.Dropout(dropout_pb)
		self.rnn = nn.GRU(hidden_size, hidden_size, batch_first = True)
		self.out = nn.Linear(hidden_size, output_size)

		self.mode = 'train'

	def set_mode(self, mode = 'train'):
		self.mode = mode

	def set_wenc(self, encoder_out):
		self.wenc = self.attn_fc2(encoder_out)

	def step(self, x, hidden, encoder_out):
		"""
			x = decoder output at step (t - 1)
			hidden = hidden state of the decoder
			encoder_out = output sequence from encoder
		"""
		x = self.embedding(x)
		x = self.dropout(x)

		x_h = hidden.squeeze().unsqueeze(1)

		wh = self.attn_fc1(torch.cat([x, x_h], 2))
		attn_wts = F.softmax(self.attn_fc(wh + self.wenc), dim = 2)
		attn_wts = attn_wts.repeat(1, 1, encoder_out.shape[-1]) 
		attn_x = torch.sum(torch.mul(encoder_out, attn_wts), dim = 1, keepdim = True)

		# attn_wts = F.softmax(self.attn(torch.cat([x, x_h], 2)), dim = 2)
		# attn_x = torch.bmm(attn_wts, encoder_out)

		x = torch.cat((x, attn_x), 2)
		x = self.attn_combine(x)
		x = F.relu(x)

		x, hidden = self.rnn(x, hidden)
		x = F.log_softmax(self.out(x), dim = 2)
		return x, hidden, attn_wts
		
	def forward(self, encoder_out, hidden = None, y = None, teacher_forcing = True, ratio = 0.5):
		if hidden is  None:
			hidden = self.init_hidden(encoder_out.shape)

		if self.mode == 'train':
			step_size = self.max_output_length
		else:
			step_size = 1

		self.set_wenc(encoder_out)
		x = y[:, 0].unsqueeze(1)
		out = []
		attn = []
		for idx in range(step_size):
			if teacher_forcing == False and np.random.rand() < ratio:
				x = y[:, idx]
			else:
				x = x.detach()
	

			x = x.view(-1, 1)

			x, hidden, attn_wts = self.step(x, hidden, encoder_out)

			x = x.squeeze().unsqueeze(2)
			out.append(x)
			attn.append(attn_wts)
			top_v, top_i = x.squeeze().topk(1)
			x = top_i
		
		out = torch.cat(out, dim = 2)
		attn = torch.cat(attn, dim = 0)
		return out, hidden, attn	
		
	def init_hidden(self, input_shape):
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		if self.batch_first:
			return torch.zeros(input_shape[0], 1, self.hidden_size, device = device)
		else:
			return torch.zeros(input_shape[1], 1, self.hidden_size, device = device)


class Seq2SeqAttnNet(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, input_length, output_length):
		super(Seq2SeqAttnNet, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.input_length = input_length
		self.output_length = output_length

		self.encoder = EncoderModule(input_size, hidden_size)
		self.decoder = AttnDecoderModule(hidden_size, 
						 output_size, 
						 max_input_length = self.input_length, 
						 max_output_length = self.output_length)
		self.decoder_inp = None
		self.mode = 'train'
		self.teacher_forcing = True

	def set_mode(self, mode):
		self.mode = mode
		if mode == 'test':
			self.teacher_forcing = False

	def set_decoder_inp(self, decoder_inp):
		self.decoder_inp = decoder_inp

	def forward(self, x):
		x, hidden = self.encoder(x)
		x, _, attn_wts = self.decoder(x, hidden, self.decoder_inp, self.teacher_forcing)
		
		return x, attn_wts
