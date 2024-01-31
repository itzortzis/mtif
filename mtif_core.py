
import time
import torch
import calendar
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from skimage import exposure
from torchmetrics import Dice
import torch.nn.functional as F
from torchmetrics import F1Score
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from torchmetrics.classification import BinaryF1Score




class Dataset(torch.utils.data.Dataset):
	def __init__(self, d, in_chnls, ts):
		self.dtst = d
		self.in_chnls = in_chnls
		self.ts = ts

	def __len__(self):
		return len(self.dtst)

	def __getitem__(self, index):
		obj = self.dtst[index, :, :, :]
		
		if self.ts == 1:
			x = torch.from_numpy(obj[:, :, :self.in_chnls-1])
			x_ = x
			y = torch.from_numpy(obj[:, :, self.in_chnls-1])
		elif self.ts == 2:
			x = torch.from_numpy(obj[:, :, 0])
			x_ = x
			y = torch.from_numpy(obj[:, :, self.in_chnls-1])
		else:
			x = torch.from_numpy(obj[:, :, 0])
			x_ = torch.from_numpy(obj[:, :, :self.in_chnls-1])
			y = torch.from_numpy(obj[:, :, self.in_chnls-1])
		
		
		return x, x_, y


class Training():

	def __init__(self, comps, params, paths):
		self.parameters = params
		self.components = comps
		self.paths = paths
		self.init()


	def init(self):

		self.init_components()
		self.init_parameters()
		self.init_paths()
		self.clean_dataset()
		self.split_dataset(self.d_start, self.d_end)
		self.normalize_sets()
		self.build_loaders()
		self.losses = np.zeros((self.epochs, 2))
		self.scores = np.zeros((self.epochs, 2))
		self.max_score = 0
		self.log = open("logs.txt", "a")  # append mode


		if self.device == 'cuda':
			print("Cuda available")
			self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
			self.model = self.model.to(self.device)
			self.t_model = self.t_model.to(self.device)


	def init_components(self):
		self.model     = self.components['model']
		self.t_model   = self.components['t_model']
		self.opt       = self.components['opt']
		self.loss_fn   = self.components['loss_fn']
		self.dataset = self.components['dataset']


	def init_parameters(self):
		self.thresh     = self.parameters['threshold']
		self.in_chnls   = self.parameters['in_channels']
		self.epochs     = self.parameters['epochs']
		self.dtst_name  = self.parameters['dtst_name']
		self.epoch_thr  = self.parameters['epoch_thresh']
		self.score_thr  = self.parameters['score_thresh']
		self.device     = self.parameters['device']
		self.batch_size = self.parameters['batch_size']
		self.clear_flag = self.parameters['clear_flag']
		self.d_start    = self.parameters['d_start']
		self.d_end      = self.parameters['d_end']
		self.inf_model  = self.parameters['inf_model_name']
		self.t_model_w  = self.parameters['t_model_name']
		self.alpha      = self.parameters['alpha']
		self.ts         = self.parameters['ts']
		# if self.ts:
		# 	print("xaxaxa")

	def init_paths(self):
		self.trained_models = self.paths['trained_models']
		self.metrics = self.paths['metrics']
		self.figures = self.paths['figures']


	def clean_dataset(self):
		if not self.clear_flag:
			return
		print(self.dataset.shape)
		temp = np.zeros(self.dataset.shape)
		idx = 0
		for i in range(len(self.dataset)):
			if np.sum(self.dataset[i, :, :, 0]) < 10000:
				continue
			temp[idx, :, :, :] = self.dataset[i, :, :, :]
			idx += 1
		self.dataset = temp[:idx, :, :, :]
		print(self.dataset.shape)


	def normalize(self, s):
		max_ = np.max(s[:, :, :, 0])
		min_ = np.min(s[:, :, :, 0])

		for i in range(len(s)):
			s[i, :, :, 0] = (s[i, :, :, 0] - min_) / (max_ - min_)
			s[i, :, :, 1] = np.where(s[i, :, :, 1] > 0, 1, 0)

		return s


	def normalize_sets(self):
		self.train_set = self.normalize(self.train_set)
		self.valid_set = self.normalize(self.valid_set)
		self.set_set = self.normalize(self.test_set)


	def split_dataset(self, d_start, d_end):
		self.dataset = self.dataset[d_start: d_end]
		rp = np.random.permutation(self.dataset.shape[0])
		self.dataset = self.dataset[rp]

		train_set_size = int(0.7 * len(self.dataset))
		valid_set_size = int(0.2 * len(self.dataset))
		test_set_size  = int(0.1 * len(self.dataset))

		train_start = 0
		train_end = train_set_size
		valid_start = train_set_size
		valid_end = valid_start + valid_set_size
		test_start = valid_end
		test_end = test_start + test_set_size

		self.train_set = self.dataset[train_start: train_end, :, :, :]
		self.valid_set = self.dataset[valid_start: valid_end, :, :, :]
		self.test_set = self.dataset[test_start: test_end, :, :, :]


	def build_loaders(self):
		train_set      = Dataset(self.train_set, self.in_chnls, self.ts)
		params         = {'batch_size': self.batch_size, 'shuffle': True}
		self.train_ldr = torch.utils.data.DataLoader(train_set, **params)
		valid_set      = Dataset(self.valid_set, self.in_chnls, self.ts)
		params         = {'batch_size': self.batch_size, 'shuffle': False}
		self.valid_ldr = torch.utils.data.DataLoader(valid_set, **params)
		test_set       = Dataset(self.test_set, self.in_chnls, self.ts)
		params         = {'batch_size': self.batch_size, 'shuffle': False}
		self.test_ldr  = torch.utils.data.DataLoader(test_set, **params)


	def create_threshold_activation(self):

		return torch.nn.Threshold(self.thresh, 0)


	def print_logo(self):
		print("""\
                    *    *    * *
         **  **     *        *
        *  **  *   ***   *  ***
        *      *    *    *   *
        *      *    **   **  *
                    """)


	def print_train_details(self):
		self.print_logo()
		print('You are about to train the model on ' + self.dtst_name)
		print('with the following details:')
		print('\t Training epochs: ', self.epochs)
		print('\t Epoch threshold: ', self.epoch_thr)
		print('\t Score threshold: ', self.score_thr)
		print('\t Trained models path: ', self.trained_models)
		print('\t Metrics path: ', self.metrics)
		print('\t Device: ', self.device)
		print()
		# option = input("Do you wish to continue? [Y/n]: ")
		return True or (option == 'Y' or option == 'y')



	# Main_training:
	# --------------
	# The supervisor of the training procedure.
	def main_training(self):
		if not (self.print_train_details()):
			return
		self.get_current_timestamp()
		print("Training is starting...")
		start_time = time.time()
		self.metric = F1Score(average='micro', num_classes=2)
		self.metric.to(self.device)
		for epoch in tqdm(range(self.epochs)):
			if self.ts == 3:
					# print("Enhanced training: Teacher - Student model")
				tr_score, tr_loss = self.enhanced_epoch_training()
			else:
				# print("Simple training: Teacher - Student model")
				tr_score, tr_loss = self.epoch_training()
			# tr_score, tr_loss = self.epoch_training()
			vl_score, vl_loss = self.epoch_validation()

			self.losses[epoch, 0] = tr_loss
			self.losses[epoch, 1] = vl_loss
			self.scores[epoch, 0] = tr_score
			self.scores[epoch, 1] = vl_score

			print()
			print("\t Training - Score: ", tr_score, " Loss: ", tr_loss)
			print("\t Validation: - Score: ", vl_score, " Loss: ", vl_loss)
			print()
			self.save_model_weights(epoch, vl_score, vl_loss)
		self.exec_time = time.time() - start_time
		print("Total execution time: ", self.exec_time, " seconds")
		self.test_set_score = self.inference()
		self.log_line = str(self.test_set_score) + " " + self.log_line
		self.save_metrics()
		self.update_log()


	def update_log(self):
		self.log.write(self.log_line)
		self.log.close()


	# Get_current_timestamp:
	# ----------------------
	# This function calculates the current timestamp that is
	# used as unique id for saving the experimental details
	def get_current_timestamp(self):
		current_GMT = time.gmtime()
		self.timestamp = calendar.timegm(current_GMT)


	# Save_model_weights:
	# -------------------
	# This funtion saves the model weights during training
	# procedure, if some requirements are satisfied.
	#
	# --> epoch: current epoch of the training
	# --> score: current epoch score value
	# --> loss: current epoch loss value
	def save_model_weights(self, epoch, score, loss):

		if score > self.max_score and epoch > self.epoch_thr:
			path_to_model = self.trained_models + self.dtst_name
			path_to_model += "_" + str(self.timestamp) + ".pth"
			torch.save(self.model.state_dict(), path_to_model)
			self.model_dict = self.model.state_dict()
			log = str(epoch) + " " + str(score) + " " + path_to_model + "\n"
			self.log_line = log
			self.max_score = score


	# Prepare_data:
	# -------------
	# Given x and y tensors, this function applies some basic
	# transformations/changes related to dimensions, data types,
	# and device.
	#
	# --> x: tensor containing a batch of input images
	# --> y: tensor containing a batch of annotation masks
	# <-- x, y: the updated tensors
	def prepare_data(self, x, x_, y):
		# print('This is x size: ', x.size())
		# print('This is x_ size: ', x_.size())
		if len(x.size()) < 4:
			x = torch.unsqueeze(x, 1)
		else:
			x = x.movedim(2, -1)
			x = x.movedim(1, 2)
		
		if len(x_.size()) < 4:
			x_ = torch.unsqueeze(x_, 1)
		else:
			x_ = x_.movedim(2, -1)
			x_ = x_.movedim(1, 2)

		# print('This is x size: ', x.size())
		# print('This is x_ size: ', x_.size())
		x = x.to(torch.float32)
		x_ = x_.to(torch.float32)
		y = y.to(torch.int64)

		x = x.to(self.device)
		x_ = x_.to(self.device)
		y = y.to(self.device)

		return x, x_, y
		

	# Epoch_training:
	# ---------------
	# This function is used for implementing the training
	# procedure during a single epoch.
	#
	# <-- epoch_score: performance score achieved during
	#                  the training
	# <-- epoch_loss: the loss function score achieved during
	#                 the training
	def epoch_training(self):
		self.model.train(True)
		current_score = 0.0
		current_loss = 0.0
		self.metric.reset()
		# print("Simple epoch training...")

		step = 0
		for x, x_, y in self.train_ldr:
			x, x_, y = self.prepare_data(x, x_, y)
			step += 1
			self.opt.zero_grad()
			outputs = self.model(x)
			# print(y, outputs)
			loss = self.loss_fn(outputs, y)
			# print("Simple training loss: ", loss)
			loss.backward()
			self.opt.step()
			preds = torch.argmax(outputs, dim=1)
			score = self.metric.update(preds, y)
			current_loss  += loss * self.train_ldr.batch_size

		epoch_score = self.metric.compute()
		self.metric.reset()
		epoch_loss  = current_loss / len(self.train_ldr.dataset)

		return epoch_score.item(), epoch_loss.item()

	
	# Enhanced_epoch_training:
	# ---------------
	# This function is used for implementing the enhanced training
	# (teacher - student approach) procedure during a single epoch.
	#
	# <-- epoch_score: performance score achieved during
	#                  the training
	# <-- epoch_loss: the loss function score achieved during
	#                 the training
	def enhanced_epoch_training(self):
		
		self.model.train(True)
		self.t_model.train(False)
		current_score = 0.0
		current_loss = 0.0
		self.metric.reset()
		self.load_teacher_model()
		# self.t_model.train(True)

		# print("Enhanced epoch training...")
		step = 0
		for x, x_, y in self.train_ldr:
			x, x_, y = self.prepare_data(x, x_, y)
			step += 1
			self.opt.zero_grad()
			with torch.no_grad():
				t_outputs = self.t_model(x_)
			outputs = self.model(x)
			s_loss = self.loss_fn(outputs, y)
			# print("Enhanced s_loss: ", s_loss)
			t_loss = self.loss_fn(outputs, t_outputs.softmax(dim=1))
			# print("Enhanced t_loss: ", t_loss)
			loss = self.alpha * s_loss + (1 - self.alpha) * t_loss
			# print(loss, t_loss, s_loss)
			loss.backward()
			self.opt.step()
			preds = torch.argmax(outputs, dim=1)
			score = self.metric.update(preds, y)
			current_loss  += loss * self.train_ldr.batch_size

		epoch_score = self.metric.compute()
		self.metric.reset()
		epoch_loss  = current_loss / len(self.train_ldr.dataset)

		return epoch_score.item(), epoch_loss.item()


	# Epoch_validation:
	# ---------------
	# This function is used for implementing the validation
	# procedure during a single epoch.
	#
	# <-- epoch_score: performance score achieved during
	#                  the validation
	# <-- epoch_loss: the loss function score achieved during
	#                 the validation
	def epoch_validation(self):
		self.model.train(False)
		self.t_model.train(False)
		current_score = 0.0
		current_loss = 0.0
		self.metric.reset()

		for x, x_, y in self.valid_ldr:
			x, x_, y = self.prepare_data(x, x_, y)

			with torch.no_grad():
				outputs = self.model(x)
				loss = self.loss_fn(outputs, y)

			preds = torch.argmax(outputs, dim=1)
			score = self.metric.update(preds, y)
			current_loss  += loss * self.train_ldr.batch_size

		epoch_score = self.metric.compute()
		epoch_loss  = current_loss / len(self.valid_ldr.dataset)

		return epoch_score.item(), epoch_loss.item()


	# Inference:
	# ----------
	# Applies inference to the testing set extracted from
	# the input dataset during the initialization phase
	#
	# <-- test_set_score: the score achieved by the trained model
	def inference(self):
		self.model.load_state_dict(self.model_dict)
		self.model.eval()
		current_score = 0.0
		current_loss = 0.0
		self.metric.reset()
		for x, x_, y in self.test_ldr:
			x, x_, y = self.prepare_data(x, x_, y)

			with torch.no_grad():
				outputs = self.model(x)

			preds = torch.argmax(outputs, dim=1)
			score = self.metric.update(preds, y)

		test_set_score = self.metric.compute()
		self.metric.reset()
		return test_set_score.item()


	def ext_inference(self, set_ldr):
		path_to_model = self.trained_models + self.inf_model
		self.model.load_state_dict(torch.load(path_to_model))
		self.model.eval()
		current_score = 0.0
		self.metric.reset()
		for x, x_, y in set_ldr:
			x, x_, y = self.prepare_data(x, x_, y)

			with torch.no_grad():
				outputs = self.model(x)

			preds = torch.argmax(outputs, dim=1)
			self.metric.update(preds, y)

		test_set_score = self.metric.compute()
		self.metric.reset()
		return test_set_score.item()


	def load_teacher_model(self):
		path_to_model = self.trained_models + self.t_model_w
		self.t_model.load_state_dict(torch.load(path_to_model))
		self.t_model.eval()

	def save_metrics(self):
		postfix = self.dtst_name + "_" + str(self.timestamp)
		np.save(self.metrics + "scores_" + postfix, self.scores)
		np.save(self.metrics + "losses_" + postfix, self.losses)
		self.save_figures()


	def save_figures(self):
		postfix = self.dtst_name + "_" + str(self.timestamp) + ".png"
		plt.figure()
		plt.plot(self.scores[:, 0])
		plt.savefig(self.figures + "train_s_" + postfix)
		plt.figure()
		plt.plot(self.scores[:, 1])
		plt.savefig(self.figures + "valid_s_" + postfix)

		plt.figure()
		plt.plot(self.losses[:, 0])
		plt.savefig(self.figures + "train_l_" + postfix)
		plt.figure()
		plt.plot(self.losses[:, 1])
		plt.savefig(self.figures + "valid_l_" + postfix)




class CV_Training(Training):


	def init(self):
		super().init_components()
		self.init_parameters()
		super().init_paths()
		self.init_cv_models()
		super().clean_dataset()
		self.split_dataset(self.d_start, self.d_end)
		# self.normalize_sets()
		self.build_test_loader()
		self.losses = np.zeros((self.k, self.epochs, 2))
		self.scores = np.zeros((self.k, self.epochs, 2))
		self.log = open("logs.txt", "a")  # append mode
		self.init_model_dict = self.model.state_dict()

		if self.device == 'cuda':
			print("Cuda available")
			self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
			self.model = self.model.to(self.device)
			self.t_model = self.t_model.to(self.device)
		
		self.metric = F1Score(task="binary", num_classes=2)
		self.metric.to(self.device)
		


	def init_cv_models(self):
		self.cvm = {}
		self.cvm['max_score'] = 0
		self.cvm['model_dicts'] = {}
		self.cvm['max_epoch_score'] = {}
		self.cvm['fold_best_epoch'] = {}


	def init_parameters(self):
		super().init_parameters()
		self.k          = self.parameters['k']


	def normalize_sets(self):
		self.train_set = self.normalize(self.train_set)
		self.set_set = self.normalize(self.test_set)


	def split_dataset(self, d_start, d_end):
		self.dataset = self.dataset[d_start: d_end]
		# rp = np.random.permutation(self.dataset.shape[0])
		# self.dataset = self.dataset[rp]

		train_set_size = int(0.8 * len(self.dataset))
		test_set_size = int(0.2 * len(self.dataset))

		train_start = 0
		train_end = train_set_size
		test_start = train_set_size
		test_end = test_start + test_set_size

		self.train_set = self.dataset[train_start: train_end, :, :, :]
		self.test_set = self.dataset[test_start: test_end, :, :, :]


	def cv_training_split(self, f_id, f_size):
		if self.k == 1:
			f_size = int(0.2 * len(self.train_set))
		# print(f_id, f_size)
		valid_fold_start = f_id * f_size
		valid_fold_end = valid_fold_start + f_size
		valid_fold = np.zeros((f_size, 256, 256, self.train_set.shape[3]))
		valid_fold = self.train_set[valid_fold_start: valid_fold_end, :, :, :]
		train_fold = np.zeros((len(self.train_set) - f_size, 256, 256, self.train_set.shape[3]))
		# print(self.train_set.shape)

		pre_end =  valid_fold_start
		post_start = valid_fold_end
		train_fold[:pre_end, :, :, :] = self.train_set[:pre_end, :, :, :]
		train_fold[pre_end:, :, :, :] = self.train_set[post_start:, :, :, :]

		self.train_fold = train_fold
		self.valid_fold = valid_fold


	def cv_build_loaders(self):
		# print(self.train_fold.shape)
		train_set      = Dataset(self.train_fold, self.in_chnls, self.ts)
		params         = {'batch_size': self.batch_size, 'shuffle': True}
		self.train_ldr = torch.utils.data.DataLoader(train_set, **params)
		valid_set      = Dataset(self.valid_fold, self.in_chnls, self.ts)
		params         = {'batch_size': self.batch_size, 'shuffle': False}
		self.valid_ldr = torch.utils.data.DataLoader(valid_set, **params)


	def build_test_loader(self):
		test_set       = Dataset(self.test_set, self.in_chnls, self.ts)
		params         = {'batch_size': self.batch_size, 'shuffle': False}
		self.test_ldr  = torch.utils.data.DataLoader(test_set, **params)


	# Cross_validation:
	# -----------------
	# The supervisor of the training procedure using a k-fold
	# cross-validation schema
	def cross_validation(self):
		if not (self.print_train_details()):
			return
		self.get_current_timestamp()
		print("Training is starting...")
		start_time = time.time()
		fold_size = len(self.train_set) // self.k
		self.cvm['max_score'] = 0
		for cv_i in range(self.k):
			self.model.load_state_dict(self.init_model_dict)
			self.cvm['model_dicts'][cv_i] = self.model.state_dict()
			self.loss_fn = torch.nn.CrossEntropyLoss()
			self.opt = torch.optim.Adam(self.model.parameters(), lr=0.01)
			self.cv_training_split(cv_i, fold_size)
			self.cv_build_loaders()
			self.cvm['max_epoch_score'][cv_i] = 0
			print()
			print("Training using ", str(cv_i), " fold for validation.")
			p_bar = tqdm(range(self.epochs), colour='green')
			for epoch in p_bar:
				if self.ts == 3:
					# print("Enhanced training: Teacher - Student model")
					tr_score, tr_loss = self.enhanced_epoch_training()
				else:
					# print("Simple training: Teacher - Student model")
					tr_score, tr_loss = self.epoch_training()
				# tr_score, tr_loss = self.epoch_training()
				vl_score, vl_loss = self.epoch_validation()
				self.save_ls(tr_loss, vl_loss, tr_score, vl_score, epoch, cv_i)
				s = self.results_to_str(tr_score, tr_loss, vl_score, vl_loss)
				p_bar.set_description(s)
				self.keep_model_weights(epoch, vl_score, cv_i)

			test_set_score = self.inference(cv_i)
			self.save_best_model(test_set_score, cv_i)
		self.exec_time = time.time() - start_time
		print("Total execution time: ", self.exec_time, " seconds")
		self.save_model_weights(self.cvm['best_model_epoch'], self.cvm['max_score'])
		self.save_metrics()
		self.update_log()


	def save_ls(self, tr_loss, vl_loss, tr_score, vl_score, epoch, cv_i):
		# print("Saving ls")
		self.losses[cv_i, epoch, 0] = tr_loss
		self.losses[cv_i, epoch, 1] = vl_loss
		self.scores[cv_i, epoch, 0] = tr_score
		self.scores[cv_i, epoch, 1] = vl_score


	def save_best_model(self, test_set_score, cv_i):

		if test_set_score > self.cvm['max_score']:
			self.cvm['max_score'] = test_set_score
			self.cvm['best_model'] = self.model.state_dict()
			self.cvm['best_model_epoch'] = self.cvm['fold_best_epoch'][cv_i]
			self.cvm['losses'] = self.losses[cv_i, :, :]
			self.cvm['scores'] = self.scores[cv_i, :, :]


	def keep_model_weights(self, epoch, score, cv_i):
		
		if score > self.cvm['max_epoch_score'][cv_i] and epoch > self.epoch_thr:
			# print("Keep model weights")
			self.cvm['fold_best_epoch'][cv_i] = epoch
			self.cvm['max_epoch_score'][cv_i] = score
			self.cvm['model_dicts'][cv_i] = self.model.state_dict()
		# else:
		# 	print("Keep: ", score, self.cvm['max_epoch_score'][cv_i], epoch, self.epoch_thr)


	def print_scores(self, tr_score, tr_loss, vl_score, vl_loss):
		print()
		print("\t Training - Score: ", tr_score, " Loss: ", tr_loss)
		print("\t Validation: - Score: ", vl_score, " Loss: ", vl_loss)
		print()

	
	def results_to_str(self, tr_score, tr_loss, vl_score, vl_loss):
		s = "TRS: " + str(round(tr_score, 3)) 
		s += ", TRL: " + str(round(tr_loss, 3)) 
		s += ", VLS: " + str(round(vl_score, 3)) 
		s += ", VLL: " + str(round(vl_loss,3))

		return s


	def save_model_weights(self, epoch, score):

		path_to_model = self.trained_models + self.dtst_name
		path_to_model += "_" + str(self.timestamp) + ".pth"
		torch.save(self.cvm['best_model'], path_to_model)
		log = str(self.k) + " " + str(self.ts) + " "
		log += str(self.cvm['best_model_epoch']) + " "
		log += str(self.cvm['max_score'])
		log += " " + path_to_model + " "
		log += str(self.d_start) + " " + str(self.d_end) + "\n"
		self.log_line = log
		# self.max_score = score



	# Inference:
	# ----------
	# Applies inference to the testing set extracted from
	# the input dataset during the initialization phase
	#
	# <-- test_set_score: the score achieved by the trained model
	def inference(self, cv_i):
		self.model.load_state_dict(self.cvm['model_dicts'][cv_i])
		self.model.eval()
		current_score = 0.0
		current_loss = 0.0
		self.metric.reset()
		idx = 0
		for x, x_, y in self.test_ldr:
			x, x_, y = self.prepare_data(x, x_, y)

			with torch.no_grad():
				outputs = self.model(x)

			preds = torch.argmax(outputs, dim=1)
			# preds = cpu().detach().numpy()
			# print(preds.size())
			score = self.metric.update(preds, y)
			# self.save_sample_preds(x, preds, y, idx)
			idx += 1

		test_set_score = self.metric.compute()
		self.metric.reset()

		return test_set_score.item()

	def save_sample_preds(self, x, preds, y, name):
		x = x.cpu().detach().numpy()
		preds = preds.cpu().detach().numpy()
		y = y.cpu().detach().numpy()
		f, axarr = plt.subplots(1, 3)
		axarr[0].imshow(x[0, 0, :, :])
		axarr[1].imshow(preds[0, :, :])
		axarr[2].imshow(y[0, :, :])
		plt.savefig(str(name) + ".png")
		plt.close()

     
	def save_figures(self):
		postfix = self.dtst_name + "_" + str(self.timestamp) + ".png"
		plt.figure()
		plt.plot(self.cvm['scores'][:, 0])
		plt.savefig(self.figures + "train_s_" + postfix)
		plt.figure()
		plt.plot(self.cvm['scores'][:, 1])
		plt.savefig(self.figures + "valid_s_" + postfix)

		plt.figure()
		plt.plot(self.cvm['losses'][:, 0])
		plt.savefig(self.figures + "train_l_" + postfix)
		plt.figure()
		plt.plot(self.cvm['losses'][:, 1])
		plt.savefig(self.figures + "valid_l_" + postfix)
