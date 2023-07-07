import cv2
import time
import math
import torch
import calendar
import numpy as np
from tqdm import tqdm
from skimage import exposure
from torchmetrics import Dice
import torch.nn.functional as F
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

class Dataset_g(torch.utils.data.Dataset):
	def __init__(self, d):
		self.dtst = d

	def __len__(self):
		return len(self.dtst)

	def __getitem__(self, index):
		obj = self.dtst[index, :, :, :]
		e = np.where(obj[:, :, 0] < 0.3)
		t = self.equalize_hist(obj, e)

		x = torch.from_numpy(obj[:, :, 0])
		y = torch.from_numpy(obj[:, :, 1])

		return x, y

	def equalize_hist(self, obj, e):
		t = exposure.equalize_hist(obj[:, :, 0])
		t[e] = 0

		return t
	

class Dataset_l(torch.utils.data.Dataset):
	def __init__(self, d, model):
		self.dtst = d
		self.model = model
		self.path_to_model = 'trained_models/PI-CAI_1683897041.pth'

	def __len__(self):
		return len(self.dtst)

	def fetch_item(self, index):
		obj = self.dtst[index, :, :, :]
		temp = obj[:, :, 0]
		norm = (temp - np.min(temp)) / (np.max(temp) - np.min(temp))
		x = torch.from_numpy(norm)
		y_gland = torch.from_numpy(obj[:, :, 1])
		y_lesion = torch.from_numpy(obj[:, :, 2])

		return x, y_gland, y_lesion


	def load_model(self):
		path_to_model = self.path_to_model
		self.model.load_state_dict(torch.load(path_to_model))
		self.model.eval()


	def prepare_input(self, x):
		x = torch.unsqueeze(x, 0)
		x = torch.unsqueeze(x, 0)
		x = x.to(torch.float32)

		return x


	def init_cropping(self, x, y_gland, gland_preds):
		x = torch.squeeze(x, 0)
		x = torch.squeeze(x, 0)
		outputs = torch.squeeze(gland_preds, 0)
		x = x.detach().numpy()
		y_gland = y_gland.detach().numpy()
		gland_preds = outputs.detach().numpy()

		return x, y_gland, gland_preds


	def __getitem__(self, index):
		
		x, y_gland, y_lesion = self.fetch_item(index)
		mri = x
		self.load_model()
		x = self.prepare_input(x)
		
		with torch.no_grad():
			gland_pred = self.model(x)

		gland_pred = torch.argmax(gland_pred, dim=1)
		x, y_gland, gland_pred = self.init_cropping(x, y_gland, gland_pred)

		c_gland_pred, (l, r, t, b) = self.crop_img(gland_pred)
		y_lesion = y_lesion.detach().numpy()
		x, y_gland, y_lesion = self.create_final_objs(x, y_gland, y_lesion, l, r, t, b)
		x = exposure.equalize_hist(x)
		e = np.where(y_lesion < 0.3)
		a = np.where(y_lesion >= 0.3)
		y_lesion[e] = 0
		y_lesion[a] = 1

		x = torch.from_numpy(x)
		y_gland = torch.from_numpy(y_gland)
		y_lesion = torch.from_numpy(y_lesion)
		
		# return mri, x, y_gland, y_lesion
		# print("X: ", x.size(), y_lesion.size())
		return x, y_lesion


	def create_final_objs(self, x, y_gland, y_lesion, l, r, t, b):
		x = x[t:b, l:r]
		y_gland = y_gland.astype(float)
		y_lesion = y_lesion.astype(float)
		y_gland = y_gland[t:b, l:r]
		y_lesion = y_lesion[t:b, l:r]
		e = np.where(y_gland < 0.3)
		y_gland[e] = 0
		e = np.where(y_lesion < 0.3)
		y_lesion[e] = 0
		y_gland = cv2.resize(y_gland, (256, 256))
		y_lesion = cv2.resize(y_lesion, (256, 256))
		x = cv2.resize(x, (256, 256))

		return x, y_gland, y_lesion

	def crop_img(self, img):
		t = 0
		b = img.shape[0] - 1
		l = 0
		r = img.shape[1] - 1
		while np.sum(img[t, :]) == 0 and t < img.shape[0] - 1: t += 1
		while np.sum(img[b, :]) == 0 and b > 1: b -= 1
		while np.sum(img[:, l]) == 0 and l < img.shape[1] - 1: l += 1
		while np.sum(img[:, r]) == 0 and r > 1: r -= 1

		if t < b and l < r:
			crpd = img[t:b, l:r]
			return crpd, (l, r, t, b)
		else:
			crpd = img
			return crpd, (0, img.shape[1] - 1, 0, img.shape[0] - 1)


	def equalize_hist(self, obj, e):
		t = exposure.equalize_hist(obj[:, :, 0])
		t[e] = 0

		return t


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
		# self.clean_dataset()
		# self.split_dataset(self.d_start, self.d_end)
		# self.normalize_sets()
		if self.g_training:
			self.build_g_loaders()
		else:
			print("Lesion_loader...")
			self.build_l_loaders()
		self.losses = np.zeros((self.epochs, 2))
		self.scores = np.zeros((self.epochs, 2))
		self.max_score = 0
		self.log = open("logs.txt", "a")  # append mode


		if self.device == 'cuda':
			print("Cuda available")
			self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
			self.model = self.model.to(self.device)


	def init_components(self):
		self.model     = self.components['model']
		self.g_model   = self.components['g_model']
		self.opt       = self.components['opt']
		self.loss_fn   = self.components['loss_fn']
		self.train_set = self.components['train']
		self.valid_set = self.components['valid']
		self.test_set  = self.components['test']


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
		self.g_training = self.parameters['g_training']

	def init_paths(self):
		self.trained_models = self.paths['trained_models']
		self.metrics = self.paths['metrics']
		self.figures = self.paths['figures']


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
		self.test_set = self.normalize(self.test_set)


	def build_g_loaders(self):
		train_set      = Dataset_g(self.train_set)
		params         = {'batch_size': self.batch_size, 'shuffle': True}
		self.train_ldr = torch.utils.data.DataLoader(train_set, **params)
		valid_set      = Dataset_g(self.valid_set)
		params         = {'batch_size': self.batch_size, 'shuffle': False}
		self.valid_ldr = torch.utils.data.DataLoader(valid_set, **params)
		test_set       = Dataset_g(self.test_set)
		params         = {'batch_size': self.batch_size, 'shuffle': False}
		self.test_ldr  = torch.utils.data.DataLoader(test_set, **params)

	
	def build_l_loaders(self):
		train_set      = Dataset_l(self.train_set, self.g_model)
		params         = {'batch_size': self.batch_size, 'shuffle': True}
		self.train_ldr = torch.utils.data.DataLoader(train_set, **params)
		valid_set      = Dataset_l(self.valid_set, self.g_model)
		params         = {'batch_size': self.batch_size, 'shuffle': False}
		self.valid_ldr = torch.utils.data.DataLoader(valid_set, **params)
		test_set       = Dataset_l(self.test_set, self.g_model)
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
		for epoch in tqdm(range(self.epochs)):

			tr_score, tr_loss = self.epoch_training()
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
			log = str(self.g_training)
			log += str(epoch) + " " + str(score) + " " + path_to_model + "\n"
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
	def prepare_data(self, x, y):
		if self.in_chnls < 2:
			x = torch.unsqueeze(x, 1)
		else:
			x = x.movedim(2, -1)
			x = x.movedim(1, 2)

		x = x.to(torch.float32)
		y = y.to(torch.int64)

		x = x.to(self.device)
		y = y.to(self.device)

		return x, y


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

		step = 0
		# print("Loader len:", len(self.train_ldr))
		for x, y in self.train_ldr:
			
			x, y = self.prepare_data(x, y)
			# print("Epoch training: ", x.size(), y.size())
			step += 1
			self.opt.zero_grad()
			outputs = self.model(x)
			# print("Outputs: ", outputs.size(), y.size())
			loss = self.loss_fn(outputs, y)
			loss.backward()
			self.opt.step()
			# print()

			score = self.calculate_dice(outputs, y)
			# print("Score: ", score)
			current_score += score# * self.train_ldr.batch_size
			current_loss  += loss# * self.train_ldr.batch_size

		batches = len(self.train_ldr)
		# print("Batches: ", batches)
		epoch_score = current_score / batches#len(self.valid_ldr.dataset)
		epoch_loss  = current_loss / batches#len(self.valid_ldr.dataset)
		# epoch_score = current_score / len(self.train_ldr.dataset)
		# epoch_loss  = current_loss / len(self.train_ldr.dataset)

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
		# self.model.train(False)
		self.model.eval()
		current_score = 0.0
		current_loss = 0.0
		elems = 0
		for x, y in self.valid_ldr:
			# print(x.size()[0])
			elems += x.size()[0]
			
			x, y = self.prepare_data(x, y)

			with torch.no_grad():
				outputs = self.model(x)
				loss = self.loss_fn(outputs, y)

			score = self.calculate_dice(outputs, y)
			current_score += score# * self.valid_ldr.batch_size
			current_loss  += loss# * self.valid_ldr.batch_size

		# print(self.valid_ldr.batch_size, len(self.valid_ldr.dataset))
		# print(len(self.valid_ldr.dataset) / self.valid_ldr.batch_size)
		batches = len(self.valid_ldr)
		epoch_score = current_score / batches#len(self.valid_ldr.dataset)
		epoch_loss  = current_loss / batches#len(self.valid_ldr.dataset)

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

		for x, y in self.test_ldr:
			x, y = self.prepare_data(x, y)

			with torch.no_grad():
				outputs = self.model(x)

			score = self.calculate_dice(outputs, y)
			current_score += score * self.test_ldr.batch_size

		test_set_score = current_score / len(self.test_ldr.dataset)

		return test_set_score.item()


	def clear_false_preds(self, pred, h, area_thresh = 200):
		c = 0
		area = 0
		area_s = 0
		area_e = 0
		found = False

		while c < pred.shape[1]:
			if h:
				col_count = np.count_nonzero(pred[:, c])
			else:
				col_count = np.count_nonzero(pred[c, :])
			if col_count > 0:
				if not found:
					found = True
					area_s = c
				area += col_count
			if col_count == 0:
				if found:
					found = False
					area_e = c
					if area < area_thresh and area_s != area_e:
						if h:
							pred[:, area_s:area_e] = 0
						else:
							pred[area_s:area_e, :] = 0
						area_s = area_e = 0
					area = 0
			c += 1

		return pred


	def ext_inference(self, set_ldr):
		path_to_model = self.trained_models + self.inf_model
		self.model.load_state_dict(torch.load(path_to_model))
		self.model.eval()
		current_score = 0.0
		i = 0
		for x, y in set_ldr:
			x, y = self.prepare_data(x, y)

			with torch.no_grad():
				outputs = self.model(x)

			score = self.calculate_dice(outputs, y)
			current_score += score * set_ldr.batch_size

			img = x.cpu().detach().numpy()
			preds = torch.argmax(outputs, dim=1)
			ano = preds.cpu().detach().numpy()

			ano = self.clear_false_preds(ano[2, :, :], False)
			ano = self.clear_false_preds(ano, True)

			print(img.shape, ano.shape, np.unique(ano))
			plt.figure()
			plt.imshow(img[2, 0, :, :], cmap='gray')
			plt.imshow(ano[:, :], alpha=0.3)
			plt.savefig("inf/img_"+str(i)+'.png')


			ano = y.cpu().detach().numpy()
			print(img.shape, ano.shape)
			plt.figure()
			plt.imshow(img[2, 0, :, :], cmap='gray')
			plt.imshow(ano[2, :, :], alpha=0.3)
			plt.savefig("inf/img_an_"+str(i)+'.png')
			i += 1

		test_set_score = current_score / len(set_ldr.dataset)

		return test_set_score.item()


	# detach_tensors:
	# ---------------
	# Given preds and targets tensors, this function
	def detach_tensors(self, preds, targets):
		preds = torch.argmax(preds, dim=1)
		preds = preds.cpu().detach().numpy()
		targets = targets.cpu().detach().numpy()

		return preds, targets


	def calculate_iou(self, preds, ys, smooth=1):
		preds, ys = self.detach_tensors(preds, ys)
		d = preds + ys
		m = 0
		for i in range(len(preds)):
			c = d[i, :, :]
			inter = np.count_nonzero(c > 1)
			union = np.count_nonzero(c > 0)
			m += (inter+smooth)/(union+smooth)

		return m/len(preds)


	def calculate_dice(self, preds, targets, smooth=1):
		preds = preds.cpu()
		targets = targets.cpu()
		preds = torch.argmax(preds, dim=1)
		preds = preds.view(-1)
		targets = targets.view(-1)
		dice = Dice(average='macro', num_classes=2)
		d = dice(preds, targets)
		return d


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
