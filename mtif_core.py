
import time
import torch
import calendar
import numpy as np
from tqdm import tqdm
from skimage import exposure
from torchmetrics import Dice
import torch.nn.functional as F
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt


class Dataset(torch.utils.data.Dataset):
	def __init__(self, d):
		self.dtst = d

	def __len__(self):
		return len(self.dtst)

	def __getitem__(self, index):
		obj = self.dtst[index, :, :, :]
		e = np.where(obj[:, :, 0] < 0.3)
		t = self.equalize_hist(obj, e)
		# t1 = t - obj[:, :, 0]

		# cl = self.apply_clustering(obj[:, :, 0], e, 5)
		# temp = np.zeros((obj.shape[0], obj.shape[0], 3))
		# temp[:, :, 0] = obj[:, :, 0]
		# temp[:, :, 1] = cl
		# temp[:, :, 2] = t
		x = torch.from_numpy(obj[:, :, 0])
		# x = torch.from_numpy(temp)
		y = torch.from_numpy(obj[:, :, 1])

		return x, y


	def equalize_hist(self, obj, e):
		t = exposure.equalize_hist(obj[:, :, 0])
		t[e] = 0

		return t

	def apply_clustering(self, img, e, clusters):
		flatImg=np.reshape(img, [-1, 1])
		kmeans = KMeans(n_clusters=clusters, random_state=0, n_init="auto").fit(flatImg)
		labels = kmeans.labels_
		clusteredImg = np.reshape(labels, img.shape)
		clusteredImg += 1
		clusteredImg[e] = 0

		return clusteredImg


class training():

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


	def init_components(self):
		self.model     = self.components['model']
		self.opt       = self.components['opt']
		self.loss_fn   = self.components['loss_fn']
		# self.train_ldr = self.components['train_ldr']
		# self.valid_ldr = self.components['valid_ldr']
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
		train_set      = Dataset(self.train_set)
		params         = {'batch_size': self.batch_size, 'shuffle': True}
		self.train_ldr = torch.utils.data.DataLoader(train_set, **params)
		valid_set      = Dataset(self.valid_set)
		params         = {'batch_size': self.batch_size, 'shuffle': False}
		self.valid_ldr = torch.utils.data.DataLoader(valid_set, **params)
		test_set       = Dataset(self.test_set)
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
			log = str(epoch) + " " + str(score) + " " + path_to_model + "\n"
			self.log_line = log
			self.max_score = score


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
		for x, y in self.train_ldr:
			x, y = self.prepare_data(x, y)
			step += 1
			self.opt.zero_grad()
			outputs = self.model(x)
			loss = self.loss_fn(outputs, y)
			loss.backward()
			self.opt.step()

			score = self.calculate_dice(outputs, y)
			current_score += score * self.train_ldr.batch_size
			current_loss  += loss * self.train_ldr.batch_size

		epoch_score = current_score / len(self.train_ldr.dataset)
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
		current_score = 0.0
		current_loss = 0.0

		for x, y in self.valid_ldr:
			x, y = self.prepare_data(x, y)

			with torch.no_grad():
				outputs = self.model(x)
				loss = self.loss_fn(outputs, y)

			score = self.calculate_dice(outputs, y)
			current_score += score * self.train_ldr.batch_size
			current_loss  += loss * self.train_ldr.batch_size

		epoch_score = current_score / len(self.valid_ldr.dataset)
		epoch_loss  = current_loss / len(self.valid_ldr.dataset)

		return epoch_score.item(), epoch_loss.item()


	def inference(self):

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


	def ext_inference(self, set_ldr):
		path_to_model = self.trained_models + self.inf_model
		self.model.load_state_dict(torch.load(path_to_model))
		self.model.eval()
		current_score = 0.0

		for x, y in set_ldr:
			x, y = self.prepare_data(x, y)

			with torch.no_grad():
				outputs = self.model(x)

			score = self.calculate_dice(outputs, y)
			current_score += score * set_ldr.batch_size

		test_set_score = current_score / len(set_ldr.dataset)

		return test_set_score.item()



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
