
import time
import torch
import calendar
import numpy as np
from tqdm import tqdm
from torchmetrics import Dice
import torch.nn.functional as F
from matplotlib import pyplot as plt

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
		self.train_ldr = self.components['train_ldr']
		self.valid_ldr = self.components['valid_ldr']


	def init_parameters(self):
		self.thresh    = self.parameters['threshold']
		self.in_chnls  = self.parameters['in_channels']
		self.epochs    = self.parameters['epochs']
		self.dtst_name = self.parameters['dtst_name']
		self.epoch_thr = self.parameters['epoch_thresh']
		self.score_thr = self.parameters['score_thresh']
		self.device    = self.parameters['device']


	def init_paths(self):
		self.trained_models = self.paths['trained_models']
		self.metrics = self.paths['metrics']
		self.figures = self.paths['figures']


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
		self.save_metrics()


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
		# score = score * 100
		# if epoch > self.epoch_thr and score > self.score_thr:
		# 	path_to_model = self.trained_models + self.dtst_name
		# 	path_to_model += "_" + str(epoch) + "_" + str(score) + "_" +str(loss)
		# 	path_to_model += "_" + str(self.timestamp) + ".pth"
		# 	torch.save(self.model.state_dict(), path_to_model)

		if score > self.max_score and epoch > self.epoch_thr:
			path_to_model = self.trained_models + self.dtst_name
			path_to_model += "_" + str(self.timestamp) + ".pth"
			torch.save(self.model.state_dict(), path_to_model)
			log = str(epoch) + " " + str(score) + " " + path_to_model + "\n"
			self.log.write(log)
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
