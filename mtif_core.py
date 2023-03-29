
import time
import torch
import calendar
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

class training():

	def __init__(self, comps, params, paths):
		self.parameters = params
		self.components = comps
		self.paths = paths
		self.init()
		self.thresh_act = self.create_threshold_activation()
		self.model = self.model.cuda()
		self.losses = np.zeros((self.epochs, 2))
		self.scores = np.zeros((self.epochs, 2))


	def init(self):
		self.init_components()
		self.init_parameters()
		self.init_paths()


	def init_components(self):
		self.model     = self.components['model']
		self.opt       = self.components['opt']
		self.loss_fn   = self.components['loss_fn']
		self.train_ldr = self.components['train_ldr']
		self.valid_ldr = self.components['valid_ldr']


	def init_parameters(self):
		self.thresh    = self.parameters['threshold']
		self.epochs    = self.parameters['epochs']
		self.dtst_name = self.parameters['dtst_name']
		self.epoch_thr = self.parameters['epoch_thresh']
		self.score_thr = self.parameters['score_thresh']


	def init_paths(self):
		self.trained_models = self.paths['trained_models']
		self.metrics = self.paths['metrics']


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
		print()
		option = input("Do you wish to continue? [Y/n]: ")
		return (option == 'Y' or option == 'y')



	# Main_training:
	# --------------
	# The supervisor of the training procedure.
	def main_training(self):
		if not (self.print_train_details()):
			return

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


	def save_model_weights(self, epoch, score, loss):
		current_GMT = time.gmtime()
		timestamp = calendar.timegm(current_GMT)

		if epoch > self.epoch_thr and score > self.score_thr:
			path_to_model = self.dtst_name
			path_to_model += "_" + str(epoch) + "_" + str(score) + "_" +str(loss)
			path_to_model += "_" + str(timestamp) + ".pth"
			torch.save(self.model.state_dict(), path_to_model)



	def prepare_data(self, x, y):
		# x = torch.unsqueeze(x, 1)
		x = x.movedim(2, -1)
		x = x.movedim(1, 2)
		# print("xaxaxa: ", x.size())

		x = x.to(torch.float32)
		y = y.to(torch.int64)

		x = x.cuda()
		y = y.cuda()

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
		# print("Epoch training")
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

			# loss.requires_grad = True
			loss.backward()
			self.opt.step()

			score = self.calculate_dice(outputs, y)
			current_score += score * self.train_ldr.batch_size
			current_loss  += loss * self.train_ldr.batch_size

		epoch_score = current_score / len(self.train_ldr.dataset)
		epoch_loss  = current_loss / len(self.train_ldr.dataset)

		return epoch_score, epoch_loss


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

		# print("Epoch validation")
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

		return epoch_score, epoch_loss



	def dice_score(self, preds, targets, smooth=1):

		# preds = torch.squeeze(preds, 1)
		# preds = self.thresh_act(preds)
		# preds = torch.ceil(preds)
		with torch.no_grad():
			preds = torch.argmax(preds, dim=1)

			preds = preds.view(-1)
			targets = targets.view(-1)

			intersection = (preds * targets).sum()
			dice = (2.*intersection + smooth)/(preds.sum() + targets.sum() + smooth)

			return dice



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
			# print("Intersection: ", inter, "Union: ", union, "IoU:", (inter+smooth)/(union+smooth))

			m += (inter+smooth)/(union+smooth)

		return m/len(preds)


	def calculate_dice(self, preds, ys, smooth=1):
		preds, ys = self.detach_tensors(preds, ys)
		d = preds + ys
		m = 0
		for i in range(len(preds)):
			c = d[i, :, :]
			inter = np.count_nonzero(c > 1)
			seg_1 = np.count_nonzero(preds > 0)
			seg_2 = np.count_nonzero(ys > 0)
			# print("Intersection: ", inter, "Union: ", union, "IoU:", (inter+smooth)/(union+smooth))

			m += (2.*inter+smooth)/(seg_1 + seg_2 + smooth)

		return m/len(preds)
