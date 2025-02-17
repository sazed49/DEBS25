from __future__ import print_function

import torch
import torch.nn.functional as F
import logging
from tools import tools
from copy import deepcopy
from clients import *
from tools.blur import GaussianSmoothing



class Unreliable_client(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss,
            device='cpu', mean=0, max_std=0.5, fraction_noise=0.5,
            fraction_train=0.3, blur_method='gaussian_smooth', inner_epochs=1,
            channels=1, kernel_size=5):
        logging.info("init UNRELIABLE Client {}".format(cid))
        super(Unreliable_client, self).__init__(cid, model, dataLoader,
            optimizer, criterion, device, inner_epochs)
        self.max_std = max_std
        self.mean = mean
        self.unreliable_fraction = fraction_noise
        self.fraction_train=fraction_train
        self.seed = 0
        self.channels = channels
        self.kernel_size = kernel_size
        self.blur_method = blur_method
        

    def data_transform(self, data, target):
        if torch.rand(1) < self.unreliable_fraction:
            if self.blur_method == 'add_noise':
                # APPROACH 1: simple add noise
                torch.manual_seed(self.seed)
                std = torch.rand(data.shape)*self.max_std
                gaussian = torch.normal(mean=self.mean, std=std)
                assert data.shape == gaussian.shape, "Inconsistent Gaussian noise shape"
                data_ = data + gaussian
            else: # gaussian_smooth
                # APPROACH 2: Gaussian smoothing
                smoothing = GaussianSmoothing(self.channels, self.kernel_size, self.max_std)
                data_ = F.pad(data, (2,2,2,2), mode='reflect')
                data_ = smoothing(data_)
        else:
            data_ = data
        self.seed += 1

        return data_, target

    def train(self):
        self.model.to(self.device)
        self.model.train()
        for epoch in range(self.inner_epochs):
            for batch_idx, (data, target) in enumerate(self.dataLoader):
                if torch.rand(1) > self.fraction_train: #just train 30% of local dataset
                    continue
                data, target = self.data_transform(data, target)
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        self.isTrained = True
        self.model.cpu()  ## avoid occupying gpu when idle
class Additive_Noise_Attacker(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss,
            device='cpu', mean=0, std=0.1 ,inner_epochs=1):
        super(Additive_Noise_Attacker, self).__init__(cid, model, dataLoader,
            optimizer, criterion, device, inner_epochs)
        self.mean = mean
        self.std = std
        
        logging.info("init ATTACK ADD NOISE TO GRAD Client {}".format(cid))
    # def update(self):
    #     assert self.isTrained, 'nothing to update, call train() to obtain gradients'
    #     newState = self.model.state_dict()
    #     trainable_parameter = tools.getTrainableParameters(self.model)
    #     for p in self.originalState:
    #         self.stateChange[p] = newState[p] - self.originalState[p]
    #         if p not in trainable_parameter:
    #             continue
    #         std = torch.ones(self.stateChange[p].shape)*self.std
    #         gaussian = torch.normal(mean=self.mean, std=std)
    #         self.stateChange[p] += gaussian
    #         self.sum_hog[p] += self.stateChange[p]
    #         K_ = len(self.hog_avg)
    #         if K_ == 0:
    #             self.avg_delta[p] = self.stateChange[p]
    #         elif K_ < self.K_avg:
    #             self.avg_delta[p] = (self.avg_delta[p]*K_ + self.stateChange[p])/(K_+1)
    #         else:
    #             self.avg_delta[p] += (self.stateChange[p] - self.hog_avg[0][p])/self.K_avg
    #     self.hog_avg.append(self.stateChange)
    #     self.isTrained = False
    
    #My Update
    def update(self):
      assert self.isTrained, 'nothing to update, call train() to obtain gradients'
      newState = self.model.state_dict()
      trainable_parameter = tools.getTrainableParameters(self.model)

      for p in self.originalState:
          self.stateChange[p] = newState[p] - self.originalState[p]
          if p not in trainable_parameter:
                continue
        # Compute current gradient
          
          std = torch.ones(self.stateChange[p].shape)*self.std
          gaussian = torch.normal(mean=self.mean, std=std)
          self.stateChange[p] += gaussian
          
          
        # Update local median as the average of current gradient and previous local median
          self.local_avg[p] = (self.stateChange[p] + self.local_avg[p]) / 2

      self.isTrained = False


    
    
class Sign_Flip_Attacker(Client):
    def __init__(self, cid, model, dataLoader, optimizer, criterion=F.nll_loss,
            device='cpu', scale=1, inner_epochs=1):
        super(Sign_Flip_Attacker, self).__init__(cid, model, dataLoader,
            optimizer, criterion, device, inner_epochs)
        logging.info("init ATTACK OMNISCIENT Client {}".format(cid))
        self.scale = scale
        #self.local_median3 = deepcopy(self.originalState)

    # def update(self):
    #     assert self.isTrained, 'nothing to update, call train() to obtain gradients'
    #     newState = self.model.state_dict()
    #     trainable_parameter = tools.getTrainableParameters(self.model)
    #     for p in self.originalState:
    #         self.stateChange[p] = newState[p] - self.originalState[p]
    #         if p not in trainable_parameter:
    #             continue
    #         #             if not "FloatTensor" in self.originalState[p].type():
    #         #                 continue
    #         self.stateChange[p] *= (-self.scale)
    #         self.sum_hog[p] += self.stateChange[p]
    #         K_ = len(self.hog_avg)
    #         if K_ == 0:
    #             self.avg_delta[p] = self.stateChange[p]
    #         elif K_ < self.K_avg:
    #             self.avg_delta[p] = (self.avg_delta[p]*K_ + self.stateChange[p])/(K_+1)
    #         else:
    #             self.avg_delta[p] += (self.stateChange[p] - self.hog_avg[0][p])/self.K_avg

    #     self.hog_avg.append(self.stateChange)
    #     self.isTrained = False
    # My Update
    def update(self):
      assert self.isTrained, 'nothing to update, call train() to obtain gradients'
      newState = self.model.state_dict()
      trainable_parameter = tools.getTrainableParameters(self.model)

      for p in self.originalState:
          self.stateChange[p] = newState[p] - self.originalState[p]
          if p not in trainable_parameter:
                continue
          #reverses th grad
          self.stateChange[p] *= (-self.scale)
        
          

        # Update local median as the average of current gradient and previous local median
          self.local_avg[p] = (self.stateChange[p] + self.local_avg[p]) / 2
      self.isTrained = False
    