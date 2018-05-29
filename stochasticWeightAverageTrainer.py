import torch
import re
import numpy as np
import collections

class StochasticWeightAvgTrainer:
	mStartEpoch = None
	mEndEpoch = None
	mLR = None
	mSWAVarMap = None

	def __init__(self, model, startEpoch, endEpoch, lr):
		self.mStartEpoch = startEpoch
		self.mEndEpoch = endEpoch
		self.mLR = lr
	#end __init__

	def swaStep(self, model, epoch):
		if epoch == self.mStartEpoch:
			self.mSWAVarMap = {}
			varMap = model.named_parameters()
			for k, v in varMap:
				if True:
					self.mSWAVarMap[k] = v.clone().detach()
				#end if
			#end for
		elif (epoch > self.mStartEpoch) and (epoch <= self.mEndEpoch):
			currEpochCount = epoch - self.mStartEpoch
			varMap = model.named_parameters()
			for k, v in varMap:
				if True:
					self.mSWAVarMap[k].mul_(currEpochCount)
					self.mSWAVarMap[k].add_(v.clone().detach())
					self.mSWAVarMap[k].div_(float(currEpochCount + 1))
				#end if
			#end for			
		#end if

		if epoch == self.mEndEpoch:
			varMap = model.named_parameters()
			for k,v in varMap:
				if True:
					v.detach().copy_(self.mSWAVarMap[k])
				#end if
			#end for
		#end if
	#end swaStep

	def getLR(self, currLR, epoch):
		if (epoch >= self.mStartEpoch) and (epoch <= self.mEndEpoch):
			return self.mLR
		else:
			return currLR
		#end if
	#end getLR

	def isSwaMode(self, epoch):
		if (epoch >= self.mStartEpoch) and (epoch <= self.mEndEpoch):
			return True
		else:
			return False
		#end if
	#end isSwaMode
#end StochasticWeightAvgTrainer
