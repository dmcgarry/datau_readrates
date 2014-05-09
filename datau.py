#!/usr/local/python/bin/python

import re
import pandas as pd
import numpy as np
import copy
from itertools import product
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2


#######################
## Tuning Parameters ##
#######################
#variables to add to model
vars = {
	'rfClass' : ['sparseZeroPred','inbox_times_read', 'campaign_size', 'avg_domain_read_rate', 'avg_user_avg_read_rate', 'avg_user_domain_avg_read_rate', 'avg_domain_inbox_rate', 'mb_superuser', 'mb_engper', 'mb_supersub', 'mb_engsec', 'mb_inper', 'mb_insec', 'mb_unengsec', 'mb_idlesub', 'inbox_times_user_read', 'sketchy_domain', 'subj_length', 'subj_speical', 'subj_cap'],
	'gbmClass' : ['sparseZeroPred','inbox_times_read', 'campaign_size', 'avg_domain_read_rate', 'avg_user_avg_read_rate', 'avg_user_domain_avg_read_rate', 'avg_domain_inbox_rate', 'mb_superuser', 'mb_engper', 'mb_supersub', 'mb_engsec', 'mb_inper', 'mb_insec', 'mb_unengsec', 'mb_idlesub', 'inbox_times_user_read', 'sketchy_domain', 'subj_length', 'subj_speical', 'subj_cap'],
	'rfReg' : ['sparsePred','inbox_times_read', 'campaign_size', 'avg_domain_read_rate', 'avg_user_avg_read_rate', 'avg_user_domain_avg_read_rate', 'avg_domain_inbox_rate', 'mb_superuser', 'mb_engper', 'mb_supersub', 'mb_engsec', 'mb_inper', 'mb_insec', 'mb_unengsec', 'mb_idlesub', 'inbox_times_user_read', 'sketchy_domain', 'subj_length', 'subj_speical', 'subj_cap'],
	'gbmReg' : ['sparsePred','inbox_times_read', 'campaign_size', 'avg_domain_read_rate', 'avg_user_avg_read_rate', 'avg_user_domain_avg_read_rate', 'avg_domain_inbox_rate', 'mb_superuser', 'mb_engper', 'mb_supersub', 'mb_engsec', 'mb_inper', 'mb_insec', 'mb_unengsec', 'mb_idlesub', 'inbox_times_user_read', 'sketchy_domain', 'subj_length', 'subj_speical', 'subj_cap']
}

#model tuning grids
models = {
	'logisticDTM' : {'ngram_range': [(1,5)], 'min_df': [2], 'max_df': [0.7], 'stop_words': [None], 'lowercase': [True], 'binary': [False], 'use_idf': [True], 'sublinear_tf': [True], 'norm': ['l2'], 'strip_accents': ['unicode'],'decode_error': ['ignore']},
	'logisticSparse' : {'C': np.arange(3.5,3.9,0.1)},
	'logisticP' : np.arange(0.58,0.60,0.01),
	'rfClass' : {'n_estimators' : [100], 'max_depth' : [21,22,23], 'min_samples_leaf' : [5], 'min_samples_split' : [2]},
	'gbmClass' : {'learning_rate' : [0.06],'n_estimators' : [120],'max_depth':[10],'min_samples_leaf':[2],'min_samples_split' : [2],'subsample': [1.0]},
	'ridgeDTM' : {'ngram_range': [(1,2)], 'min_df': [3], 'max_df': [0.7], 'stop_words': [None], 'lowercase': [True], 'binary': [False], 'use_idf': [False], 'sublinear_tf': [False], 'norm': ['l2'], 'strip_accents': ['unicode'],'decode_error': ['ignore']},
	'ridgeSparse' : {'alpha': np.arange(0.3,0.9,0.1)},
	'ridgeP' : np.arange(0.99,1.0,0.01),
	'rfReg' : {'n_estimators' : [800], 'max_depth' : [19], 'min_samples_leaf' : [2], 'min_samples_split' : [2]},
	'gbmReg' : {'learning_rate' : [0.039],'n_estimators' : [215],'max_depth':[9],'min_samples_leaf':[5],'min_samples_split' : [2],'subsample': [1.0]}
}

######################
## Helper Functions ##
######################
def bestF1(obs,pred):
	""" Find the optimal F1 value over a grid of cutoffs """
	best = 0
	bestcut = 0
	for cutoff in np.arange(0.3,0.99,0.01):
		tmp = f1_score(obs,pd.Series(pred > cutoff).apply(lambda x: 1 if x else 0))
		if tmp > best:
			best = tmp
			bestcut = cutoff
	return best	

def checkCutOff(obsReg,predReg,predClass):
	""" Find the best cutoff from the zero read model to the read rate model """
	best = 99
	tmp = np.copy(predReg)
	for cutoff in np.arange(0.99,0.00,-0.01):
		tmp[predClass > cutoff] = 0
		if np.sqrt(mean_squared_error(obsReg,tmp)) < best:
			best = np.sqrt(mean_squared_error(obsReg,tmp))
			bestcut = cutoff
	return {'rmse': best, 'cutoff': bestcut}
	
	
def applyCut(predReg,predClass,cut):
	""" Apply a cutoff from the default model to the loss model """
	pred = np.copy(predReg)
	pred[predClass > cut] = 0.0
	return pred


###############
## Prep Data ##
###############
def prepData(dir="/na/home/dmcgarry/rpkaggle/comp1_readRates/data/"):
	train = pd.read_csv(dir+"train.csv")
	test = pd.read_csv(dir+"test.csv")
	#rollup redactions
	train.subject = [re.sub('(redact[^\s]+|( |^)\_( |$))',' redact ',s).strip() for s in train.subject]
	test.subject = [re.sub('(redact[^\s]+|( |^)\_( |$))',' redact ',s).strip() for s in test.subject]
	#add inbox times read
	train['inbox_times_read'] = train.avg_domain_read_rate * train.avg_domain_inbox_rate
	test['inbox_times_read'] = test.avg_domain_read_rate * test.avg_domain_inbox_rate
	#add inbox times user read
	train['inbox_times_user_read'] = train.avg_user_domain_avg_read_rate * train.avg_domain_inbox_rate
	test['inbox_times_user_read'] = test.avg_user_domain_avg_read_rate * test.avg_domain_inbox_rate
	#add grouped days of the week
	dayGroupMap = {'Sat':'weekend','Sun':'weekend','Mon':'weektail','Tues':'weekmid','Wed':'weekmid','Thurs':'weekmid','Fri':'weektail'}
	train = train.join(pd.get_dummies(train['day'].map(dayGroupMap)))
	test = test.join(pd.get_dummies(test['day'].map(dayGroupMap)))
	#add flag for sketchy domains
	train['sketchy_domain'] = train['from_domain'].apply(lambda x: 1 if bool(re.match("^[0-9]",x)) else 0)
	test['sketchy_domain'] = test['from_domain'].apply(lambda x: 1 if bool(re.match("^[0-9]",x)) else 0)
	#domains with caps
	train['cap_domain'] = train['from_domain'].apply(lambda x: 0 if x.islower() else 1)
	test['cap_domain'] = test['from_domain'].apply(lambda x: 0 if x.islower() else 1)
	#subject length
	train['subj_length'] = [len(x) for x in train['subject']]
	test['subj_length'] = [len(x) for x in test['subject']]
	#subject special chars
	train['subj_speical'] = [len(re.sub("(\w|\s)",'',x)) for x in train['subject']]
	test['subj_speical'] = [len(re.sub("(\w|\s)",'',x)) for x in test['subject']]
	#subject cap percentage
	train['subj_cap'] = [sum(1 for y in x if y.isupper()) for x in train['subject']]
	test['subj_cap'] = [sum(1 for y in x if y.isupper()) for x in test['subject']]
	#subject starts with [...]
	train['subj_bracket'] = train['subject'].apply(lambda x: 1 if re.match("^\[.*\]",x) else 0)
	test['subj_bracket'] = test['subject'].apply(lambda x: 1 if re.match("^\[.*\]",x) else 0)
	#zero read rates
	train['zero_read'] = train['read_rate'].apply(lambda x: 0 if x > 0 else 1)
	return train, test


###########################################
## Sparse Textual Model - Classification ##
###########################################
def runSparseClass(train,test,dtm,model,grid,pGrid,bestScore,bestP,bestPred,bestModel,bestDTM):
	""" Create a sparse text model for zero reads """
	#convert text to ngrams
	dtm.fit(train.subject.append(test.subject))
	#make dataset to train model
	trainDTM = dtm.transform(train.subject)
	testDTM = dtm.transform(test.subject)
	cv = KFold(len(train), n_folds=10, indices=True, shuffle=True, random_state=27)
	#try different p-values for chi-squared feature selection
	for p in pGrid:
		#try model tuning parameters
		for tune in [dict(zip(grid, v)) for v in product(*grid.values())]:
			for k in tune.keys():
				setattr(model,k,tune[k])
			#create sparse model
			tmp = np.zeros(len(train))
			for tr, valid in cv:
				feat = chi2(trainDTM[tr],train['zero_read'][tr].values)[1]
				feat = [x for x in range(len(feat)) if feat[x] <= p]
				model.fit(trainDTM[:,feat][tr],train['zero_read'][tr])
				tmp[valid] = model.predict_proba(trainDTM[:,feat][valid])[:,1]
			if bestScore < bestF1(train.zero_read,tmp):
				bestScore = bestF1(train.zero_read,tmp)
				bestP = copy.copy(p)
				bestPred = tmp.copy()
				bestModel = copy.copy(model)
				bestDTM = copy.copy(dtm)
	return bestScore, bestPred, bestP, bestModel, bestDTM



def buildSparseModelClass(train,test,dtm,dtmGrid,model,modelGrid,pGrid):
	""" Build the sparse tf-idf transformation and text model for zero reads """
	bestScore = 0
	bestPred = np.zeros(len(train))
	bestP = pGrid[0]
	bestModel = copy.copy(model)
	bestDTM = copy.copy(dtm)
	#build all permutations of the tf-idf models
	for tune in [dict(zip(dtmGrid, v)) for v in product(*dtmGrid.values())]:
		#update tf-idf model
		for k in tune.keys():
			setattr(dtm,k,tune[k])
		#create sparse model
		bestScore, bestPred, bestP, bestModel, bestDTM  = runSparseClass(train,test,dtm,model,modelGrid,pGrid,bestScore,bestP,bestPred,bestModel,bestDTM)
	train['sparseZeroPred'] = bestPred
	print "F1: ", bestScore
	print "AUC:", roc_auc_score(train.zero_read,train.sparseZeroPred)
	print "p:", bestP
	print "DTM:", bestDTM.get_params()
	print "Model:", bestModel.get_params()
	#re-fit and get test set prediction
	trainDTM = bestDTM.transform(train.subject)
	testDTM = bestDTM.transform(test.subject)
	feat = chi2(trainDTM,train['zero_read'].values)[1]
	feat = [x for x in range(len(feat)) if feat[x] <= bestP]
	bestModel.fit(trainDTM[:,feat],train['zero_read'])
	test['sparseZeroPred'] = bestModel.predict_proba(testDTM[:,feat])[:,1]
	return train, test

############################
## Stacked Classification ##
############################

def makeClass(train,test,model,vars,grid,mtype="rf"):
	""" Build a model to predict zero reads """
	cv = KFold(len(train), n_folds=10, indices=True, shuffle=True, random_state=27)
	bestScore = 0
	bestPred = np.zeros(len(train))
	bestModel = copy.copy(model)
	#run each model through a tuning grid
	for tune in [dict(zip(grid, v)) for v in product(*grid.values())]:
		#update model
		for k in tune.keys():
			setattr(model,k,tune[k])
		#set up tmp predictions
		tmpPred = np.zeros(len(train))
		#run models through CV loop 
		for tr, val in cv:
			model.fit(train[vars].ix[tr],train['zero_read'].ix[tr])
			tmpPred[val] = model.predict_proba(train[vars].ix[val])[:,1]
		#see if last model was best
		if bestScore < bestF1(train.zero_read,tmpPred):
			bestScore = bestF1(train.zero_read,tmpPred)
			bestPred = tmpPred.copy()
			bestModel = copy.copy(model)
	#update variable in dataset with best results
	train[mtype+'ZeroPred'] = bestPred
	print "F1: ",bestScore
	print "AUC:", roc_auc_score(train.zero_read,train[mtype+'ZeroPred'])
	print bestModel.get_params()
	bestModel.fit(train[vars],train['zero_read'])
	test[mtype+'ZeroPred'] = bestModel.predict_proba(test[vars])[:,1]
	return train, test

#######################################
## Sparse Textual Model - Regression ##
#######################################

def runSparseReg(train,test,dtm,model,grid,pGrid,bestScore,bestP,bestPred,bestModel,bestDTM):
	""" Create a sparse text model for non-zero reads """
	#convert text to ngrams
	dtm.fit(train.subject.append(test.subject))
	#make dataset to train model
	trainDTM = dtm.transform(train.subject)
	testDTM = dtm.transform(test.subject)
	cv = KFold(len(train), n_folds=10, indices=True, shuffle=True, random_state=27)
	#try different p-values for chi-squared feature selection
	for p in pGrid:
		#try model tuning parameters
		for tune in [dict(zip(grid, v)) for v in product(*grid.values())]:
			for k in tune.keys():
				setattr(model,k,tune[k])
			#create sparse model
			tmp = np.zeros(len(train))
			for tr, valid in cv:
				feat = chi2(trainDTM[tr],train['zero_read'][tr].values)[1]
				feat = [x for x in range(len(feat)) if feat[x] <= p]
				model.fit(trainDTM[:,feat][tr],train['read_rate'][tr])
				tmp2 = model.predict(trainDTM[:,feat][valid])
				tmp2[tmp2 < 0] = 0
				tmp2[tmp2 > 1] = 1
				tmp[valid] = tmp2
			if bestScore > np.sqrt(mean_squared_error(train.read_rate,tmp)):
				bestScore = np.sqrt(mean_squared_error(train.read_rate,tmp))
				bestP = p
				bestPred = tmp.copy()
				bestModel = copy.copy(model)
				bestDTM = copy.copy(dtm)
	return bestScore, bestPred, bestP, bestModel, bestDTM



def buildSparseModelReg(train,test,dtm,dtmGrid,model,modelGrid,pGrid):
	""" Build the sparse tf-idf transformation and text model for non-zero reads """
	bestScore = 1
	bestPred = np.zeros(len(train))
	bestP = pGrid[0]
	bestModel = copy.copy(model)
	bestDTM = copy.copy(dtm)
	#build all permutations of the tf-idf models
	for tune in [dict(zip(dtmGrid, v)) for v in product(*dtmGrid.values())]:
		#update tf-idf model
		for k in tune.keys():
			setattr(dtm,k,tune[k])
		#create sparse model
		bestScore, bestPred, bestP, bestModel, bestDTM  = runSparseReg(train,test,dtm,model,modelGrid,pGrid,bestScore,bestP,bestPred,bestModel,bestDTM)
	train['sparsePred'] = bestPred
	print "RMSE:", bestScore
	print "p:", bestP
	print "DTM:", bestDTM.get_params()
	print "Model:", bestModel.get_params()
	#re-fit and get test set prediction
	trainDTM = bestDTM.transform(train.subject)
	testDTM = bestDTM.transform(test.subject)
	feat = chi2(trainDTM,train['zero_read'].values)[1]
	feat = [x for x in range(len(feat)) if feat[x] <= bestP]
	bestModel.fit(trainDTM[:,feat],train['read_rate'])
	tmp = bestModel.predict(testDTM[:,feat])
	tmp[tmp < 0] = 0
	tmp[tmp > 1] = 1
	test['sparsePred'] = tmp
	return train, test

########################
## Stacked Regression ##
########################

def makeReg(train,test,model,vars,grid,mtype="rf"):
	""" Build a model to predict read rate """
	cv = KFold(len(train), n_folds=10, indices=True, shuffle=True, random_state=27)
	bestScore = 99
	bestPred = np.zeros(len(train))
	bestModel = copy.copy(model)
	#run each model through a tuning grid
	for tune in [dict(zip(grid, v)) for v in product(*grid.values())]:
		#update model
		for k in tune.keys():
			setattr(model,k,tune[k])
		#set up tmp predictions
		tmpPred = np.zeros(len(train))
		#run models through CV loop 
		for tr, val in cv:
			model.fit(train[vars].ix[tr],train['read_rate'].ix[tr])
			tmp2 = model.predict(train[vars].ix[val])
			tmp2[tmp2 < 0] = 0
			tmp2[tmp2 > 1] = 1
			tmpPred[val] = tmp2
		#see if last model was best
		tmpScore = checkCutOff(train.read_rate,tmpPred,train.avgZeroPred)
		print tmpScore
		if bestScore > tmpScore['rmse']:
			bestScore = tmpScore['rmse']
			bestPred = applyCut(tmpPred,train.avgZeroPred,tmpScore['cutoff'])
			bestModel = copy.copy(model)
			bestCut = tmpScore['cutoff']
	#update variable in dataset with best results
	train[mtype+'Pred'] = bestPred
	print "RMSE: ",bestScore
	print bestModel.get_params()
	bestModel.fit(train[vars],train['read_rate'])
	tmp = bestModel.predict(test[vars])
	tmp[tmp < 0] = 0
	tmp[tmp > 1] = 1
	test[mtype+'Pred'] = applyCut(tmp,test.avgZeroPred,bestCut)
	return train, test

##########
## Main ##
##########
def main():
	#load data
	train,test = prepData()
	# run classification models
	train,test = buildSparseModelClass(train,test,TfidfVectorizer(),models['logisticDTM'],LogisticRegression(),models['logisticSparse'],models['logisticP'])
	train,test = makeClass(train,test,RandomForestClassifier(max_features='sqrt',n_jobs=5,random_state=12),vars['rfClass'],models['rfClass'],"rf")
	train,test = makeClass(train,test,GradientBoostingClassifier(max_features='sqrt',random_state=12),vars['gbmClass'],models['gbmClass'],"gbm")
	# blend classification models
	train['avgZeroPred'] = train.rfZeroPred*0.5 + train.gbmZeroPred*0.5
	test['avgZeroPred'] = test.rfZeroPred*0.5 + test.gbmZeroPred*0.5
	print bestF1(train.zero_read,train.avgZeroPred)
	print roc_auc_score(train.zero_read,train.avgZeroPred)
	#run regression models
	train, test = buildSparseModelReg(train,test,TfidfVectorizer(),models['ridgeDTM'],Ridge(),models['ridgeSparse'],models['ridgeP'])
	train,test = makeReg(train,test,RandomForestRegressor(max_features='sqrt',n_jobs=5,random_state=12),vars['rfReg'],models['rfReg'],"rf")
	train,test = makeReg(train,test,GradientBoostingRegressor(max_features='sqrt',random_state=12),vars['gbmReg'],models['gbmReg'],"gbm")
	#blend regression models
	lm = LinearRegression(fit_intercept=False)
	lm.fit(train[['rfPred','gbmPred']],train['read_rate'])
	train['finalPred'] = lm.predict(train[['rfPred','gbmPred']])
	train['finalPred'][train['finalPred'] < 0] = 0
	train['finalPred'][train['finalPred'] > 1] = 1
	print "Final RMSE:", np.sqrt(mean_squared_error(train.read_rate,train.finalPred))
	#make final predictions
	test['finalPred'] = lm.predict(test[['rfPred','gbmPred']])
	test['finalPred'][test['finalPred'] < 0] = 0
	test['finalPred'][test['finalPred'] > 1] = 1
	test[['id','finalPred']].to_csv("/na/home/dmcgarry/rpkaggle/comp1_readRates/secretsauce.csv")

# run everything when calling script from CLI
if __name__ == "__main__":
	main()

