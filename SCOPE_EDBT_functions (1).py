import ipywidgets as widgets #in modulo
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import random
import os
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import shap
shap.initjs()
import gurobipy as gp
from gurobipy import GRB
import time
from prettytable import PrettyTable


def graph(target, Column, df, Dataset):  # in modulo
  sns.countplot(df, x=df[Column.value], hue=target[Dataset.value])

def RF(df, columns_of_interest, Dataset, target, train_dataset, test_dataset): #in modulo
  mod_rf=RandomForestClassifier(n_estimators=100, random_state=1)
  mod_rf.fit(train_dataset.drop([target[Dataset.value]], axis=1),train_dataset[target[Dataset.value]])
  y_pred=mod_rf.predict(test_dataset.drop(target[Dataset.value], axis=1))
  print("Accuracy global:",metrics.accuracy_score(test_dataset[target[Dataset.value]], y_pred))
  print("REPORT:\n"+metrics.classification_report(test_dataset[target[Dataset.value]], y_pred))
  explainer = shap.TreeExplainer(mod_rf, train_dataset.drop([target[Dataset.value]], axis=1))
  shap_values = explainer.shap_values(test_dataset.drop([target[Dataset.value]], axis=1), check_additivity=False)
  print(shap.summary_plot(shap_values[:,:,1], test_dataset.drop([target[Dataset.value]], axis=1)))

  return mod_rf

def Logit(df, Dataset, target, columns_for_analysis, train_dataset, test_dataset): #in modulo
  mod_log = LogisticRegression(random_state=1).fit(train_dataset.drop([target[Dataset.value]], axis=1),train_dataset[target[Dataset.value]])
  print("Accuracy global:", mod_log.score(test_dataset.drop([target[Dataset.value]], axis=1),test_dataset[target[Dataset.value]]))
  features_names=(columns_for_analysis[Dataset.value])
  print("Features names:", features_names)
  print("Coefficients:", mod_log.coef_, mod_log.intercept_)
  print(classification_report(test_dataset[target[Dataset.value]], mod_log.predict(test_dataset.drop(target[Dataset.value], axis=1))))
  explainer = shap.LinearExplainer(mod_log, train_dataset.drop([target[Dataset.value]], axis=1))
  shap_values = explainer.shap_values(test_dataset.drop([target[Dataset.value]], axis=1))
  print(shap.summary_plot(shap_values[:,:], test_dataset.drop([target[Dataset.value]], axis=1)))

  return mod_log

def local(Dataset, target, columns, sliders, mod, model, train_dataset, test_dataset): #in modulo
  local_test=test_dataset
  for col in columns:
    local_test=local_test[(local_test[col]>=sliders[col].value[0]) & (local_test[col]<=sliders[col].value[1])]
  y_pred_loc=mod.predict(local_test.drop(target[Dataset.value], axis=1))
  print("Accuracy local:",metrics.accuracy_score(local_test[target[Dataset.value]], y_pred_loc))
  print("REPORT:\n"+metrics.classification_report(local_test[target[Dataset.value]], y_pred_loc))

  if model == 'Random forest':
    explainer = shap.TreeExplainer(mod, train_dataset.drop([target[Dataset.value]], axis=1))
    shap_values = explainer.shap_values(local_test.drop([target[Dataset.value]], axis=1), check_additivity=False)
    shap.summary_plot(shap_values[:,:,1], local_test.drop([target[Dataset.value]], axis=1))
  elif model == 'Logistic regression':
    explainer = shap.LinearExplainer(mod, train_dataset.drop([target[Dataset.value]], axis=1), check_additivity=False)
    shap_values = explainer.shap_values(local_test.drop([target[Dataset.value]], axis=1))
    shap.summary_plot(shap_values[:,:], local_test.drop([target[Dataset.value]], axis=1))
  else:
    pass

def optimizer_high(Dataset, beta, train_dataset, test_dataset, columns, target, sliders, mod, model): #in moduli
  up_original = []
  for i in range(len(columns)):
    up_original.append(sliders[columns[i]].value[1])
  print(np.array(up_original))
  low_original = []
  for i in range(len(columns)):
    low_original.append(sliders[columns[i]].value[0])
  print(np.array(low_original))

  class range_model_accuracy():
    GAMMA = 0.99
    eps = 1e-4

    def __init__(self,
                 x, # the dataset, x[i][k] is the k-th feature of the i-th sample
                 n_of_samples,  # lenght of the dataset
                 n_of_features, # number of features not considering the target
                 c, # a list of binary values representing the correct classification. e.g. c[i] = 1 if psi(x[i]) = y[i]
                 M, # a list of length n_of_features where M[i] = 1+ maximum of the absolute value in the range. E.g. for binary features {0,1} we have M = 2
                 beta, # minimum threshold for the accuracy
                 lower_original_value,
                 upper_original_value,
                 upper_bounds = {}, # a dictionary of upper bounds for upper bound range variables.E.g. lower_bounds[i] = 20 >= u[i]
                 lower_bounds ={}, # a dictionary of lower bounds for lower bound range variables.E.g. upper_bounds[i] = 10 <= l[i]
                 fixed_upper_range ={}, # a dictionary of fixed values for the upper range variables. E.g. fixed_upper_range[i] = 19 = u[i]
                 fixed_lower_range={} # a dictionary of fixed values for the lower range variables. E.g. fixed_lower_range[i] = 12 = l[i]
                 ):

        params = {
        "WLSACCESSID": 'cebd302e-fd24-4581-9b76-95e0001e5d28',
        "WLSSECRET": 'c966bb4d-17c6-4203-8be4-7055a9b12227',
        "LICENSEID": 2442013,
        }
        env = gp.Env(params=params)

        self._mdl = gp.Model(env=env)

        self._mdl.params.FeasibilityTol = 1e-6
        self._mdl.params.TimeLimit = 600
        #self._mdl.params.MIPFocus = 1
        #self._mdl.params.RINS = 3



        self.v = self._mdl.addVars(n_of_samples, vtype = GRB.BINARY, name = "v-")
        self.p = self._mdl.addVars([(i,k) for i in range(n_of_samples) for k in range(n_of_features)], vtype=GRB.BINARY, name="p-")
        self.sup = self._mdl.addVars([(i,k) for i in range(n_of_samples) for k in range(n_of_features)], vtype = GRB.BINARY, name = "sup-")

        ## range variables ##
        #- u upper
        self.u = self._mdl.addVars(n_of_features, vtype=GRB.CONTINUOUS, name="u-")
        #- l lower
        self.l = self._mdl.addVars(n_of_features, vtype=GRB.CONTINUOUS, name="l-")


        ### CONSTRAINTS ###
        self.c = np.asarray(c)

        self.threshold_constraint = self._mdl.addConstr( gp.quicksum([self.v[i]*self.c[i] for i in range(n_of_samples)]) >= beta*self.v.sum(),       
                                                         name='threshold_constraint')

        self.bigM_lower = self._mdl.addConstrs(
            (-M[k] * (1 - self.p[i, k]) + self.l[k] <= x[i][k] * self.p[i, k]
              for i in range(n_of_samples) for k in range(n_of_features)),
              name='bigM_lower-')

        self.bigM_upper = self._mdl.addConstrs(
            (M[k] * (1 - self.p[i, k]) + self.u[k] >= x[i][k] *self.p[i, k]
              for i in range(n_of_samples) for k in range(n_of_features)),
              name='bigM_upper-')

        self.eps_lower = self._mdl.addConstrs(
            (M[k] * ( self.p[i, k]+self.sup[i, k])-range_model_accuracy.eps* (1 - self.p[i, k])+ self.l[k] >= x[i][k]
              for i in range(n_of_samples) for k in range(n_of_features)),
              name='eps_lower-')

        self.eps_upper = self._mdl.addConstrs(
            (-M[k] * (self.p[i, k] + 1 - self.sup[i, k]) + range_model_accuracy.eps * (1 - self.p[i, k]) + self.u[k] <=
             x[i][k]
             for i in range(n_of_samples) for k in range(n_of_features)),
            name='eps_upper-')

        self.v_upper = self._mdl.addConstrs(
            (self.v[i] * n_of_features <= gp.quicksum(self.p[i, k] for k in range(n_of_features))
             for i in range(n_of_samples)),
            name='v_upper-')

        self.v_lower = self._mdl.addConstrs(
            (self.v[i] >=  gp.quicksum(self.p[i, k] for k in range(n_of_features)) - n_of_features + 1
             for i in range(n_of_samples)),
            name='v_lower-')

        self.basic_ordering = self._mdl.addConstrs(
            (self.l[k] <= self.u[k] for k in range(n_of_features)),
            name='upper_bounds_on_upper_range-')

        self.upper_bounds_on_upper_range = self._mdl.addConstrs(
            (self.u[k] <= upper_bounds[k] for k in upper_bounds),
            name='upper_bounds_on_upper_range-')

        self.lower_bounds_on_lower_range = self._mdl.addConstrs(
            (self.l[k] >= lower_bounds[k] for k in lower_bounds),
            name='lower_bounds_on_lower_range-')

        self.fixed_upper_range = self._mdl.addConstrs(
            (self.u[k] == fixed_upper_range[k] for k in fixed_upper_range),
            name='fixed_upper_range-')

        self.fixed_lower_range = self._mdl.addConstrs(
            (self.l[k] == fixed_lower_range[k] for k in fixed_lower_range),
            name='fixed_lower_range-')
        range_length = [abs(upper_original_value[i] - lower_original_value[i]) for i in range(n_of_features)]
        self.range_length = range_length
        self.n_of_features = n_of_features

        self._mdl.setObjective(1+self.v.sum()-gp.quicksum([(self.u[i] - self.l[i])/range_length[i] for i in range(n_of_features)])/(1 + n_of_features), GRB.MAXIMIZE)

        self._mdl.update()

        return

    def pobj(self):
        return gp.quicksum([(self.u[i] - self.l[i])/self.range_length[i] for i in range(self.n_of_features)])/(1 + self.n_of_features)

    def write_to_file(self, pathtofile):
        self._mdl.write(pathtofile)



    def solve(self):
        self._mdl.optimize()

    def print_ranges(self, lower_original_value, upper_original_value):
        for i in range(len(self.l)):
            print(lower_original_value[i], '<=  ', self.l[i].X, '   --   ', self.u[i].X, '<=  ',
                  upper_original_value[i])
    def ranges(self):
        l, u = [], []
        for i in range(len(self.l)):
            l.append( self.l[i].X)
            u.append(self.u[i].X)
        return l, u

  n_of_samples = test_dataset.shape[0]
  n_of_features = len(sliders)
  x = test_dataset.drop(target[Dataset.value], axis=1)[columns].to_numpy()
  #y_pred #_tot=clf.predict(df[columns].to_numpy())
  y_pred=mod.predict(test_dataset.drop(target[Dataset.value], axis=1))
  c=np.abs(y_pred-test_dataset[target[Dataset.value]].to_numpy())<=1e-3
  #beta  = 0.70
  M = [max((abs(i), abs(j))) for i,j in zip(up_original, low_original)]
  upper_bounds = {}
  lower_bounds = {}
  fixed_upper_range = {}
  fixed_lower_range = {}
  lower_original_value = np.array(low_original)
  upper_original_value = np.array(up_original)

  mdl = range_model_accuracy(x, # the dataset, x[i][k] is the k-th feature of the i-th sample
                  n_of_samples,  # lenght of the dataset
                  n_of_features, # number of features not considering the target
                  c, # a list of binary values representing the correct classification. e.g. c[i] = 1 if psi(x[i]) = y[i]
                  M, # a list of length n_of_features where M[i] = 1+ maximum of the absolute value in the range. E.g. for binary features {0,1} we have M = 2
                  beta, # minimum threshold for the accuracy
                  lower_original_value,
                  upper_original_value,
                  upper_bounds, # a dictionary of upper bounds for upper bound range variables.E.g. lower_bounds[i] = 20 >= u[i]
                  lower_bounds, # a dictionary of lower bounds for lower bound range variables.E.g. upper_bounds[i] = 10 <= l[i]
                  fixed_upper_range , # a dictionary of fixed values for the upper range variables. E.g. fixed_upper_range[i] = 19 = u[i]
                  fixed_lower_range)
  #mdl.write_to_file('model.lp')
  # with open('model.lp', 'r') as thefile:
  #   lines = thefile.readlines()
  #   for l in lines: print(l)
  mdl.solve()
  mdl.print_ranges(lower_original_value, upper_original_value)
  score =[c[i] for i in mdl.v if mdl.v[i].X>=0.5]
  print(score)
  print(sum(score), sum(score)/len(score))

  local_test=test_dataset
  for i in range(len(columns)):
    local_test=local_test[(local_test[columns[i]]>=mdl.l[i].X) & (local_test[columns[i]]<=mdl.u[i].X)]
  y_pred_loc=mod.predict(local_test.drop(target[Dataset.value], axis=1))

  if model == 'Random forest':
    explainer = shap.TreeExplainer(mod, train_dataset.drop([target[Dataset.value]], axis=1))
    shap_values = explainer.shap_values(local_test.drop([target[Dataset.value]], axis=1), check_additivity=False)
    shap.summary_plot(shap_values[:,:,1], local_test.drop([target[Dataset.value]], axis=1))
  elif model == 'Logistic regression':
    explainer = shap.LinearExplainer(mod, train_dataset.drop([target[Dataset.value]], axis=1), check_additivity=False)
    shap_values = explainer.shap_values(local_test.drop([target[Dataset.value]], axis=1))
    shap.summary_plot(shap_values[:,:], local_test.drop([target[Dataset.value]], axis=1))
  else:
     pass
  return [mdl.l[i].X for i in range(len(columns))], [mdl.u[i].X for i in range(len(columns))]

def optimizer_low(Dataset, beta, train_dataset, test_dataset, columns, target, sliders, mod, model): #in moduli
  up_original = []
  for i in range(len(columns)):
    up_original.append(sliders[columns[i]].value[1])
  print(np.array(up_original))
  low_original = []
  for i in range(len(columns)):
    low_original.append(sliders[columns[i]].value[0])
  print(np.array(low_original))

  class range_model_accuracy():
    GAMMA = 0.99
    eps = 1e-4

    def __init__(self,
                 x, # the dataset, x[i][k] is the k-th feature of the i-th sample
                 n_of_samples,  # lenght of the dataset
                 n_of_features, # number of features not considering the target
                 c, # a list of binary values representing the correct classification. e.g. c[i] = 1 if psi(x[i]) = y[i]
                 M, # a list of length n_of_features where M[i] = 1+ maximum of the absolute value in the range. E.g. for binary features {0,1} we have M = 2
                 beta, # minimum threshold for the accuracy
                 lower_original_value,
                 upper_original_value,
                 upper_bounds = {}, # a dictionary of upper bounds for upper bound range variables.E.g. lower_bounds[i] = 20 >= u[i]
                 lower_bounds ={}, # a dictionary of lower bounds for lower bound range variables.E.g. upper_bounds[i] = 10 <= l[i]
                 fixed_upper_range ={}, # a dictionary of fixed values for the upper range variables. E.g. fixed_upper_range[i] = 19 = u[i]
                 fixed_lower_range={} # a dictionary of fixed values for the lower range variables. E.g. fixed_lower_range[i] = 12 = l[i]
                 ):

        params = {
        "WLSACCESSID": 'cebd302e-fd24-4581-9b76-95e0001e5d28',
        "WLSSECRET": 'c966bb4d-17c6-4203-8be4-7055a9b12227',
        "LICENSEID": 2442013,
        }
        env = gp.Env(params=params)

        self._mdl = gp.Model(env=env)

        self._mdl.params.FeasibilityTol = 1e-6
        self._mdl.params.TimeLimit = 600
        #self._mdl.params.MIPFocus = 1
        #self._mdl.params.RINS = 3



        self.v = self._mdl.addVars(n_of_samples, vtype = GRB.BINARY, name = "v-")
        self.p = self._mdl.addVars([(i,k) for i in range(n_of_samples) for k in range(n_of_features)], vtype=GRB.BINARY, name="p-")
        self.sup = self._mdl.addVars([(i,k) for i in range(n_of_samples) for k in range(n_of_features)], vtype = GRB.BINARY, name = "sup-")

        ## range variables ##
        #- u upper
        self.u = self._mdl.addVars(n_of_features, vtype=GRB.CONTINUOUS, name="u-")
        #- l lower
        self.l = self._mdl.addVars(n_of_features, vtype=GRB.CONTINUOUS, name="l-")


        ### CONSTRAINTS ###
        self.c = np.asarray(c)

        self.threshold_constraint = self._mdl.addConstr( gp.quicksum([self.v[i]*self.c[i] for i in range(n_of_samples)]) <= beta*self.v.sum(),       
                                                         name='threshold_constraint')

        self.bigM_lower = self._mdl.addConstrs(
            (-M[k] * (1 - self.p[i, k]) + self.l[k] <= x[i][k] * self.p[i, k]
              for i in range(n_of_samples) for k in range(n_of_features)),
              name='bigM_lower-')

        self.bigM_upper = self._mdl.addConstrs(
            (M[k] * (1 - self.p[i, k]) + self.u[k] >= x[i][k] *self.p[i, k]
              for i in range(n_of_samples) for k in range(n_of_features)),
              name='bigM_upper-')

        self.eps_lower = self._mdl.addConstrs(
            (M[k] * ( self.p[i, k]+self.sup[i, k])-range_model_accuracy.eps* (1 - self.p[i, k])+ self.l[k] >= x[i][k]
              for i in range(n_of_samples) for k in range(n_of_features)),
              name='eps_lower-')

        self.eps_upper = self._mdl.addConstrs(
            (-M[k] * (self.p[i, k] + 1 - self.sup[i, k]) + range_model_accuracy.eps * (1 - self.p[i, k]) + self.u[k] <=
             x[i][k]
             for i in range(n_of_samples) for k in range(n_of_features)),
            name='eps_upper-')

        self.v_upper = self._mdl.addConstrs(
            (self.v[i] * n_of_features <= gp.quicksum(self.p[i, k] for k in range(n_of_features))
             for i in range(n_of_samples)),
            name='v_upper-')

        self.v_lower = self._mdl.addConstrs(
            (self.v[i] >=  gp.quicksum(self.p[i, k] for k in range(n_of_features)) - n_of_features + 1
             for i in range(n_of_samples)),
            name='v_lower-')

        self.basic_ordering = self._mdl.addConstrs(
            (self.l[k] <= self.u[k] for k in range(n_of_features)),
            name='upper_bounds_on_upper_range-')

        self.upper_bounds_on_upper_range = self._mdl.addConstrs(
            (self.u[k] <= upper_bounds[k] for k in upper_bounds),
            name='upper_bounds_on_upper_range-')

        self.lower_bounds_on_lower_range = self._mdl.addConstrs(
            (self.l[k] >= lower_bounds[k] for k in lower_bounds),
            name='lower_bounds_on_lower_range-')

        self.fixed_upper_range = self._mdl.addConstrs(
            (self.u[k] == fixed_upper_range[k] for k in fixed_upper_range),
            name='fixed_upper_range-')

        self.fixed_lower_range = self._mdl.addConstrs(
            (self.l[k] == fixed_lower_range[k] for k in fixed_lower_range),
            name='fixed_lower_range-')
        range_length = [abs(upper_original_value[i] - lower_original_value[i]) for i in range(n_of_features)]
        self.range_length = range_length
        self.n_of_features = n_of_features

        self._mdl.setObjective(1+self.v.sum()-gp.quicksum([(self.u[i] - self.l[i])/range_length[i] for i in range(n_of_features)])/(1 + n_of_features), GRB.MAXIMIZE)

        self._mdl.update()

        return

    def pobj(self):
        return gp.quicksum([(self.u[i] - self.l[i])/self.range_length[i] for i in range(self.n_of_features)])/(1 + self.n_of_features)

    def write_to_file(self, pathtofile):
        self._mdl.write(pathtofile)



    def solve(self):
        self._mdl.optimize()

    def print_ranges(self, lower_original_value, upper_original_value):
        for i in range(len(self.l)):
            print(lower_original_value[i], '<=  ', self.l[i].X, '   --   ', self.u[i].X, '<=  ',
                  upper_original_value[i])
    def ranges(self):
        l, u = [], []
        for i in range(len(self.l)):
            l.append( self.l[i].X)
            u.append(self.u[i].X)
        return l, u

  n_of_samples = test_dataset.shape[0]
  n_of_features = len(sliders)
  x = test_dataset.drop(target[Dataset.value], axis=1)[columns].to_numpy()
  #y_pred #_tot=clf.predict(df[columns].to_numpy())
  y_pred=mod.predict(test_dataset.drop(target[Dataset.value], axis=1))
  c=np.abs(y_pred-test_dataset[target[Dataset.value]].to_numpy())<=1e-3
  #beta  = 0.70
  M = [max((abs(i), abs(j))) for i,j in zip(up_original, low_original)]
  upper_bounds = {}
  lower_bounds = {}
  fixed_upper_range = {}
  fixed_lower_range = {}
  lower_original_value = np.array(low_original)
  upper_original_value = np.array(up_original)

  mdl = range_model_accuracy(x, # the dataset, x[i][k] is the k-th feature of the i-th sample
                  n_of_samples,  # lenght of the dataset
                  n_of_features, # number of features not considering the target
                  c, # a list of binary values representing the correct classification. e.g. c[i] = 1 if psi(x[i]) = y[i]
                  M, # a list of length n_of_features where M[i] = 1+ maximum of the absolute value in the range. E.g. for binary features {0,1} we have M = 2
                  beta, # minimum threshold for the accuracy
                  lower_original_value,
                  upper_original_value,
                  upper_bounds, # a dictionary of upper bounds for upper bound range variables.E.g. lower_bounds[i] = 20 >= u[i]
                  lower_bounds, # a dictionary of lower bounds for lower bound range variables.E.g. upper_bounds[i] = 10 <= l[i]
                  fixed_upper_range , # a dictionary of fixed values for the upper range variables. E.g. fixed_upper_range[i] = 19 = u[i]
                  fixed_lower_range)
  #mdl.write_to_file('model.lp')
  # with open('model.lp', 'r') as thefile:
  #   lines = thefile.readlines()
  #   for l in lines: print(l)
  mdl.solve()
  mdl.print_ranges(lower_original_value, upper_original_value)
  score =[c[i] for i in mdl.v if mdl.v[i].X>=0.5]
  print(score)
  print(sum(score), sum(score)/len(score))

  local_test=test_dataset
  for i in range(len(columns)):
    local_test=local_test[(local_test[columns[i]]>=mdl.l[i].X) & (local_test[columns[i]]<=mdl.u[i].X)]
  y_pred_loc=mod.predict(local_test.drop(target[Dataset.value], axis=1))

  if model == 'Random forest':
    explainer = shap.TreeExplainer(mod, train_dataset.drop([target[Dataset.value]], axis=1))
    shap_values = explainer.shap_values(local_test.drop([target[Dataset.value]], axis=1), check_additivity=False)
    shap.summary_plot(shap_values[:,:,1], local_test.drop([target[Dataset.value]], axis=1))
  elif model == 'Logistic regression':
    explainer = shap.LinearExplainer(mod, train_dataset.drop([target[Dataset.value]], axis=1), check_additivity=False)
    shap_values = explainer.shap_values(local_test.drop([target[Dataset.value]], axis=1))
    shap.summary_plot(shap_values[:,:], local_test.drop([target[Dataset.value]], axis=1))
  else:
     pass
  return [mdl.l[i].X for i in range(len(columns))], [mdl.u[i].X for i in range(len(columns))]


def tab_optimizer_high(model, dataset, beta): #, train_dataset, test_dataset, columns, target, sliders): #, mod, model): #in moduli
  start = time.time()

  if dataset == 'campione':
      dataset1 = 'campione'
      dataset2 = 'titanic'
      target = {dataset1: 'Recidivist', dataset2: 'Survived'}
      columns_of_interest = {dataset2: ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Survived' ], dataset1: ['Age at entry', 'Length of imprisonment', 'Numbers of entries', 'Citizenship', 'Treatment', 'Recidivist']}
      columns_for_analysis = {dataset1: ['Age at entry', 'Length of imprisonment', 'Numbers of entries', 'Citizenship', 'Treatment'], dataset2: ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch'] }

      df=pd.read_excel('%s.xlsx' % dataset, sheet_name='Sheet2')
      np.random.seed(1)
      df = df.sample(frac=1)
      columns = df.columns.tolist()
      X_tot=df[columns_of_interest[dataset1]]
      X=df[columns_for_analysis[dataset1]]
      y=df[target[dataset1]]
      train_dataset, test_dataset, _, _ = train_test_split(X_tot, y, test_size=0.3, random_state=1, stratify=y)
      lower,upper = None, None
      if dataset in columns_for_analysis:
          columns = columns_for_analysis[dataset]
      else:
          columns = columns_of_interest[dataset]
      sliders = {}
      for i,col in enumerate(columns):
          col_values = df[col]
          col_min = min(col_values)
          col_max = max(col_values)
          slider = widgets.IntRangeSlider(min=col_min, max=col_max, step=1, description='%s:' % col, value=[col_min  if lower is None else lower[i], col_max if upper is None else upper[i]])
          sliders[col] = slider
  elif dataset == 'titanic':
      dataset1 = 'campione'
      dataset2 = 'titanic'
      target = {dataset1: 'Recidivist', dataset2: 'Survived'}
      columns_of_interest = {dataset2: ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Survived' ], dataset1: ['Age at entry', 'Length of imprisonment', 'Numbers of entries', 'Citizenship', 'Treatment', 'Recidivist']}
      columns_for_analysis = {dataset1: ['Age at entry', 'Length of imprisonment', 'Numbers of entries', 'Citizenship', 'Treatment'], dataset2: ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch'] }

      df=pd.read_excel('%s.xlsx' % dataset, sheet_name='Sheet2')
      np.random.seed(1)
      df = df.sample(frac=1)
      columns = df.columns.tolist()
      X_tot=df[columns_of_interest[dataset2]]
      X=df[columns_for_analysis[dataset2]]
      y=df[target[dataset2]]
      train_dataset, test_dataset, _, _ = train_test_split(X_tot, y, test_size=0.3, random_state=1, stratify=y)
      lower,upper = None, None
      if dataset in columns_for_analysis:
          columns = columns_for_analysis[dataset]
      else:
          columns = columns_of_interest[dataset]
      sliders = {}
      for i,col in enumerate(columns):
          col_values = df[col]
          col_min = min(col_values)
          col_max = max(col_values)
          slider = widgets.IntRangeSlider(min=col_min, max=col_max, step=1, description='%s:' % col, value=[col_min  if lower is None else lower[i], col_max if upper is None else upper[i]])
          sliders[col] = slider

  np.random.seed(1)
  up_original = []
  for i in range(len(columns)):
    up_original.append(sliders[columns[i]].value[1])
  print(np.array(up_original))
  low_original = []
  for i in range(len(columns)):
    low_original.append(sliders[columns[i]].value[0])
  print(np.array(low_original))

  class range_model_accuracy():
    GAMMA = 0.99
    eps = 1e-4

    def __init__(self,
                 x, # the dataset, x[i][k] is the k-th feature of the i-th sample
                 n_of_samples,  # lenght of the dataset
                 n_of_features, # number of features not considering the target
                 c, # a list of binary values representing the correct classification. e.g. c[i] = 1 if psi(x[i]) = y[i]
                 M, # a list of length n_of_features where M[i] = 1+ maximum of the absolute value in the range. E.g. for binary features {0,1} we have M = 2
                 beta, # minimum threshold for the accuracy
                 lower_original_value,
                 upper_original_value,
                 upper_bounds = {}, # a dictionary of upper bounds for upper bound range variables.E.g. lower_bounds[i] = 20 >= u[i]
                 lower_bounds ={}, # a dictionary of lower bounds for lower bound range variables.E.g. upper_bounds[i] = 10 <= l[i]
                 fixed_upper_range ={}, # a dictionary of fixed values for the upper range variables. E.g. fixed_upper_range[i] = 19 = u[i]
                 fixed_lower_range={} # a dictionary of fixed values for the lower range variables. E.g. fixed_lower_range[i] = 12 = l[i]
                 ):

        params = {
        "WLSACCESSID": 'cebd302e-fd24-4581-9b76-95e0001e5d28',
        "WLSSECRET": 'c966bb4d-17c6-4203-8be4-7055a9b12227',
        "LICENSEID": 2442013,
        }
        env = gp.Env(params=params)

        self._mdl = gp.Model(env=env)

        self._mdl.params.FeasibilityTol = 1e-6
        self._mdl.params.TimeLimit = 600
        #self._mdl.params.MIPFocus = 1
        #self._mdl.params.RINS = 3



        self.v = self._mdl.addVars(n_of_samples, vtype = GRB.BINARY, name = "v-")
        self.p = self._mdl.addVars([(i,k) for i in range(n_of_samples) for k in range(n_of_features)], vtype=GRB.BINARY, name="p-")
        self.sup = self._mdl.addVars([(i,k) for i in range(n_of_samples) for k in range(n_of_features)], vtype = GRB.BINARY, name = "sup-")

        ## range variables ##
        #- u upper
        self.u = self._mdl.addVars(n_of_features, vtype=GRB.CONTINUOUS, name="u-")
        #- l lower
        self.l = self._mdl.addVars(n_of_features, vtype=GRB.CONTINUOUS, name="l-")


        ### CONSTRAINTS ###
        self.c = np.asarray(c)

        self.threshold_constraint = self._mdl.addConstr( gp.quicksum([self.v[i]*self.c[i] for i in range(n_of_samples)]) >= beta*self.v.sum(),
                                                         name='threshold_constraint')

        self.bigM_lower = self._mdl.addConstrs(
            (-M[k] * (1 - self.p[i, k]) + self.l[k] <= x[i][k] * self.p[i, k]
              for i in range(n_of_samples) for k in range(n_of_features)),
              name='bigM_lower-')

        self.bigM_upper = self._mdl.addConstrs(
            (M[k] * (1 - self.p[i, k]) + self.u[k] >= x[i][k] *self.p[i, k]
              for i in range(n_of_samples) for k in range(n_of_features)),
              name='bigM_upper-')

        self.eps_lower = self._mdl.addConstrs(
            (M[k] * ( self.p[i, k]+self.sup[i, k])-range_model_accuracy.eps* (1 - self.p[i, k])+ self.l[k] >= x[i][k]
              for i in range(n_of_samples) for k in range(n_of_features)),
              name='eps_lower-')

        self.eps_upper = self._mdl.addConstrs(
            (-M[k] * (self.p[i, k] + 1 - self.sup[i, k]) + range_model_accuracy.eps * (1 - self.p[i, k]) + self.u[k] <=
             x[i][k]
             for i in range(n_of_samples) for k in range(n_of_features)),
            name='eps_upper-')

        self.v_upper = self._mdl.addConstrs(
            (self.v[i] * n_of_features <= gp.quicksum(self.p[i, k] for k in range(n_of_features))
             for i in range(n_of_samples)),
            name='v_upper-')

        self.v_lower = self._mdl.addConstrs(
            (self.v[i] >=  gp.quicksum(self.p[i, k] for k in range(n_of_features)) - n_of_features + 1
             for i in range(n_of_samples)),
            name='v_lower-')

        self.basic_ordering = self._mdl.addConstrs(
            (self.l[k] <= self.u[k] for k in range(n_of_features)),
            name='upper_bounds_on_upper_range-')

        self.upper_bounds_on_upper_range = self._mdl.addConstrs(
            (self.u[k] <= upper_bounds[k] for k in upper_bounds),
            name='upper_bounds_on_upper_range-')

        self.lower_bounds_on_lower_range = self._mdl.addConstrs(
            (self.l[k] >= lower_bounds[k] for k in lower_bounds),
            name='lower_bounds_on_lower_range-')

        self.fixed_upper_range = self._mdl.addConstrs(
            (self.u[k] == fixed_upper_range[k] for k in fixed_upper_range),
            name='fixed_upper_range-')

        self.fixed_lower_range = self._mdl.addConstrs(
            (self.l[k] == fixed_lower_range[k] for k in fixed_lower_range),
            name='fixed_lower_range-')
        range_length = [abs(upper_original_value[i] - lower_original_value[i]) for i in range(n_of_features)]
        self.range_length = range_length
        self.n_of_features = n_of_features

        self._mdl.setObjective(1+self.v.sum()-gp.quicksum([(self.u[i] - self.l[i])/range_length[i] for i in range(n_of_features)])/(1 + n_of_features), GRB.MAXIMIZE)

        self._mdl.update()

        return

    def pobj(self):
        return gp.quicksum([(self.u[i] - self.l[i])/self.range_length[i] for i in range(self.n_of_features)])/(1 + self.n_of_features)

    def write_to_file(self, pathtofile):
        self._mdl.write(pathtofile)



    def solve(self):
        self._mdl.optimize()

    def print_ranges(self, lower_original_value, upper_original_value):
        for i in range(len(self.l)):
            print(lower_original_value[i], '<=  ', self.l[i].X, '   --   ', self.u[i].X, '<=  ',
                  upper_original_value[i])
    def ranges(self):
        l, u = [], []
        for i in range(len(self.l)):
            l.append( self.l[i].X)
            u.append(self.u[i].X)
        return l, u


  n_of_samples = test_dataset.shape[0]
  n_of_features = len(sliders)
  x = test_dataset.drop(target[dataset], axis=1).to_numpy()
  #y_pred #_tot=clf.predict(df[columns].to_numpy())
  #y_pred=mod.predict(test_dataset.drop(target[Dataset.value], axis=1))
  if model == 'Random forest':
    mod=RandomForestClassifier(n_estimators=100, random_state=1)#
    mod.fit(train_dataset.drop([target[dataset]], axis=1),train_dataset[target[dataset]])#
    y_pred=mod.predict(test_dataset.drop(target[dataset], axis=1))#
  elif model == 'Logistic regression':
    mod = LogisticRegression(random_state=1).fit(train_dataset.drop([target[dataset]], axis=1),train_dataset[target[dataset]])#
    y_pred=mod.predict(test_dataset.drop(target[dataset], axis=1))#
  else:
     pass
  c=np.abs(y_pred-test_dataset[target[dataset]].to_numpy())<=1e-3
  #beta  = 0.70
  M = [max((abs(i), abs(j))) for i,j in zip(up_original, low_original)]
  upper_bounds = {}
  lower_bounds = {}
  fixed_upper_range = {}
  fixed_lower_range = {}
  lower_original_value = np.array(low_original)
  upper_original_value = np.array(up_original)

  mdl = range_model_accuracy(x, # the dataset, x[i][k] is the k-th feature of the i-th sample
                  n_of_samples,  # lenght of the dataset
                  n_of_features, # number of features not considering the target
                  c, # a list of binary values representing the correct classification. e.g. c[i] = 1 if psi(x[i]) = y[i]
                  M, # a list of length n_of_features where M[i] = 1+ maximum of the absolute value in the range. E.g. for binary features {0,1} we have M = 2
                  beta, # minimum threshold for the accuracy
                  lower_original_value,
                  upper_original_value,
                  upper_bounds, # a dictionary of upper bounds for upper bound range variables.E.g. lower_bounds[i] = 20 >= u[i]
                  lower_bounds, # a dictionary of lower bounds for lower bound range variables.E.g. upper_bounds[i] = 10 <= l[i]
                  fixed_upper_range , # a dictionary of fixed values for the upper range variables. E.g. fixed_upper_range[i] = 19 = u[i]
                  fixed_lower_range)
  #mdl.write_to_file('model.lp')
  # with open('model.lp', 'r') as thefile:
  #   lines = thefile.readlines()
  #   for l in lines: print(l)
  mdl.solve()
  mdl.print_ranges(lower_original_value, upper_original_value)
  score =[c[i] for i in mdl.v if mdl.v[i].X>=0.5]
  print(score)
  print(sum(score), sum(score)/len(score))

  local_test=test_dataset
  for i in range(len(columns)):
    local_test=local_test[(local_test[columns[i]]>=mdl.l[i].X) & (local_test[columns[i]]<=mdl.u[i].X)]
  y_pred_loc=mod.predict(local_test.drop(target[dataset], axis=1))

  if model == 'Random forest':
    explainer = shap.TreeExplainer(mod, train_dataset.drop([target[dataset]], axis=1))
    shap_values = explainer.shap_values(local_test.drop([target[dataset]], axis=1), check_additivity=False)
    shap.summary_plot(shap_values[:,:,1], local_test.drop([target[dataset]], axis=1))
  elif model == 'Logistic regression':
    explainer = shap.LinearExplainer(mod, train_dataset.drop([target[dataset]], axis=1), check_additivity=False)
    shap_values = explainer.shap_values(local_test.drop([target[dataset]], axis=1))
    shap.summary_plot(shap_values[:,:], local_test.drop([target[dataset]], axis=1))
  else:
     pass
  end = time.time()
  sec=(end - start)
  points=len(score)
  print(sec, points)
  #return [mdl.l[i].X for i in range(len(columns))], [mdl.u[i].X for i in range(len(columns))]
  return sec, points