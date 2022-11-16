import math
import numpy as np
import pandas as pd
import sys
'''
new approach: 
in the train step, compute class conditional densities
use kernels to estimate probability of x
'''

class NaiveBayesClassifier:
  
  def __init__(self, kernel_type, h):
    if(kernel_type == 'hypercube'):
      self.kernel = self.hypercube
    if(kernel_type == 'rbf'):
      self.kernel = self.rbf
    self.h = h
    

  def hypercube(self, x):
    for i in range(len(x)):
      if((math.isnan(x[i])) or (x[i] == 0)):
        continue
      if(np.log(abs(float(x[i]))) >= .5):
        return 0
    return 1

  def rbf(self, x):
    #xTx = np.dot(x,x)
    xTx = 0
    for i in range(len(x)):
      if(math.isnan(x[i])):
        continue
      xTx += abs(float(x[i]))**2
    xTx = math.sqrt(xTx)
    result = math.pow(2*math.pi,(-self.dim/2)) * math.pow(math.e,-(1/2)*xTx)
    #print("rbf returning {}".format(result))
    return result

  def pdke(self, x):
    result = 0
    for i in range(self.N):
      #print("x before={}".format(x))
      result += self.kernel((x-self.test_data[i])/self.h)
      p = result/(self.N*self.h**self.dim)
    return p

  def fit(self, training_file):
    train_df = pd.read_csv(training_file)
    train_df.pop('game_num')
    train_df.pop('event_id')
    self.dim = train_df.shape[1]
    self.N = train_df.shape[0]

    self.priors = dict()

    self.priors['A'] =  (np.array(train_df['team_scoring_next'])=='A').sum() / self.N
    self.priors['B'] =  (np.array(train_df['team_scoring_next'])=='B').sum() / self.N

    if(self.priors['A'] == 0):
      self.priors['A'] = 1 - self.priors['B']

    if(self.priors['B'] == 0):
      self.priors['B'] = 1 - self.priors['A']
    




  def predict(self, testing_file):
    test_df = pd.read_csv(testing_file)
    self.test_data = np.array(test_df)

    self.N = test_df.shape[0]

    

    submission_df = pd.DataFrame(columns=["id","team_A_scoring_within_10sec","team_B_scoring_within_10sec"])

    final_labels = []

    for i in range(len(test_df)):
      c = abs(test_df['ball_pos_x'][i])
      pa_x = 0
      pb_x = 0

      pkde = self.pdke(self.test_data[i])

      pa_x = pkde * self.priors['A'] * c

      pb_x = pkde * self.priors['B'] * c

      final_labels.insert(i,[i,pa_x,pb_x])
      

    #final_labels = self.normalize(final_labels)

    for i in range(len(test_df)):
      submission_df = submission_df.append({'id': int(final_labels[i][0]),
                                            'team_A_scoring_within_10sec' : final_labels[i][1],
                                            'team_B_scoring_within_10sec' : final_labels[i][2]}, ignore_index = True)

    submission_df['team_A_scoring_within_10sec'] = submission_df['team_A_scoring_within_10sec'].astype(str)
    submission_df['team_B_scoring_within_10sec'] = submission_df['team_B_scoring_within_10sec'].astype(str)
    submission_df['team_A_scoring_within_10sec'] = submission_df['team_A_scoring_within_10sec'].str.slice(0,4)
    submission_df['team_B_scoring_within_10sec'] = submission_df['team_B_scoring_within_10sec'].str.slice(0,4)
    submission_df['team_A_scoring_within_10sec'] = submission_df['team_A_scoring_within_10sec'].astype(float)
    submission_df['team_B_scoring_within_10sec'] = submission_df['team_B_scoring_within_10sec'].astype(float)
    submission_df['team_A_scoring_within_10sec'] = submission_df['team_A_scoring_within_10sec']/10
    submission_df['team_B_scoring_within_10sec'] = submission_df['team_B_scoring_within_10sec']/10
    submission_df['team_A_scoring_within_10sec'] = submission_df['team_A_scoring_within_10sec'].astype(str)
    submission_df['team_B_scoring_within_10sec'] = submission_df['team_B_scoring_within_10sec'].astype(str)
    submission_df['team_A_scoring_within_10sec'] = submission_df['team_A_scoring_within_10sec'].str.slice(0,5)
    submission_df['team_B_scoring_within_10sec'] = submission_df['team_B_scoring_within_10sec'].str.slice(0,5)

    submission_df['id'] = submission_df['id'].astype(int)
    submission_df.to_csv('out.csv', index=False)  

  def normalize(self, arr):
    result = []
    curr_min = sys.maxsize
    curr_max = -sys.maxsize
    
    for i in range(len(arr)):
      a = arr[i][1]
      b = arr[i][2]
      if(min(a,b) < curr_min):
        curr_min = min(a,b)
      if(max(a,b) > curr_max):
        curr_max = max(a,b)

    new_min = 0
    new_max = 1


    for i in range(len(arr)):
      new_a = ((a-curr_min)/(curr_max-curr_min))*(new_max-new_min)+new_min
      new_b = ((b-curr_min)/(curr_max-curr_min))*(new_max-new_min)+new_min
      result.insert(i,(arr[i][0],new_a,new_b))

    return result

nb1 = NaiveBayesClassifier('rbf', .001)
nb1.fit('../train0/train_0_tail_10k.csv')

nb1.predict('../test.csv')
