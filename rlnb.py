import numpy as np
import pandas as pd
import math
import sys

def fit(training_file):
  '''
  new approach, here i'm not using event_id or game_num as a feature
  '''
  df = pd.read_csv(training_file)

  df.pop('boost0_timer')
  df.pop('boost1_timer')
  df.pop('boost2_timer')
  df.pop('boost3_timer')
  df.pop('boost4_timer')
  df.pop('boost5_timer')

  df.pop('p0_boost')
  df.pop('p1_boost')
  df.pop('p2_boost')
  df.pop('p3_boost')
  df.pop('p4_boost')
  df.pop('p5_boost')

  df.pop('p0_pos_z')
  df.pop('p1_pos_z')
  df.pop('p2_pos_z')
  df.pop('p3_pos_z')
  df.pop('p4_pos_z')
  df.pop('p5_pos_z')

  df.pop('p0_vel_z')
  df.pop('p1_vel_z')
  df.pop('p2_vel_z')
  df.pop('p3_vel_z')
  df.pop('p4_vel_z')
  df.pop('p5_vel_z')

  df.pop('ball_pos_z')
  df.pop('ball_vel_z')

  # event x_a1 - ball closer to team A goal
  # event x_a2 - ball closer to team B goal
  # event x_b1 - ball colliding with team A player
  # event x_b2 - ball colliding with team B player
  # event x_c1 - ball rolling toward team A goal
  # event x_c2 - ball rolling toward team B goal
    




  #for each kickoff, let's consider our three events in an event table

  event_table=pd.DataFrame(columns=["ball_loc", "x_a1", "x_a2", "x_b1", "x_b2", "x_c1", "x_c2", "team_scoring_in_10s"])

  #for all snapshots in the event, we need to determine if any x_a,x_b,or x_c occurring
  for i in range(len(df)):
    

    #look through original dataframe
    
    df_row = []
    df_row_check = []
    


    #determine which side of field the ball is on
    if(df['ball_pos_x'][i] < 0): 
      df_row.insert(2,'A')
      df_row_check.insert(2,'ball_loc')

    if(df['ball_pos_x'][i] > 0): 
      df_row.insert(2,'B')
      df_row_check.insert(2,'ball_loc')

    if(df['ball_pos_x'][i] == 0): 
      df_row.insert(2,math.nan)
      df_row_check.insert(2,'ball_loc')

    min_ball_goal_dist = 5
    #is ball close to team A goal
      ##assume team A goal at x = -80 
    d_ball_goal_a = math.sqrt((df['ball_pos_x'][i]+80)**2 + df['ball_pos_y'][i]**2)

    
    
    if(d_ball_goal_a <= min_ball_goal_dist):
      df_row.insert(3,'yes')
      df_row_check.insert(3,'x_a1')

    if(d_ball_goal_a > min_ball_goal_dist):
      df_row.insert(3,'no')
      df_row_check.insert(3,'x_a1')


    #is ball close to team B goal
      ##assume team B goal at x = 80
    d_ball_goal_b = math.sqrt((df['ball_pos_x'][i]-80)**2 + df['ball_pos_y'][i]**2)
    
    if(d_ball_goal_b <= min_ball_goal_dist):
      df_row.insert(4,'yes')
      df_row_check.insert(4,'x_a2')

    if(d_ball_goal_b > min_ball_goal_dist):
      df_row.insert(4,'no')
      df_row_check.insert(4,'x_a2')




    #is ball colliding with player

    #nearest_player_idx = index of nearest player
    ##need to compute nearest player
    
    min_pbd = sys.maxsize
    min_pbi = -1
    for n in range(0,6):
      player_str = 'p{ind}'.format(ind = n)
      player_str_x = player_str +  '_pos_x'
      player_str_y = player_str +  '_pos_y'
      currdist = math.sqrt((df['ball_pos_x'][i] - df[player_str_x][i])**2 + (df['ball_pos_y'][i]-df[player_str_y][i])**2)
      if(currdist < min_pbd):
        min_pbd = currdist
        min_pbi = n

    nearest_player_idx = min_pbi
    d_player_ball = min_pbd

    min_ball_player_dist = 5

    player_on_team_a = (nearest_player_idx == 0) or (nearest_player_idx == 1) or (nearest_player_idx == 2)

    player_on_team_b = (nearest_player_idx == 3) or (nearest_player_idx == 4) or (nearest_player_idx == 5)


    #is ball colliding with team A player
    if((d_player_ball <= min_ball_player_dist) and player_on_team_a):
      df_row.insert(5,'yes')
      df_row_check.insert(5,'x_b1')
      df_row.insert(6,'no')
      df_row_check.insert(6,'x_b2')



    #is ball colliding with team B player
    if((d_player_ball <= min_ball_player_dist) and player_on_team_b):
      df_row.insert(5,'no')
      df_row_check.insert(5,'x_b1')
      df_row.insert(6,'yes')
      df_row_check.insert(6,'x_b2')


    #no player is currently colliding with ball
    if(d_player_ball > min_ball_player_dist):
      df_row.insert(5,'no')
      df_row_check.insert(5,'x_b1')
      df_row.insert(6,'no')
      df_row_check.insert(6,'x_b2')


    


    #is ball rolling toward player A goal
    if(df['ball_vel_x'][i] < 0):
        df_row.insert(7,'yes')
        df_row_check.insert(7,'x_c1')
        df_row.insert(8,'no')
        df_row_check.insert(8,'x_c2')


    #is ball rolling toward player B goal
    if(df['ball_vel_x'][i] > 0):
        df_row.insert(7,'no')
        df_row_check.insert(7,'x_c2')
        df_row.insert(8,'yes')
        df_row_check.insert(8,'x_c2')


    #ball is not moving
    if(df['ball_vel_x'][i] == 0):
        df_row.insert(7,'no')
        df_row_check.insert(7,'x_c1')
        df_row.insert(8,'no')
        df_row_check.insert(8,'x_c2')


    #determine which team will score in next ten seconds
    if(df['team_A_scoring_within_10sec'][i] == 1):
      df_row.insert(9,'A')
      df_row_check.insert(9,'team_scoring_in_10s') 

    if(df['team_B_scoring_within_10sec'][i] == 1):
      df_row.insert(9,'B')
      df_row_check.insert(9,'team_scoring_in_10s')
    
    neither_team_scoring_wts = (df['team_A_scoring_within_10sec'][i] == 0) and (df['team_B_scoring_within_10sec'][i] == 0)

    if(neither_team_scoring_wts):
      df_row.insert(9,math.nan)
      df_row_check.insert(9,'team_scoring_in_10s')

    

    event_table = event_table.append({'ball_loc': df_row[0],
                                        'x_a1' : df_row[1],
                                        'x_a2' : df_row[2],
                                        'x_b1' : df_row[3],
                                        'x_b2' : df_row[4],
                                        'x_c1' : df_row[5],
                                        'x_c2' : df_row[6],
                                        'team_scoring_in_10s' : df_row[7]}, ignore_index = True)

  row_names=["x_a1", "x_a2", "x_b1", "x_b2", "x_c1", "x_c2","team A swts", "team B swts"]
  col_names=["num_yes","num_no","p_yes","p_no"]
  data = [[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]
  data[0][0] = (np.array(event_table['x_a1']) == 'yes').sum()
  data[0][1] = (np.array(event_table['x_a1']) == 'no').sum()
  data[0][2] = data[0][0]/len(event_table['x_a1'])
  data[0][3] = data[0][1]/len(event_table['x_a1'])

  data[1][0] = (np.array(event_table['x_a2']) == 'yes').sum()
  data[1][1] = (np.array(event_table['x_a2']) == 'no').sum()
  data[1][2] = data[1][0]/len(event_table['x_a2'])
  data[1][3] = data[1][1]/len(event_table['x_a2'])


  data[2][0] = (np.array(event_table['x_b1']) == 'yes').sum()
  data[2][1] = (np.array(event_table['x_b1']) == 'no').sum()
  data[2][2] = data[2][0]/len(event_table['x_b1'])
  data[2][3] = data[2][1]/len(event_table['x_b1'])


  data[3][0] = (np.array(event_table['x_b2']) == 'yes').sum()
  data[3][1] = (np.array(event_table['x_b2']) == 'no').sum()
  data[3][2] = data[3][0]/len(event_table['x_b2'])
  data[3][3] = data[3][1]/len(event_table['x_b2'])


  data[4][0] = (np.array(event_table['x_c1']) == 'yes').sum()
  data[4][1] = (np.array(event_table['x_c1']) == 'no').sum()
  data[4][2] = data[4][0]/len(event_table['x_c1'])
  data[4][3] = data[4][1]/len(event_table['x_c1'])

  data[5][0] = (np.array(event_table['x_c2']) == 'yes').sum()
  data[5][1] = (np.array(event_table['x_c2']) == 'no').sum()
  data[5][2] = data[5][0]/len(event_table['x_c2'])
  data[5][3] = data[5][1]/len(event_table['x_c2'])

  data[6][0] = (np.array(event_table['team_scoring_in_10s']) == 'A').sum()
  data[6][1] = abs(data[6][0]-len(event_table['team_scoring_in_10s'])) 
  data[6][2] = data[6][0]/len(event_table['team_scoring_in_10s'])
  data[6][3] = data[6][1]/len(event_table['team_scoring_in_10s'])

  data[7][0] = (np.array(event_table['team_scoring_in_10s']) == 'B').sum()
  data[7][1] = abs([7][0]-len(event_table['team_scoring_in_10s']))
  data[7][2] = data[7][0]/len(event_table['team_scoring_in_10s'])
  data[7][3] = data[7][1]/len(event_table['team_scoring_in_10s'])

  freq_table=pd.DataFrame(data, row_names, col_names)

  return freq_table

def predict(testing_file,freq_table):
  '''
  compute posterior probabilities using bayes theorem P(C|X) ~ P(X|C)*P(C)
  '''
  test_df = pd.read_csv(testing_file)
  #P(A-swts | x_A1, x_A2, x_B1, x_B2,x_C1, x_C2) ~ P(x_A1)*P(x_A2)*P(x_B1)*P(x_B2)*P(x_C1)*P(x_C2)*P(A-stws)
  #P(B-swts | x_A1, x_A2, x_B1, x_B2,x_C1, x_C2) ~ P(x_A1)*P(x_A2)*P(x_B1)*P(x_B2)*P(x_C1)*P(x_C2)*P(B-stws)

  final_labels = []


  submission_df = pd.DataFrame(columns=["id","team_A_scoring_within_10sec","team_B_scoring_within_10sec"])

  for i in range(len(test_df)):
    pa_x = freq_table['p_yes'][0] * freq_table['p_yes'][1] * freq_table['p_yes'][2] * freq_table['p_yes'][3] \
                    * freq_table['p_yes'][4] * freq_table['p_yes'][5] * freq_table['p_yes'][6] 
    pb_x = freq_table['p_yes'][0] * freq_table['p_yes'][1] * freq_table['p_yes'][2] * freq_table['p_yes'][3] \
                    * freq_table['p_yes'][4] * freq_table['p_yes'][5] * freq_table['p_yes'][7] 

    

    final_labels.insert(i,[i,pa_x,pb_x])
    
    submission_df = submission_df.append({'id': int(final_labels[i][0]),
                                          'team_A_scoring_within_10sec' : final_labels[i][1],
                                          'team_B_scoring_within_10sec' : final_labels[i][2]}, ignore_index = True)


  submission_df.to_csv('out.csv')  


freq_table = fit('../train0/train_0_tail_10k.csv')

predict('test.csv',freq_table)
