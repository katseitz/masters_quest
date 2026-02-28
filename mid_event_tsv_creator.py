import pandas as pd
import numpy as np
import glob
import os 
import traceback

save_dir = '/Users/katharinaseitz/Documents/projects/gimme_first_year/data/crest_events/'

mid = 1
make_plot = 0
save_output = 0


#TO-DO: Do file testing here 

counts = pd.DataFrame()

def happy_mid(path):
    """
    Helper function that launches MID eprime txt file
    conversion for subjects with one txt file containing
    both MID1 and MID2
    ----------
    string : str
        Path of good MID file
    Returns
    -------
    nothing
    """
    subject = path.split("/")[-4]
    df = file_to_df(path)
    df_to_timing_txt(df, subject, 1)


def sus_out_files(path): #= dirs[10]
    """
    Iterates through a participants MID output files and 
    checks to see whether MID runs were split.
    ----------
    string : str #TODO update here
        String from which to remove unicode characters.
    Returns
    -------
    nothing
    """   
    files =  glob.glob(path +'/ses-1/beh/3_MID*.txt')
    midshared_files = []
    mid1_files = []
    mid2_files = []
    
    if(len(files) == 0):
        print("No MID files or path DNE " + path)
        return
    
    for file in files: 
        #print(file)
        df = file_to_df(file)
        if('Run1Cue.OnsetTime' in df[0].unique() and 'Run2Cue.OnsetTime' in df[0].unique()):
            midshared_files.append([file, len(df)])
        elif('Run1Cue.OnsetTime' in df[0].unique()):
            mid1_files.append([file, len(df)])
        elif('Run2Cue.OnsetTime' in df[0].unique()):
            mid2_files.append([file, len(df)])
        
        #best case - one eprime file with run1 and run2...
        #TODO bullet proof this logic
    #print(midshared_files)
    if(len(midshared_files) == 1): #and len(mid1_files) == 0 and len(mid1_files) == 0): 
        happy_mid(midshared_files[0][0])
    elif(len(mid1_files) > 0 and len(mid2_files) > 0):
        print("this path has two split MID runs " + path)
    elif(len(mid1_files) > 1):
        print("this path only has MID one " + path)
    else:
        print("this path has confusing mid files " + path)


def remove_unicode(string):
    """
    Removes unicode characters in string.
    Parameters
    ----------
    string : str
        String from which to remove unicode characters.
    Returns
    -------
    str
        Input string, minus unicode characters.
    """
    return ''.join([val for val in string if 31 < ord(val) < 127])


def file_to_df(text_file):
    with open(text_file, 'rb') as fo:
        text_data = list(fo)

    # Remove unicode characters.
    filtered_data = [remove_unicode(row.decode('utf-8', 'ignore')) for row in text_data]
    list_data = []
    for row in filtered_data: 
        if ":" in row:
            row_as_list = row.split(": ")
            list_data.append(row_as_list)
            
        else:
            list_data.append([row])
    df = pd.DataFrame(list_data)
    return df
    
def df_to_timing_txt(df, subject, mid2=0):
    print(mid2)
    """
    add info here
    ----------
    df : str
        String from which to remove unicode characters.
    subject : str
        String from which to remove unicode characters.
    mid2 : str
        String from which to remove unicode characters.
    Returns
    -------
    str
        Input string, minus unicode characters.
    """
    global counts
    #### MID RUN 1
    #### TRIAL TYPES
    trial_type1 = df.loc[df[0] == 'RunList1'][1]+ "-" 
    trial_type1 = trial_type1.replace('1-','win_5')
    trial_type1 = trial_type1.replace('2-','win_1.5')
    trial_type1 = trial_type1.replace('3-','win_0')
    trial_type1 = trial_type1.replace('4-','lose_5')
    trial_type1 = trial_type1.replace('5-','lose_1.5')
    trial_type1 = trial_type1.replace('6-','lose_0')
    trial_type1 = trial_type1.reset_index()
    
    #### ANTICIPATION PHASE
    cue_on1 = df.loc[df[0] == 'Run1Cue.OnsetTime'][1].astype(int).reset_index() 
    cue_on1 = (cue_on1 - int(df.loc[df[0] == 'Run1Fix.OnsetTime'].iloc[0][1])) / 1000  #We think this is the first fixation
    #Anticipation Phase data frane
    MID_ant=pd.concat([cue_on1[1], "ant_" + trial_type1[1]], axis = 1)
    MID_ant["duration"] = 2
    MID_ant.columns =['onset', 'trial_type', 'duration']
    
    
    #### REWARD PHASE
    #TODO: Run1Tgt.ACCfeedback
    fbk_on1 = df.loc[df[0] == 'Run1Fbk.OnsetTime'][1].astype(int).reset_index() 
    fbk_on1 = (fbk_on1 -  int(df.loc[df[0] == 'Run1Fix.OnsetTime'].iloc[0][1])) / 1000 
    rt1 = df.loc[df[0] == 'Run1Tgt.RT'][1].astype(int).reset_index() #response times to guage accuracy
    rt1 = rt1.replace(0, np.nan) 
    acc1 = ~(rt1.isna())  #Was the participant accurate?
    acc1 = acc1.replace({True: 'Hit', False: 'Miss'})
    #Reward Phase data frame
    MID_rew=pd.concat([fbk_on1[1]],axis=1)
    MID_rew['trial_type'] = "rew_"+trial_type1[1]+ "_" + acc1[1]
    MID_rew["duration"] = 2
    MID_rew.columns =['onset', 'trial_type', 'duration']
    
    
    #### BUTTON PRESS
    target_onset1 = df.loc[df[0] == 'Run1Tgt.OnsetTime'][1].astype(int).reset_index() 
    target_onset1 = (target_onset1 -  int(df.loc[df[0] == 'Run1Fix.OnsetTime'].iloc[0][1])) / 1000
    MID_press=pd.concat([target_onset1[1]],axis=1)
    MID_press['trial_type'] = "motor_period"
    duration = (df.loc[df[0] == 'Run1Tgt.Duration'][1].astype(float)/ 1000).reset_index()
    MID_press = pd.concat([MID_press, duration[1]], axis = 1, ignore_index=True)
    MID_press.columns =['onset', 'trial_type', 'duration']
    
    #### three fixations
    
     #anticipation fixation
    fix_on1 = df.loc[df[0] == 'Run1Dly.OnsetTime'][1].astype(int).reset_index() 
    fix_on1 = (fix_on1 - int(df.loc[df[0] == 'Run1Fix.OnsetTime'].iloc[0][1])) / 1000  #We think this is the first fixation
    #Anticipation Fixation Phase data frane
    MID_ant_fix=pd.concat([fix_on1[1], "ant_fix_" + trial_type1[1]], axis = 1)
    duration = (df.loc[df[0] == 'Run1Dly.Duration'][1].astype(float)/ 1000).reset_index()# MID_ant_fix["duration"]
    MID_ant_fix = pd.concat([MID_ant_fix, duration[1]], axis = 1, ignore_index=True)
    MID_ant_fix.columns =['onset', 'trial_type', 'duration']
    
    #probe fixation
    fix_on2 = df.loc[df[0] == 'Run1Dly2.OnsetTime'][1].astype(int).reset_index() 
    fix_on2 = (fix_on2 - int(df.loc[df[0] == 'Run1Fix.OnsetTime'].iloc[0][1])) / 1000  #We think this is the first fixation
    #Anticipation Fixation Phase data frane
    MID_probe_fix=pd.concat([fix_on2[1], "probe_fix_" + trial_type1[1]], axis = 1)
    duration2 = (df.loc[df[0] == 'Run1Dly2.Duration'][1].astype(float)/ 1000).reset_index()# MID_ant_fix["duration"]
    MID_probe_fix = pd.concat([MID_probe_fix, duration2[1]], axis = 1, ignore_index=True)
    MID_probe_fix.columns =['onset', 'trial_type', 'duration']

    
    #reward fixation
    fix_on3 = fbk_on1 + 2
    #Anticipation Fixation Phase data frane
    MID_reward_fix=pd.concat([fix_on3[1], "rew_fix_" + trial_type1[1]], axis = 1)
    #print(MID_reward_fix)
    MID_reward_fix.columns =['onset', 'trial_type']
    #print(MID_ant['onset'])
    #print(MID_reward_fix.columns)
    duration3 = MID_ant['onset'].shift(-1) - MID_reward_fix['onset'] 
    print(type(duration3))
    MID_reward_fix = pd.concat([MID_reward_fix, duration3], axis = 1, ignore_index=True)
    MID_reward_fix.columns =['onset', 'trial_type', 'duration']

    
    
    ### CREATE ONE DF
    final_MID1 = pd.concat([MID_ant, MID_rew, MID_press, MID_ant_fix, MID_probe_fix, MID_reward_fix], axis = 0, ignore_index=True)
    final_MID1.columns =['onset', 'trial_type', 'duration']
    final_MID1 = final_MID1.iloc[:,[0,2,1]]
    final_MID1 = final_MID1.sort_values(by = 'onset')
    final_MID1 = final_MID1.iloc[:-1]
    print(final_MID1)
    if(mid2 == 2):
        final_MID1.to_csv(save_dir  + subject + '_ses-1_task-mid_run-02_events.tsv', sep="\t", index=False) 
    else:    
        final_MID1.to_csv(save_dir +  subject + '_ses-1_task-mid_run-01_events.tsv', sep="\t", index=False) 
  
    # Define data frame that counts the number of each condition by participant
    counts_MID1 = final_MID1['trial_type'].value_counts().reset_index().rename(columns={"index": "trial_type", "trial_type": "count"})
    counts_MID1["PID"] = subject
    counts_MID1["run"] = 1
    counts_MID1 = counts_MID1.iloc[:,[2, 3, 0, 1]]
    counts = pd.concat([counts, counts_MID1], ignore_index=True)
    #num_trials = len(cue_on1[1])
    #print(subject + " suceeded")
  
     #MID RUN 2 
    if(mid2 == 1):
        #### MID RUN 2
        #### TRIAL TYPES
        trial_type2 = df.loc[df[0] == 'RunList2'][1]+ "-" #strcat(string(txt.Var2(find(contains(txt.Var1,'RunList1')))),'-');
        trial_type2 = trial_type2.replace('1-','win_5') #replace(trial_type2,'1-','Run1 Win $5.00');
        trial_type2 = trial_type2.replace('2-','win_1.5')#replace(trial_type2,'2-','Run1 Win $1.50');
        trial_type2 = trial_type2.replace('3-','win_0')#replace(trial_type2,'3-','Run1 Win $0.00');
        trial_type2 = trial_type2.replace('4-','lose_5')#replace(trial_type2,'4-','Run1 Lose $5.00');
        trial_type2 = trial_type2.replace('5-','lose_1.5')#replace(trial_type2,'5-','Run1 Lose $1.50');
        trial_type2 = trial_type2.replace('6-','lose_0')#replace(trial_type2,'6-','Run1 Lose $0.00');
        trial_type2 = trial_type2.reset_index()
        
        #### ANTICIPATION PHASE
        cue_on2 = df.loc[df[0] == 'Run2Cue.OnsetTime'][1].astype(int).reset_index() #txt.Var2(find(contains(txt.Var1,'Run2Cue.OnsetTime')));
        cue_on2 = (cue_on2 - int(df.loc[df[0] == 'Run2Fix.OnsetTime'].iloc[0][1])) / 1000   
        #Anticipation Phase
        MID2_ant=pd.concat([cue_on2[1], "ant_" + trial_type2[1]], axis = 1)
        MID2_ant["duration"] = 2
        MID2_ant.columns =['onset', 'trial_type', 'duration']

        #### REWARD PHASE
        rt2 = df.loc[df[0] == 'Run2Tgt.RT'][1].astype(int).reset_index()
        rt2 = rt2.replace(0, np.nan)
        fbk_on2 = df.loc[df[0] == 'Run2Fbk.OnsetTime'][1].astype(int).reset_index()
        fbk_on2 = (fbk_on2 -  int(df.loc[df[0] == 'Run2Fix.OnsetTime'].iloc[0][1])) / 1000  
        acc2 = ~(rt2.isna())
        acc2 = acc2.replace({True: 'Hit', False: 'Miss'})
        #make df
        MID2_rew=pd.concat([fbk_on2[1]],axis=1)
        MID2_rew['trial_type'] = "rew_" + trial_type2[1] +  "_" + acc2[1]
        MID2_rew["duration"] = 2
        MID2_rew.columns =['onset', 'trial_type', 'duration']
        
        #### BUTTON PRESS
        target_onset2 = df.loc[df[0] == 'Run2Tgt.OnsetTime'][1].astype(int).reset_index() 
        target_onset2 = (target_onset2 -  int(df.loc[df[0] == 'Run2Fix.OnsetTime'].iloc[0][1])) / 1000
        MID2_press=pd.concat([target_onset2[1]],axis=1)
        MID2_press['trial_type'] = "motor_period"
        
        duration = (df.loc[df[0] == 'Run2Tgt.Duration'][1].astype(float)/ 1000).reset_index()
        MID2_press = pd.concat([MID2_press, duration[1]], axis = 1, ignore_index=True)
        MID2_press.columns =['onset', 'trial_type', 'duration']
        
        #### three fixations
         #anticipation fixation
        fix_on1 = df.loc[df[0] == 'Run2Dly.OnsetTime'][1].astype(int).reset_index() 
        fix_on1 = (fix_on1 - int(df.loc[df[0] == 'Run2Fix.OnsetTime'].iloc[0][1])) / 1000  #We think this is the first fixation
        #Anticipation Fixation Phase data frane
        MID_ant_fix=pd.concat([fix_on1[1], "ant_fix_" + trial_type2[1]], axis = 1)
        duration = (df.loc[df[0] == 'Run2Dly.Duration'][1].astype(float)/ 1000).reset_index()# MID_ant_fix["duration"]
        MID_ant_fix = pd.concat([MID_ant_fix, duration[1]], axis = 1, ignore_index=True)
        MID_ant_fix.columns =['onset', 'trial_type', 'duration']
        
        #probe fixation
        fix_on2 = df.loc[df[0] == 'Run2Dly2.OnsetTime'][1].astype(int).reset_index() 
        fix_on2 = (fix_on2 - int(df.loc[df[0] == 'Run2Fix.OnsetTime'].iloc[0][1])) / 1000  #We think this is the first fixation
        #Anticipation Fixation Phase data frane
        MID_probe_fix=pd.concat([fix_on2[1], "probe_fix_" + trial_type2[1]], axis = 1)
        duration2 = (df.loc[df[0] == 'Run2Dly2.Duration'][1].astype(float)/ 1000).reset_index()# MID_ant_fix["duration"]
        MID_probe_fix = pd.concat([MID_probe_fix, duration2[1]], axis = 1, ignore_index=True)
        MID_probe_fix.columns =['onset', 'trial_type', 'duration']

        
        #reward fixation
        fix_on3 = fbk_on1 + 2
        #Anticipation Fixation Phase data frane
        MID_reward_fix=pd.concat([fix_on3[1], "rew_fix_" + trial_type2[1]], axis = 1)
        MID_reward_fix.columns =['onset', 'trial_type']
        duration3 = MID_ant['onset'].shift(-1) - MID_reward_fix['onset'] 
        MID_reward_fix = pd.concat([MID_reward_fix, duration3], axis = 1, ignore_index=True)
        MID_reward_fix.columns =['onset', 'trial_type', 'duration']

        
        
        #### CREATE ON DF
        final_MID2 = pd.concat([MID2_ant,MID2_rew, MID2_press, MID_ant_fix, MID_probe_fix, MID_reward_fix], axis = 0, ignore_index=True)
        print(final_MID2)
        final_MID2.columns =['onset', 'trial_type', 'duration']
        final_MID2 = final_MID2.iloc[:,[0,2,1]]
        final_MID2 = final_MID2.sort_values(by = 'onset') 
        final_MID2 = final_MID2.iloc[:-1]
        final_MID2.to_csv(save_dir + subject + '_ses-1_task-mid_run-02_events.tsv', sep="\t", index=False) 
  
        # Define data frame that counts the number of each condition by participant
        counts_MID2 = final_MID2['trial_type'].value_counts().reset_index().rename(columns={"index": "trial_type", "trial_type": "count"})
        counts_MID2["PID"] = subject
        counts_MID2["run"] = 2
        counts_MID2 = counts_MID2.iloc[:,[2, 3, 0, 1]]
        counts = pd.concat([counts, counts_MID2], ignore_index=True)    
#MAIN

def main():
    #dirs = ["/projects/b1108/studies/transitions/data/raw/neuroimaging/behavioural/sub-t1269"] #ses-1/beh/3_MID_Scanner_HARP_11-1082-2.txt
    dirs = glob.glob('/Users/katharinaseitz/Documents/projects/gimme_first_year/data/crest_edats/*')


    ##USE THIS CHUNK TO CONVERT ONE SUB's DATA
    ## have to change line 174 depending on which split file it is.
    '''
    e_prime_file = '/projects/b1108/studies/transitions/data/raw/neuroimaging/behavioral/sub-t1134/ses-1/beh/3_MID_Scanner_HARP_11.15-1134-1.txt'
    df = file_to_df(e_prime_file)
    id = 'sub-t1134' #TODO MAKE SURE THIS IS THE RIGHT ID 
    if(id == e_prime_file.split("/")[-4]):
        df_to_timing_txt(df, id, 1) #0 for mid1, 2 for mid2
    else:
        print("ID mismatch") 
    '''
    
    for path in dirs :
        try:
            sus_out_files(path)
        except Exception as e:
            print("FAILURE: " + path)
            print(traceback.format_exc())
    #counts.to_csv("../output/MID_trial_types_counts.csv")
    
if __name__ == "__main__":
    main()
    
    

    
    