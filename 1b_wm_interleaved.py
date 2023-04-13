# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 10:56:41 2023

@author: ab61diru

scripts to run 1b wm interleaved task and preliminary treshold assessment. 
One preliminary(typet,name_ID, nr) should be ran for each type (room_size, voice & angle). 

then the task proper interleaved_1back(typet,name_ID, nr) should be ran 4 times (100 trials block) 
2 for each type (room_size & angle)

for each interleaved block 2 binary sequences (50/50 0/1 101100101000101101110010 etc) are determined. 
Sequence generation aims at a balance between having as many consecutive repetitions as possible and having them to be 
as unpredictable as possible. 

The actual number of consecutive repetitions for each stimulus (1/0) is always more than 20 and within 1 for each stimulus

preliminary is, at the moment, staircase 4 up 1 down. (85%). 

packages needed: slab
folders with stimuli needed in working folder: 'USO_angles',"resynthesized","USO_echoes"

input:
    
    typet (str): the type of block. "voice" block (only in preliminary()), "room_size" block, "angle" block. 
    Name_ID (str): of subject
    Nr: Session
    
output preliminary():
    typet+name_ID+nr+"_preliminary_stair.pkl" (pickle file) stair object from slab toolbox. contains performance data
    from preliminary tests
    typet+name_ID+nr+"_preliminary_latencies.pkl" (pickle file) list of latencies for each stair datapoint
    
output interleaved_1back():
    typet+name_ID+nr+"_wm_"+typet1+"_"+typet2+"_seq1_seq2_res1_res2_tim1_tim2.pkl" (pickle file) list containing 
    sequence of first stimulus type, sequence of 2nd stimulus type, responses for each, latencies for each

"""



import numpy as np
import pickle
import time
import keyboard
import os
import slab
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
def determine_prel_order():
    return random.sample(["room_size","voice","angle"],3)
def determine_test_start():
    return random.sample(["room_size","angle"],1)
def calculate_dprime(hr, far):
    z_hr = norm.ppf(hr)
    z_far = norm.ppf(far)
    d_prime = z_hr - z_far
    return d_prime

def calculate_hit_and_far(stimuli, responses):
    
    hits = sum([s == r for s, r in zip(stimuli, responses)])
    false_alarms = sum([s != r for s, r in zip(stimuli, responses)])

    hit_rate = hits / len(stimuli)
    false_alarm_rate = false_alarms / len(stimuli)

    return hit_rate, false_alarm_rate    

def present_stimuli_stair(stimuli_list,typet,ramp):
    indices = list(range(len(stimuli_list)))

    random.shuffle(indices)
    shuffled_list = [stimuli_list[i] for i in indices]
    
    target_position = indices.index(0)
    
    aaa=random.sample(list(np.arange(0,len(shuffled_list[0]))),3)
    wave_obj1 = shuffled_list[0][aaa[0]].ramp("offset",ramp)
    wave_obj2 = shuffled_list[1][aaa[1]].ramp("offset",ramp)
    wave_obj3 = shuffled_list[2][aaa[2]].ramp("offset",ramp)
    
    wave_obj1.play()
    time.sleep(0.25)
    wave_obj2.play()
    time.sleep(0.25)
    wave_obj3.play()
    ticc=time.time()
    
# prompt user for input except for first audio file

    print("which stimulus was from a different" +typet+" than the others? press 1, 2 or 3")
    response = 0
    while True:
        # Check for key press
        if keyboard.is_pressed('1'):
            if target_position != 0:
                response = 0
            else:
                response=1
            break
        elif keyboard.is_pressed('2'):
            if target_position != 1:
                response = 0
            else:
                response=1
            break
        elif keyboard.is_pressed('3'):
            if target_position != 2:
                response = 0
            else:
                response=1
            break
        # Wait for 0.1 seconds before checking again
        time.sleep(0.1)
    tocc=time.time()-ticc
    return response,tocc

def present_stimuli_wm(stimulus,typet,ramp,samp):
    aaa=list(np.arange(0,len(stimulus)))
    aaa.remove(samp)
    bbb=random.sample(aaa, 1)[0]
    stimulus[bbb].ramp("offset",ramp).play()
    
    ticc=time.time()
    
# prompt user for input except for first audio file

    print("is the stimulus from the same" +typet+" than the last one of the same type that you heard? press y or n")
    response = 0
    while True:
        # Check for key press
        if keyboard.is_pressed('y'):
            
            response = 1
            break
        elif keyboard.is_pressed('n'):
            response = 0
            break
        # Wait for 0.1 seconds before checking again
        time.sleep(0.1)
    tocc=time.time()-ticc
    return response,tocc,bbb


def generate_sequence(length):
    sequence = ""
    prev_digit = None
    consecutive_repetitions = {0: 0, 1: 0}

    for _ in range(length):
        if prev_digit is None:
            digit = random.choice([0, 1])
        else:
            repetition = random.choice([True, False])

            if repetition:
                if consecutive_repetitions[0] == consecutive_repetitions[1]:
                    digit = prev_digit
                else:
                    least_repeated_digit = min(consecutive_repetitions, key=consecutive_repetitions.get)
                    if prev_digit == least_repeated_digit:
                        digit = prev_digit
                    else:
                        digit = 1 - prev_digit
                consecutive_repetitions[digit] += 1
            else:
                digit = 1 - prev_digit

        sequence += str(digit)
        prev_digit = digit

    return sequence



def count_total_consecutive_repetitions(sequence):
    counts = {}
    current_item = sequence[0]
    current_count = 0

    for item in sequence[1:]:
        if item == current_item:
            current_count += 1
            counts[current_item] = counts.get(current_item, 0) + 1
        else:
            current_item = item
            current_count = 0

    return counts


def preliminary(typet="angle",name_ID="Ale",nr="1",ramp=.2):
    if typet=="angle":
        root_folder = 'USO_angles'
    elif typet=="voice":
        root_folder = "resynthesized"
    elif typet=="room_size":
        root_folder = "USO_echoes"
    zip_files = []
    
    for folder, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.zip'):
                zip_files.append(os.path.join(folder, file))
    times=[]
    stims=[]
    for f in zip_files:
        print(f)
        stims.append(slab.Precomputed.read(f))
    print(stims[0][0].duration)
    input("after you press enter you will hear four sounds, all being played from the same "+typet+". After those four sounds, there will be a three seconds pause, then the task will start. During the task you will be presented wit three stimuli. One will be from a different "+typet+" than the other two. You will tell us which sound came from a different "+typet+" by pressing 1, 2 or 3. Now press Enter to continue. "  )
    stims[0].play()
    time.sleep(0.25)
    stims[0].play()
    time.sleep(0.25)
    stims[0].play()
    time.sleep(0.25)
    stims[0].play()
    time.sleep(3)
    
    stairs = slab.Staircase(start_val=6, step_sizes=1, n_down=4,max_val=10)
    #target=stims[-1]
    for n in stairs:
       #distractors=[stims[n],stims [n]]
       #stairs.present_afc_trial(target=target, distractors=distractors)
       stimuli=[stims[n],stims[0],stims[0]]
       response,latency=present_stimuli_stair(stimuli,typet,ramp)
       times.append(latency)
       if n==0:
           response=0
       stairs.add_response(response)
       time.sleep(0.5)
    stairs.plot()
    stairs.threshold()
    stairs.save_pickle(typet+name_ID+nr+"_"+str(ramp)+"_preliminary_stair.pkl")   
    with open(typet+name_ID+nr+"_"+str(ramp)+"_preliminary_latencies.pkl", 'wb') as f:
        pickle.dump(times, f)
    
    # if typet=="angle":
    #     angle_use= stims[max(np.array(stairs.intensities)[np.array(stairs.data)==False])+1]
    #     stims_angle=[stims[0],angle_use]
    # elif typet=="voice":
    #     voice_use= stims[max(np.array(stairs.intensities)[np.array(stairs.data)==False])+1]
    #     stims_voice=[stims[0],voice_use]
    # elif typet=="room_size":
    #     echo_use= stims[max(np.array(stairs.intensities)[np.array(stairs.data)==False])+1]
    #     stims_echo = [stims[0],echo_use]
# typet="room_size"
# name_ID="Ale"
# nr="1"

def interleaved_1back(typet="room_size",name_ID="Ale",nr="1",ramp=.2):
    
    with open(typet+name_ID+nr+"_"+str(ramp)+"_preliminary_stair.pkl", 'rb') as f:
        stair=pickle.load( f)
    if typet=="angle":
        root_folder = 'USO_angles'
    
    elif typet=="room_size":
        root_folder = "USO_echoes"
    zip_files = []
    
    for folder, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.zip'):
                zip_files.append(os.path.join(folder, file))
    stims=[]
    for f in zip_files:
        print(f)
        stims.append(slab.Precomputed.read(f))    
        
        
        
    if typet=="angle":
        angle_use= stims[int(np.ceil((np.mean((stair["reversal_intensities"][-(stair["n_reversals"]-1):])))))]
        stims_angle=[stims[0],angle_use]
    
    elif typet=="room_size":
        echo_use= stims[int(np.ceil((np.mean((stair["reversal_intensities"][-(stair["n_reversals"]-1):])))))]
        stims_echo = [stims[0],echo_use]    
    print(zip_files[int(np.ceil((np.mean((stair["reversal_intensities"][-(stair["n_reversals"]-1):])))))]+"_vs_"+zip_files[0])

        
        
        
    with open("voice"+name_ID+str(nr)+"_"+str(ramp)+"_preliminary_stair.pkl", 'rb') as f:
        stair=pickle.load( f)
    root_folder = "resynthesized"   
    zip_files = []
    
    for folder, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.zip'):
                zip_files.append(os.path.join(folder, file))
    stims=[]
    for f in zip_files:
        print(f)
        stims.append(slab.Precomputed.read(f))       
    voice_use= stims[int(np.ceil((np.mean((stair["reversal_intensities"][-(stair["n_reversals"]-1):])))))]
    stims_voice=[stims[0],voice_use]    
    print(zip_files[int(np.ceil((np.mean((stair["reversal_intensities"][-(stair["n_reversals"]-1):])))))]+"_vs_"+zip_files[0])

    
    
    
    
    
    
    if typet=="angle":
        if random.random() > 0.5:
            typet1=typet
            starto="not a voice"
            secondo="voice"
            typet2=secondo
            stims_1= stims_angle
            stims_2=stims_voice
        else:
            starto="voice"
            typet1=starto
            secondo="not a voice"
            typet2=typet
            stims_2= stims_angle
            stims_1=stims_voice
    elif typet=="room_size":
        if random.random() > 0.5:
            typet1=typet
            starto="not a voice"
            secondo="voice"
            typet2=secondo
            stims_1= stims_echo
            stims_2=stims_voice
        else:
            starto="voice"   
            typet1=starto
    
            secondo="not a voice"
            typet2=typet
    
            stims_2= stims_echo
            stims_1=stims_voice
    sequence_length = 100
    generated_sequence_1 = generate_sequence(sequence_length)
    generated_sequence_2 = generate_sequence(sequence_length)
    
    
    total_counts1 = count_total_consecutive_repetitions(generated_sequence_1)
    
    while abs(total_counts1["1"]-total_counts1["0"]) >1 or total_counts1["1"]+total_counts1["0"] <40:
        generated_sequence_1=generate_sequence(100)
        total_counts1=count_total_consecutive_repetitions(generated_sequence_1)
        
    total_counts2 = count_total_consecutive_repetitions(generated_sequence_2)
    
    while abs(total_counts2["1"]-total_counts2["0"]) >1 or total_counts2["1"]+total_counts2["0"] <40:
        generated_sequence_2=generate_sequence(100)
        total_counts2=count_total_consecutive_repetitions(generated_sequence_2)
        
    generated_sequence_2 = [int(digit) for digit in generated_sequence_2] 
    generated_sequence_1 = [int(digit) for digit in generated_sequence_1] 
    input("you are now going to hear a series of stimuli. Stimuli will be alternated between two types. One will be "+starto+""",
          the next one will be """+secondo+", and then again "+starto+ """ and so on and so on. After every voice stimulus 
          you have to press Y if it is from the same voice you heard previously, N if it is not. 
          After every non-voice stimulus you have to press Y if it is from the same """+typet+""" than the previous non-voice stimulus, press N if it is not.
          the task is difficult: do not worry about making mistakes, that is ok. Try to respond as fast as possible and without 
          thinking about it too much. Press enter when you are ready to start""")
    responses1=[]
    times1=[] 
    responses2=[]
    times2=[] 
    aaa=list(np.arange(0,len(stims_1[0])))
    samp1=random.sample(aaa, 1)[0]
    aaa=list(np.arange(0,len(stims_2[0])))
    samp2=random.sample(aaa, 1)[0]
  
    for trial_nr,trial_val in enumerate(np.array(generated_sequence_1)):    
        if trial_nr>0:
           response,timeo,samp1=present_stimuli_wm(stims_1[trial_val],typet1,ramp,samp1)
           time.sleep(0.25)
           responses1.append(response)
           times1.append(timeo)
           response,timeo,samp2=present_stimuli_wm(stims_2[np.array(generated_sequence_1)[trial_nr]],typet2,ramp,samp2)
           responses2.append(response)
           times2.append(timeo)
           time.sleep(0.25)
        else:
            
            stims_1[trial_val][samp1].play()
            time.sleep(0.45)
            stims_2[trial_val][samp2].play()
            time.sleep(0.45)
            
    with open(typet+name_ID+nr+"_wm_"+typet1+"_"+typet2+"_"+str(ramp)+"_"+str(int(time.time()))+"_seq1_seq2_res1_res2_tim1_tim2.pkl", 'wb') as f:
        pickle.dump([generated_sequence_1,generated_sequence_2,responses1,responses2,times1,times2], f)
        
# if __name__ == '__main__':
    # determine_prel_order()
#     preliminary()
#   determine_test_start()
#     interleaved_1back()
