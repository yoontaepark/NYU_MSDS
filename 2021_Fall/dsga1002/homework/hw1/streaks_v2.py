import numpy as np
import matplotlib.pyplot as plt
plt.close("all")
np.random.seed(2017)

def p_longest_streak(n, tries):
    # Write your Monte Carlo code here, n is the length of the sequence and tries is the number
    # of sampled sequences used to produce the estimate of the probability
    answer =[]

    for i in range(1,n+1):
        consecSuccess = 0   ## number of trials where 4 heads were flipped consecutively
        coin = 0  ## to be assigned to a number between 0 and 1
        numTrials = 10000
        for _ in range(numTrials):
            consecCount = 0
            for _ in range(tries):
                coin = np.random.random()
                #since random.random() generates [0.0, 1.0), we'll make the lower bound inclusive
                if coin <= .5:
                    #Treat the lower half as heads, so increment the count
                    consecCount += 1 
                # No need for an elif here, it wasn't a head, so it must have been a tail
                else:
                    # Tails, so we don't care about anything, just reset the count
                    consecCount = 0 
                #After each flip, check if we've had 4 consecutive, otherwise keep flipping
                if consecCount == i:
                    consecSuccess += 1 
                    break
        
        answer.append(consecCount)

    print(answer)

n_tries = [1e3,5e3,1e4,5e4,1e5]

n_vals = [5,200]

color_array = ['orange','darkorange','tomato','red', 'darkred', 'tomato', 'purple', 'grey', 'deepskyblue', 
            'maroon','darkgray','darkorange', 'steelblue', 'forestgreen', 'silver']


for ind_n in range(len(n_vals)):
    n = n_vals[ind_n]

    plt.figure(figsize=(20,5))

    for ind_tries in range(len(n_tries)):
        tries = n_tries[ind_tries]
        print ("tries: " + str(tries))
        p_longest_tries = p_longest_streak(n, int(tries))
        plt.plot(range(n+1),p_longest_tries, marker='o',markersize=6,linestyle="dashed",lw=2,
                color=color_array[ind_tries],
                markeredgecolor= color_array[ind_tries],label=str(tries))
    plt.legend()
    
print ("The probability that the longest streak of ones in a Bernoulli iid sequence of length 200 has length 8 or more is ")
print (p_longest_streak(5,1)) # Compute the probability and print it here