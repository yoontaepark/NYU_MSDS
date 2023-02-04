import numpy as np
import matplotlib.pyplot as plt
import os.path

plt.close("all")


### --- IMPORT THE DATA --- ###
### CHANGE PATH TO WHEREVER YOU SAVE THE DATA FILES ### 
data = np.load(".../heart_disease_data.npz") 


### --- HELPER FUNCTIONS --- ###
# Function that outputs the indices of x where val occurs
def ind_x_eq_val(x, val):
    return np.where(x==val)[0]


# Function that counts the proportion of all entries in x that are equal to val
def count_x_eq_val(x, val):
    return len(ind_x_eq_val(x, val))/float(len(x))
    

# Function that computes a Gaussian pdf with mean mu and std deviation sig at the values in x
def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) / sig / np.sqrt(2 * np.pi)


### --- QUESTION (B) --- ###
# Estimate the pmf of H
P_H0 = ### INSERT CODE HERE ###
P_H1 = ### INSERT CODE HERE ###

# Estimate the conditional pmf of S given H
P_S_H0 = np.zeros(2)
P_S_H1 = np.zeros(2)
for ind_S in range(2):
    P_S_H0[ind_S] = ### INSERT CODE HERE ### 
    P_S_H1[ind_S] = ### INSERT CODE HERE ### 

# Estimate the conditional pmf of C given H
P_C_H0 = np.zeros(4)
P_C_H1 = np.zeros(4)
for ind_C in range(4):
    P_C_H0[ind_C] = ### INSERT CODE HERE ###
    P_C_H1[ind_C] = ### INSERT CODE HERE ###

# Calculate the MAP estimate
MAP_estimate_S_C = ### INSERT CODE HERE ###

# Calculate the error rate - i.e. the proportion of all predictions that were incorrect
error_rate_S_C = ### INSERT CODE HERE ###

print "Probability of error " + str(error_rate_S_C)


### --- QUESTION (D) --- ###
## Estimate conditional pdf of X given H
mean_X_H = np.zeros(2)
std_X_H = np.zeros(2)
mean_X_H[0]= ### INSERT CODE HERE ###
std_X_H[0] = ### INSERT CODE HERE ###
mean_X_H[1]= ### INSERT CODE HERE ###
std_X_H[1] = ### INSERT CODE HERE ###

n_plot = 100
for i in range(2):
    plt.figure(figsize=(12, 9))  
    ax = plt.subplot(111)    
    ax.spines["top"].set_visible(False)    
    ax.spines["right"].set_visible(False)    
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left() 
    yticks = ax.yaxis.get_major_ticks()
    yticks[0].label1.set_visible(False) 
    plt.xticks(fontsize=22) 
    plt.yticks(fontsize=22) 
    plt.xlabel("Cholesterol", fontsize=22)  
    
    plt.hist(### INSERT CODE HERE ###,
             60,normed=True,edgecolor = "none", color="skyblue")
    
    plt.plot(np.linspace(50, 450, n_plot),gaussian(np.linspace(50, 450, n_plot), 
                     mean_X_H[i], std_X_H[i]), color="darkred", lw=2)

                 
### --- QUESTION (E) --- ###
# Calculate the MAP estimate
MAP_estimate_S_C_X = ### INSERT CODE HERE ###

# Calculate the error rate
error_rate_S_C_X = ### INSERT CODE HERE ###
print "Probability of error using cholesterol " + str(error_rate_S_C_X)
