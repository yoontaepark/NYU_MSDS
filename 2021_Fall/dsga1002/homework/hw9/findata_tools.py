from __future__ import print_function
import pandas as pd


"""
Loads financial data as a pandas dataframe
"""
def load_dataframe(filename) :
    return pd.read_csv(filename,index_col=0)

"""
Loads financial data as a tuple: names,data.  
names is a list of the stock names represented in each column.
data is a 2d numpy array.  Each row of data corresponds to a trading day.
data[i,j] is the price (technically the adjusted closing price) of 
instrument names[j] on the ith day.  The days are ordered chronologically.
"""
def load_data(filename) :
    df = pd.read_csv(filename,index_col=0)
    names = df.columns.values.tolist()
    data = df.as_matrix()
    return names,data

"""
Given a 1d numpy array vec of n values, and a list of n names,
prints the values and their associated names.
"""
def pretty_print(vec,names) :
    print(pd.DataFrame(vec,names,['']).transpose())

"""
Given a 1d numpy array vec of n values, and a list of n names,
prints the values and their associated names in a LaTeX friendly
format.
"""
def pretty_print_latex(vec,names,num_col=6) :
    print("\\begin{center}")
    print("\\begin{tabular}{c"+("|c"*(num_col-1))+"}")
    for i in range(0,len(names),num_col) :
        start = True
        for j in range(i,min(i+num_col,len(names))) :
            if not start :
                print(" & ",end='')
            start = False
            print(names[j],end='')
        print("\\\\")
        start = True
        for j in range(i,min(i+num_col,len(names))) :
            if not start :
                print(" & ",end='')
            start = False
            print("%.04f"%vec[j],end='')
        print("\\\\")
        if i+num_col < len(names) :
            print("\\hline")
    print("\\end{tabular}")
    print("\\end{center}")

def main() :
    names,data = load_data('stockprices.csv')
    print("# of stocks = %d, # of days = %d"%(data.shape[1],data.shape[0]))
    pretty_print(data[0,:],names)
    #pretty_print_latex(data[0,:],names)

if __name__ == "__main__": 
    main()
