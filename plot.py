import matplotlib.pyplot as plt
import pandas as pd


import sys, getopt

def plot(argv):
    filename = 'average_W_too_long.csv'
    input_size = len(argv)
    
    if input_size==0:
        print('No file is given, plot '+filename)
    else:
        filename = argv[0]
        if input_size>1:
            print('arguments are neglected:')
            for items in argv[1:]:
                print(items)
    
    data = pd.read_csv(filename)
    fig, ax = plt.subplots()
    ax.semilogx(data['taus'], data['mean_W_forward'],'o', data['taus'],data['mean_W_backward'], 'o')
    ax.grid()
    
    outfile = filename.split('.')
    outfile = outfile[0] + '.png'
    plt.savefig(outfile)
    plt.show()

if __name__ == "__main__":
    plot(sys.argv[1:])






