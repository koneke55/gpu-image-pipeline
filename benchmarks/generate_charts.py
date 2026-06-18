import csv
import matplotlib.pyplot as plt

def read_csv(path):
    rows=[]
    with open(path) as f:
        r=csv.reader(f)
        for row in r: rows.append(row)
    return rows

if __name__=='__main__':
    print('Placeholder chart generator - save CSV in benchmarks/results.csv')
