# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 19:29:03 2020
Homework 7 - Q3
@author: Adi Pall
"""

def read_graph_data(fname):
    fcur = open(fname,'r')
    adj = dict()
    names = dict()
    line = fcur.readline()
    while line:
        parts = line.split(' ')
        if parts[0] == 'n': # determine if reading edge or vertex
            idx = int(parts[1])
            names[idx] = parts[2].strip() # remove what it considers to be space at end
        if parts[0] == 'e':
            idx_i = int(parts[1])
            adj.setdefault(idx_i,[]) # sets the type for this given key to be a list
            adj[idx_i].append(int(parts[2]))
        line = fcur.readline() # read next line
    fcur.close()
    print("adj: ", adj)
    print("names: ",names)
    return names, adj
    
if __name__ == '__main__':
    names, adj = read_graph_data('graph.txt')