# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 20:31:12 2018

@author: Claire Danaher
"""

import math 
filename = 'C:\\WPI\\MachineLearning\\HW\\HW1\\data1.csv'
i=0
Y=[]
X=[]
print(X)
for line in open(filename,encoding="utf8"):
    vals=[]
    if i != 0:
        vals=line.strip('\n').split(',')
        Y.append(vals[0])
        X.append(vals[1:])
    i+=1



len(X[0])
class Node:
    '''
        Decision Tree Node (with discrete attributes)
        Inputs: 
            X: the data instances in the node, a numpy matrix of shape p by n.
               Each element can be int/float/string.
               Here n is the number data instances in the node, p is the number of attributes.
            Y: the class labels, a numpy array of length n.
               Each element can be int/float/string.
            i: the index of the attribute being tested in the node, an integer scalar 
            C: the dictionary of attribute values and children nodes. 
               Each (key, value) pair represents an attribute value and its corresponding child node.
            isleaf: whether or not this node is a leaf node, a boolean scalar
            p: the label to be predicted on the node (i.e., most common label in the node).
    '''
    def __init__(self,X,Y, i=None,C=None, isleaf= False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.C= C
        self.isleaf = isleaf
        self.p = p
attr_list={}
att_cnt=len(X[0])
for row in X:
    for i in range(0, att_cnt):
        target=row[i]
        if i in attr_list:
                sublist=attr_list[i]
                if target not in sublist:
                    sublist.append(target)
                    attr_list[i]=sublist
        else:
            sublist=[]
            sublist.append(target)
            attr_list[i]=sublist


print(attr_list)
        
        '''
        if target not in sublist:
            sublist[target]=row[i]
        attr[i]=sublist
        '''