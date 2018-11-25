# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 18:31:55 2018

@author: Claire Danaher
"""

filename = 'data1.csv'
i=0
Y=[]
X=[]
for line in open(filename,encoding="utf8"):
    vals=[]
    if i != 0:
        vals=line.strip('\n').split(',')
        Y.append(vals[0])
        X.append(vals[1:])
    i+=1




'''
            Compute the conditional entropy of y given x. The conditional entropy H(Y|X) means average entropy of children nodes, given attribute X. Refer to https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
            Input:
                X: a list of values , a numpy array of int/float/string values. The size of the array means the number of instances/examples. X contains each instance's attribute value. 
                Y: a list of values, a numpy array of int/float/string values. Y contains each instance's corresponding target label. For example X[0]'s target label is Y[0]
            Output:
                ce: the conditional entropy of y given x, a float scalar
 '''




S={}
n=len(x)
cats=[]
attrs=[]
for i in range(0,n):
    target=Y[i]
    if target not in cats:
        cats.append(target)
    if attr not in attrs:
        attrs.append(attr)
###Sum target values for node 
    if target in S:
        prev=S.get(target)
        prev+=1
        S[target]=prev
    else:
        S[target]=1     
        
print(S)
    

node_e=0

##CALCULATE ENTROPY OF ROOT NODE
for cat in cats:
    val=S.get(cat)
    calc=-1*(val/n)*math.log2(val/n)
    node_e+=calc
print(node_e)


S={}
n=len(Y)
cats=[]
cnt_max=0
for i in range(0,n):           
    target=Y[i]
    if target not in cats:
        cats.append(target)
        ###Sum target values for node 
        if target in S:
            prev=S.get(target)
            prev+=1
            S[target]=prev
            if pre>cnt_max:
                y=target
        else:
            S[target]=1 
            if i==0:
                y=target
print(y)
print(n)