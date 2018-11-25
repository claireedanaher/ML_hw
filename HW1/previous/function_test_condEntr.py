# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 18:31:55 2018

@author: Claire Danaher
"""
import math 
filename = 'C:\\WPI\\MachineLearning\\HW\\HW1\\data1.csv'
i=0
Y=[]
X=[]
x=[]
print(X)
for line in open(filename,encoding="utf8"):
    vals=[]
    if i != 0:
        vals=line.strip('\n').split(',')
        Y.append(vals[0])
        X.append(vals[1:])
    i+=1

for line in X:
    x.append(line[0])
    
    
    


'''
            Compute the conditional entropy of y given x. The conditional entropy H(Y|X) means average entropy of children nodes, given attribute X. Refer to https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
            Input:
                X: a list of values , a numpy array of int/float/string values. The size of the array means the number of instances/examples. X contains each instance's attribute value. 
                Y: a list of values, a numpy array of int/float/string values. Y contains each instance's corresponding target label. For example X[0]'s target label is Y[0]
            Output:
                ce: the conditional entropy of y given x, a float scalar
 '''




S_v={}
n=len(x)
cats=[]
attrs=[]
for i in range(0,n):
    target=Y[i]
    attr=x[i]
    if attr not in attrs:
        attrs.append(attr)
    if target not in cats:
        cats.append(target)
    
####Sum target values by attribute
    if attr in S_v:
        sublist=S_v[attr]
        total=sublist.get('Total')
        total+=1
        sublist['Total']=total
        if target in sublist:
            count=sublist.get(target)
            count+=1
            sublist[target]=count
            S_v[attr]=sublist
        else:
            count={}
            sublist[target]=1
            S_v[attr]=sublist
    else:
        sublist={}
        sublist[target]=1
        sublist['Total']=1
        S_v[attr]=sublist

##CALCULATE ENTROPY OF SUBGROUPS

node_sub=0
for attr in attrs:
    attr_list=S_v.get(attr)
    node_cats=0
    m=attr_list.get('Total')
    print('\n')
    print('start attribute')
    print(attr)
    for cat in cats:
        val=attr_list.get(cat)
        if val != None:
            print(cat)
            print(val)
            if val/m == 1:
                node_cats=0
            else:
                calc=-(val/m)*math.log2(val/m)
                node_cats+=calc
    node_sub+=(m/n)*node_cats
print(S_v)
'''           
print(node_sub)

print(val/n)
print(cats)
print(attrs)
print(m)


print(S_v)
print(attrs)
eight=0
four=-(4/25)*math.log2(4/25)-(21/25)*math.log2(21/25)
print(four)
six=-(2/10)*math.log2(2/10)-(8/10)*math.log2(8/10)
print(six)

print((7/42)*eight+(25/42)*four+(10/42)*six)

test=-(5/10)*math.log2(5/10)-(5/10)*math.log2(5/10)
print(test)
'''