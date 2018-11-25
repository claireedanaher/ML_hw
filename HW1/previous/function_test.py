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
S_v={}
n=len(x)
cats=[]
attrs=[]
for i in range(0,n):
    target=Y[i]
    attr=x[i]
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

node_e=0
node_sub_all=0   
node_sub=0
for cat in cats:
    val=S.get(cat)
    print(cat)
    print(val)
    calc=-1*(val/n)*math.log2(val/n)
    node_e+=calc
    
node_sub=0   
print()
for attr in attrs:
    attr_list=S_v.get(attr)
    node_cats=0
    m=attr_list.get('total')
    print('start attribute')
    print(attr)
    for cat in cats:
        print('target')
        print(cat)
        val=attr_list.get(cat)
        if val != None:
            if val/n == 1:
                node_cats-=1
            else:
                calc=-(val/m)*math.log2(val/m)
                node_cats+=calc
                print(node_cats)
            node_sub+=node_cats
    node_sub_all=(m/n)*node_sub
            
    print(node_sub)
print(math.log2(0))
print(val/n)


print(m)


node_total=-(19/42)*math.log2(19/42)-(23/42)*math.log2(23/42)
eight=1
four=-(4/25)*math.log2(4/25)-(21/25)*math.log2(21/25)
print(four)
six=-(2/10)*math.log2(2/10)-(8/10)*math.log2(8/10)
print(six)

calc=42/
print(four)


print(node_e)   
print(S_v)     
print(n)    
print(S)   
print(cats)    
print(attrs)     




S={}
S_v={}
n=len(x)
cats=[]
attrs=[]
for i in range(0,n):
    target=Y[i]
    attr=x[i]
    if target not in cats:
        cats.append(target)
    if attr not in attrs:
        attrs.append(attr)
    if target in S:
        prev=S.get(target)
        prev+=1
        S[target]=prev
    else:
        S[target]=1     
        
        
    if attr in S_v:
        sublist=S_v[attr]
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
        S_v[attr]=sublist

    
    
print(S_v)     
print(n)    
print(S)   
print(cats)    
print(attrs)   



    
        
    if x[i] in S_v:
        
    else:
        att_val=x[i]
        S_v[att_val]={}
        val={}
        sub_val[Y[i]]=1
        
        
        val=x[i]
        S_v[val]={}
    
        
            
        prev=S.get(Y[i])
        prev+=1
        S[Y[i]]=prev
    else:
        S[Y[i]]=1     
        
        
        

        term_dict={}
        index=[]
        master_stemmed_tokens=[]
        num_files_indexed = 0
        
        for filename in glob.glob(base_path+'*.txt'):
            self._documents.append(filename[5:])
            
            with open(filename,encoding='utf8') as f:    
                
                text=f.read()
                tokens=Index.tokenize(self,text) 
                stemmed_tokens=Index.stemming(self,tokens)
                
                for token in stemmed_tokens:
                    if token in master_stemmed_tokens:
                        prev_docs=self._inverted_index.get(token,[])
                        prev_docs.append(num_files_indexed)
                        self._inverted_index[token]=prev_docs
                    else:
                        master_stemmed_tokens.append(token)
                        term_dict[token]=index
                        prev_docs=self._inverted_index.get(token,[])
                        prev_docs.append(num_files_indexed)
                        self._inverted_index[token]=prev_docs
                num_files_indexed+=1        
        
        return num_files_indexed