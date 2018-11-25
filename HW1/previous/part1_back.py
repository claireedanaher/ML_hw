import math
import numpy as np
from collections import Counter
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Part 1: Decision Tree (with Discrete Attributes) -- 40 points --
    In this problem, you will implement the decision tree method for classification problems.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''
        
#-----------------------------------------------
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
    def __init__(self,X,Y, i=None,C=None, isleaf=False,p=None):
        self.X = X
        self.Y = Y
        self.i = i
        self.C= C
        self.isleaf = isleaf
        self.p = p

#-----------------------------------------------
class Tree(object):
    '''
        Decision Tree (with discrete attributes). 
        We are using ID3(Iterative Dichotomiser 3) algorithm. So this decision tree is also called ID3.
    '''
    #--------------------------
    @staticmethod
    def entropy(Y):
        '''
            Compute the entropy of a list of values.
            Input:
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                e: the entropy of the list of values, a float scalar
            Hint: you could use collections.Counter.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        #[print(i) for i in c2]
        S={}
        n=len(Y)
        cats=[]
        for i in range(0,n):
            target=Y[i]
            if target not in cats:
                cats.append(target)
                ###Sum target values for node 
            if target in S:
                prev=S.get(target)
                prev+=1
                S[target]=prev
            else:
                S[target]=1     
        
        e=0
        
        ##CALCULATE ENTROPY OF ROOT NODE
        for cat in cats:
            val=S.get(cat)
            calc=-1*(val/n)*math.log2(val/n)
            e+=calc
    
        #########################################
        return e

    
            
    #--------------------------
    @staticmethod
    def conditional_entropy(Y,X):
        '''
            Compute the conditional entropy of y given x. The conditional entropy H(Y|X) means average entropy of children nodes, given attribute X. Refer to https://en.wikipedia.org/wiki/Information_gain_in_decision_trees
            Input:
                X: a list of values , a numpy array of int/float/string values. The size of the array means the number of instances/examples. X contains each instance's attribute value. 
                Y: a list of values, a numpy array of int/float/string values. Y contains each instance's corresponding target label. For example X[0]'s target label is Y[0]
            Output:
                ce: the conditional entropy of y given x, a float scalar
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        S_v = dict()
        n=len(Y)
        cats=[]
        attrs=[]
        for i in range(0,n):
            target=Y[i]
        ##MAJOR NOTE!!!!!
        #########Requires updating after solving the problem of X
            attr=X[i]
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
        ce=0
        for attr in attrs:
            attr_list=S_v.get(attr)
            node_cats=0
            m=attr_list.get('Total')
            for cat in cats:
                val=attr_list.get(cat)
                if val != None:
                    if val/m == 1:
                        node_cats=0
                    else:
                        calc=-(val/m)*math.log2(val/m)
                        node_cats+=calc
            ce+=(m/n)*node_cats
        #########################################
        return ce 
    
    
    
    #--------------------------
    @staticmethod
    def information_gain(Y,X):
        '''
            Compute the information gain of y after spliting over attribute x
            InfoGain(Y,X) = H(Y) - H(Y|X) 
            Input:
                X: a list of values, a numpy array of int/float/string values.
                Y: a list of values, a numpy array of int/float/string values.
            Output:
                g: the information gain of y after spliting over x, a float scalar
        '''
        #########################################
        ## 
        g=Tree.entropy(Y)-Tree.conditional_entropy(Y,X)
        
       
        return g


    #--------------------------
    @staticmethod
    def best_attribute(X,Y):
        '''
            Find the best attribute to split the node. 
            Here we use information gain to evaluate the attributes. 
            If there is a tie in the best attributes, select the one with the smallest index.
            Input:
                X: the feature matrix, a numpy matrix of shape p by n. 
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n. Each element can be int/float/string.
            Output:
                i: the index of the attribute to split, an integer scalar
        '''
        #########################################
        ## 
        max_infogain=0
        i=0

        for j in range(0,len(X)):
            att_infogain=Tree.information_gain(Y,X[j])
            if max_infogain<att_infogain:
                i=j
                max_infogain=att_infogain

                
        #########################################
        return i

        
    #--------------------------
    @staticmethod
    def split(X,Y,i):
        '''
            Split the node based upon the i-th attribute.
            (1) split the matrix X based upon the values in i-th attribute
            (2) split the labels Y based upon the values in i-th attribute
            (3) build children nodes by assigning a submatrix of X and Y to each node
            (4) build the dictionary to combine each  value in the i-th attribute with a child node.
    
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                i: the index of the attribute to split, an integer scalar
            Output:
                C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        C={}
        x_unique=np.unique(X[i])
        print(x_unique)
        n=len(X)
        for attr_cat in x_unique:
            x_attr=[]
            y_attr=[]
            rows=np.where(X[i]==attr_cat)
            y_attr=Y[rows]
            array=np.array([])
            for j in range(0,n):
                vals=X[j]
                check=vals[rows]                    
                x_attr.append(check)
            x_attr=np.array(x_attr)
            C[attr_cat]=Node(x_attr,y_attr)

        #########################################
        return C

    #--------------------------
    @staticmethod
    def stop1(Y):
        '''
            Test condition 1 (stop splitting): whether or not all the instances have the same label. 
    
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                s: whether or not Conidtion 1 holds, a boolean scalar. 
                True if all labels are the same. Otherwise, false.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        n=len(np.unique(Y))
        if n>1:
            s=False
        else:
            s=True
        #########################################
        return s
    
    #--------------------------
    @staticmethod
    def stop2(X):
        '''
            Test condition 2 (stop splitting): whether or not all the instances have the same attribute values. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node, p is the number of attributes.
            Output:
                s: whether or not Conidtion 2 holds, a boolean scalar. 
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        s=False
        cnt=0
        for attr in X:
            print(attr)
            attr_unique=len(np.unique(attr))
            if attr_unique==1:
                cnt+=1
        if len(X)==cnt:
            s=True
        print(s)
        #########################################
        print('final')
        print(s)
        return s

     
    #--------------------------
    @staticmethod
    def most_common(Y):
        '''
            Get the most-common label from the list Y. 
            Input:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the node.
            Output:
                y: the most common label, a scalar, can be int/float/string.
        '''
        #########################################
        max_cnt=0
        targets=np.unique(Y)
        for val in targets:
            cnt=sum(1 for x in Y if x == val)
            if cnt>max_cnt:
                y=val
                max_cnt=cnt
        
        #########################################
        return y
    
    
    
    #--------------------------
    @staticmethod
    def build_tree(t):
        '''
            Recursively build tree nodes.
            Input:
                t: a node of the decision tree, without the subtree built.
                t.X: the feature matrix, a numpy float matrix of shape n by p.
                   Each element can be int/float/string.
                    Here n is the number data instances, p is the number of attributes.
                t.Y: the class labels of the instances in the node, a numpy array of length n.
                t.C: the dictionary of attribute values and children nodes. 
                   Each (key, value) pair represents an attribute value and its corresponding child node.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        t.p=Tree.most_common(t.Y)
        if Tree.stop1(t.Y)==True or Tree.stop2(t.X)==True:
            t.isleaf=True
            return
        else:
            t.i=Tree.best_attribute(t.X,t.Y)
            t.C=Tree.split(t.X, t.Y, t.i)
            for child in t.C.values():
                Tree.build_tree(child)
        return t
        
        
   


 
        #########################################
    
    
    #--------------------------
    @staticmethod
    def train(X, Y):
        '''
            Given a training set, train a decision tree. 
            Input:
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the training set, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
            Output:
                t: the root of the tree.
        '''
        #########################################
        ## 
        t=Node(X,Y)
        t=Tree.build_tree(t)


 
        #########################################
        return t
    
    
    
    #--------------------------
    @staticmethod
    def inference(t,x):
        '''
            Given a decision tree and one data instance, infer the label of the instance recursively. 
            Input:
                t: the root of the tree.
                x: the attribute vector, a numpy vectr of shape p.
                   Each attribute value can be int/float/string.
            Output:
                y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

   




 
        #########################################
        return y
    
    #--------------------------
    @staticmethod
    def predict(t,X):
        '''
            Given a decision tree and a dataset, predict the labels on the dataset. 
            Input:
                t: the root of the tree.
                X: the feature matrix, a numpy matrix of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
            Output:
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## INSERT YOUR CODE HERE





        #########################################
        return Y



    #--------------------------
    @staticmethod
    def load_dataset(filename = 'data1.csv'):
        '''
            Load dataset 1 from the CSV file: 'data1.csv'. 
            The first row of the file is the header (including the names of the attributes)
            In the remaining rows, each row represents one data instance.
            The first column of the file is the label to be predicted.
            In remaining columns, each column represents an attribute.
            Input:
                filename: the filename of the dataset, a string.
            Output:
                X: the feature matrix, a numpy matri===x of shape p by n.
                   Each element can be int/float/string.
                   Here n is the number data instances in the dataset, p is the number of attributes.
                Y: the class labels, a numpy array of length n.
                   Each element can be int/float/string.
        '''
        #########################################
        ## 
        '''
        i=0
        Y=[]
        X=[]
        for line in open(filename,encoding="utf8"):
            vals=[]
            if i != 0:
                vals=line.strip('\n').split(',')
                Y.append(vals[0])
                for i in range(1,len(vals[1:])):
                    
                
                X.append(vals[1:])
            i+=1
        '''
        #########################################
        #return X,Y



