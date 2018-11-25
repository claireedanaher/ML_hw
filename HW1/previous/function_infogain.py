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
            print(attr)
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
        print(S_v)
        ce=0
        for attr in attrs:
            attr_list=S_v.get(attr)
            node_cats=0
            m=attr_list.get('Total')
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
            ce+=(m/n)*node_cats
        #########################################
        print(attrs)
        print('conditional entropy')
        print(ce)
        return ce 
    
    
    
    #-------------------------
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
        g=Tree.entropy(Y)+Tree.conditional_entropy(Y,X)
        
       
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
        ## INSERT YOUR CODE HERE

    
   

 
        #########################################
        return i
        


       return g
    
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
