# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:39:07 2020

@author: cyy
"""

import csv
import math as math
import numpy as np
import random as rd
import matplotlib.pyplot as plt

class DecisionTreeNode:
    
    def __init__(self, dataSet_X, dataSet_y):
        """
        Init the node for decssion tree

        Parameters
        ----------
        dataSet_X : np.array
            the data of traing fratures.
        dataSet_y : Stri
            the data if training result.
        dataType : int
            the type of child factor.
            0: the type of child factor is categorical
            1: the type of child factor is numverical
        Returns
        -------
        None.

        """
        self.train_x = dataSet_X
        self.train_y = dataSet_y
        self.factorIndex = -1
        self.entropy = self.__entropy()
        self.childs = list()
            
    def __entropy(self):
        """
        Caculate and return the entropy of current node

        Returns
        -------
        entropy : float
            entropy.

        """
        entropy = 0
        (branch, count) = np.unique(self.train_y, return_counts= True)
        for i in range(len(branch)):
            prob = count[i]/count.sum()
            entropy -=  prob * math.log(prob, 2)
    
        return entropy
    
    def probability(self):
        """
        The probability of "YES", when a instance reaches this node

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return np.count_nonzero(self.train_y)/len(self.train_y)
    
    
class RandomForest:
    def __init__(self,  fname_x  = "pa4_train_X.csv", fname_y = "pa4_train_y.csv"):
        # Forest factors
        self.dtList = list()
        self.treeNum = 0
        self.factorNum = 0
        
        # Tree factors
        self.maxDepth = 0
        self.train_x = np.array(0)
        self.train_y = np.array(0)
        self.selected_train_x = np.array(0)
        self.val_x = np.array(0)
        self.val_y = np.array(0)
        self.factorIndexTable = list()
        self.selectedfactorTable = list()
        self.__loadTrainDataFrom(fname_x, fname_y)
        
    def learnForest(self, treeNum = 10, factorSampleNum = 25, maxDepth = 10):
        """
        Issue the process of learn a random forest

        Parameters
        ----------
        treeNum : Int
            The num of tree. The default is 10.
        factorSampleNum : Int
            The num of factor will be used in the tree The default is 25.
        maxDepth : Int
            max depth of each tree. The default is 10.

        Returns
        -------
        None.

        """
        self.treeNum = treeNum
        self.factorNum = factorSampleNum
        self.maxDepth = maxDepth
        self.dtList = list()
        
        if factorSampleNum < maxDepth :
            self.maxDepth = factorSampleNum
        
        for i in range(self.treeNum):
            self.__randomSelectFactor()
            self.learningSingleTree()
            self.dtList.append(self.root)
            
      
    def learningSingleTree(self):
        """
        Initial the learning process

        Parameters
        ----------
        depth : int
            the max depth of decision tree.

        Returns
        -------
        None.

        """
        # set the entropy of root node
        self.root= DecisionTreeNode(self.selected_train_x, self.train_y)
        self.__leanring(self.root, 0, self.selectedFactorTable)
    
    def predictionTrainData(self):
        """
        get the correctness of decision tree with train data

        Returns
        -------
        float
            the corecttness(0< c <=1).

        """
        return self.__predictCorrectRate(self.train_x, self.train_y)
    
    
    def loadValidationDataFrom(self, fname_x  = "pa4_dev_X.csv", fname_y = "pa4_dev_y.csv"):
        """
        load Validation data from file

        Parameters
        ----------
        fname_x : String
            The filename that store the factor data for learning instance. The default is "pa4_dev_X.csv".
        fname_y : TYPE, optional
            The filename that store the result data for learning instance. The default is "pa4_dev_X.csv".DESCRIPTION. The default is "pa4_dev_y.csv".

        Returns
        -------
        None.

        """
        self.__loadValidationDataFrom()
        
    def predictionValidationData(self):
        """
        get the correctness of decision tree with validation data

        Returns
        -------
        float
            the corecttness(0< c <=1).

        """
        return self.__predictCorrectRate(self.val_x, self.val_y)

    
    def __loadValidationDataFrom(self, fname_x  = "pa4_dev_X.csv", fname_y = "pa4_dev_y.csv"):
        """
        

        Parameters
        ----------
        fname_x : TYPE
            DESCRIPTION.
        fname_y : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        dataX = []
        dataY = []
        with open(fname_x) as csvfile:
            trainData_reader  = csv.reader(csvfile, delimiter=" ")
            line_count = 0
            for row in trainData_reader:
                if line_count != 0:
                    line = ",".join(row).split(",")
                    dataX.append(list(map(float, line)))
                line_count += 1
            
        with open(fname_y) as csvfile:
            trainData_reader  = csv.reader(csvfile, delimiter=" ")
            line_count = 0
            for row in trainData_reader:
                line = ",".join(row).split(",")
                dataY.append(list(map(float, line)))
                line_count += 1
                
        self.val_x = np.array(dataX)
        self.val_y = np.array(dataY).reshape( self.val_x.shape[0],)
        

    def __loadTrainDataFrom(self, fname_x  = "pa4_train_X.csv", fname_y = "pa4_train_y.csv"):
        """
        Load data form csv file

        Parameters
        ----------
        fname_x : str
            the name of the file where the training factors were recorded
        fname_y : str
            the name of the file where the training result were recorded 

        Returns
        -------
        None.

        """
        # load data from file
        dataX = []
        dataY = []
        with open(fname_x) as csvfile:
            trainData_reader  = csv.reader(csvfile, delimiter=" ")
            line_count = 0
            for row in trainData_reader:
                if line_count != 0:
                    line = ",".join(row).split(",")
                    dataX.append(list(map(float, line)))
                line_count += 1
            
        with open(fname_y) as csvfile:
            trainData_reader  = csv.reader(csvfile, delimiter=" ")
            line_count = 0
            for row in trainData_reader:
                line = ",".join(row).split(",")
                dataY.append(list(map(float, line)))
                line_count += 1
                
        self.train_x = np.array(dataX)
        self.train_y = np.array(dataY).reshape( self.train_x.shape[0],)
        self.factorIndexTable = list(range(np.shape( self.train_x)[1]))
        
    def __maxMutualInformation(self, train_x, train_y):
        """
        Find the factor which has the maxmium mutual information

        Returns
        -------
        TYPE
            the indx of factor which has the maxmium mutual information.

        """
        _, factorsNum = np.shape(train_x)
        mutualInformations = list()
        for i in range(factorsNum):
            cur_column = train_x[:,i]
            factorValues, factorCounts = np.unique(cur_column, return_counts=True)
            mutualInformation = 0
            for idx, val in enumerate(factorValues):
                pos = np.where(cur_column == val)[0]
                prob = (factorCounts[idx]/np.sum(factorCounts))
                mutualInformation += prob * DecisionTreeNode(train_x[pos], train_y[pos]).entropy
            mutualInformations.append(mutualInformation)
        return mutualInformations.index(min(mutualInformations))
    
    
    def __randomSelectFactor(self):
        self.selectedFactorTable = rd.sample(self.factorIndexTable, self.factorNum)
        self.selectedFactorTable.sort()
        self.selected_train_x = self.train_x[:,self.selectedFactorTable]
           
    def __leanring(self, node, curDepth, factorIdxTable):
        """
        recursivelt learning process

        Parameters
        ----------
        node : DecisionTreeNode
            one node of decision tree.
        curDepth : int
            The depth of current node in the decision tree.
        factorIdxTable : list of int
            being used to find the factor order.

        Returns
        -------
        None.

        """
        if curDepth is self.maxDepth:
            return 
        
        fidx = self.__maxMutualInformation(node.train_x, node.train_y)
        cur_column = node.train_x[:,fidx]
        factorValues, factorCounts = np.unique(cur_column, return_counts=True)
        if len(factorValues) == 1:
            return
        node.factorIndex = factorIdxTable[fidx]
        
        curDepth += 1
        for _, val in enumerate(factorValues):
            pos = np.where(cur_column == val)[0]
            childNode = DecisionTreeNode(np.concatenate((node.train_x[pos,0:fidx], node.train_x[pos,fidx+1:]), axis=1), node.train_y[pos])
            self.__leanring(childNode, curDepth, factorIdxTable[0:fidx] + factorIdxTable[fidx+1:])
            node.childs.append(childNode)
        
                  
    def __predictCorrectRate(self, data_x, data_y):
        
        """
        helper function for computing the prediction result
        para.: 
            data_x: conditon
            data_y; result
        ret.: 
            accuracy
        """
        c = 0
        w = 0
        for idx, xi in enumerate(data_x):
            voteResult = 0
            for ti in range(self.treeNum):
                ptr = self.dtList[ti]
                while ptr.factorIndex != -1:
                    ptr = ptr.childs[int(xi[ptr.factorIndex])]
                if (ptr.probability() >= 0.5):
                    voteResult += 1
            if (voteResult >= self.treeNum /2) and data_y[idx] == 1:
                c += 1
            elif(voteResult< self.treeNum /2) and data_y[idx] == 0:
                c += 1
            else:
                w += 1       
            
        return c / (c + w)
        
            
       
class RFplot:
    def __init__(self):
        """
        This calss is just for assignment question, it is nothing about random forest

        Returns
        -------
        None.

        """
        self.rf = RandomForest()
        self.rf.loadValidationDataFrom()
        self.train5Rates = list()
        self.train25Rates = list()
        self.train50Rates = list()
        self.train100Rates = list()
        
        self.val5Rates = list()
        self.val25Rates = list()
        self.val50Rates = list()
        self.val100Rates = list()
        
             
    def drawPlot_train(self, dmax):
        for val in [10,20,30,40,50,60,70,80,90,100]:
            self.rf.learnForest(val, 5, dmax)
            trainRate = self.rf.predictionTrainData()
            valRate = self.rf.predictionValidationData()
            self.train5Rates.append(trainRate)
            self.val5Rates.append(valRate)
            
        for val in [10,20,30,40,50,60,70,80,90,100]:
            self.rf.learnForest(val, 25, dmax)
            trainRate = self.rf.predictionTrainData()
            valRate = self.rf.predictionValidationData()
            self.train25Rates.append(trainRate)
            self.val25Rates.append(valRate)
            
        for val in [10,20,30,40,50,60,70,80,90,100]:
            self.rf.learnForest(val, 50, dmax)
            trainRate = self.rf.predictionTrainData()
            valRate = self.rf.predictionValidationData()
            self.train50Rates.append(trainRate)
            self.val50Rates.append(valRate)
            
        for val in [10,20,30,40,50,60,70,80,90,100]:
            self.rf.learnForest(val, 100, dmax)
            trainRate = self.rf.predictionTrainData()
            valRate = self.rf.predictionValidationData()
            self.train100Rates.append(trainRate)
            self.val100Rates.append(valRate)

        plt.xlabel("Tree Num")
        plt.ylabel("accuracy")
        plt.scatter([10,20,30,40,50,60,70,80,90,100], self.train5Rates, c = 'red', label="m=5")
        plt.scatter([10,20,30,40,50,60,70,80,90,100], self.train25Rates, c = 'green', label="m=25")
        plt.scatter([10,20,30,40,50,60,70,80,90,100], self.train50Rates, c = 'blue', label="m=50")
        plt.scatter([10,20,30,40,50,60,70,80,90,100], self.train100Rates, c = 'yellow', label="m=100")
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.title("The Train Accuracy, depth: 25")
        plt.show()
        
        plt.xlabel("Tree Num")
        plt.ylabel("accuracy")
        plt.scatter([10,20,30,40,50,60,70,80,90,100], self.val5Rates, c = 'red', label="m=5")
        plt.scatter([10,20,30,40,50,60,70,80,90,100], self.val25Rates, c = 'green', label="m=25")
        plt.scatter([10,20,30,40,50,60,70,80,90,100], self.val50Rates, c = 'blue', label="m=50")
        plt.scatter([10,20,30,40,50,60,70,80,90,100], self.val100Rates, c = 'yellow', label="m=100")
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.title("The Validation Accuracy, depth: 25")
        plt.show()
    
    
            
    