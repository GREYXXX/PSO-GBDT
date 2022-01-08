from data import DataSet
import random


class Node(object):

    def __init__(self, merge, remain_indexs, id, loss, depth, parent):

        """
        param merge: tuple of (feature, feature value)
        param remain_indexs: indexs of dataset remains at the current node
        param id: id number of the current node
        param loss: the loss type (SquaresError, BinomialDeviance, MultinomialDeviance)
        param depth: depth of the current node
        param parent: parent node
        """

        self.id = id
        self.merge = merge
        self.remain_indexs = remain_indexs
        self.loss = loss
        self.left = None
        self.right = None
        self.predict_value = None
        self.parent = parent
        self.depth = depth

    def is_leaf(self):
        if self.left == None and self.right == None:
            return True
        else:
            return False

    def update_predict_value(self, data, targer_name, label_name):
        """
        Update the predict value for the leaf nodes
        """
        self.predict_value = self.loss.update_leaf_values(data[targer_name], data[label_name])


class Tree(object):

    def __init__(self, dataset, target_name, loss, tree_input, max_depth, tree_id):

        """
        param dataset : dataset -->(class DataSet)
        param target_name : column name used to identify the round of the residual (e.g. f_0, res_1, f_2, res_2 ...)
        param max_depth: max depth of the tree
        param tree_input: an array of a tree, [] represents the init stage, else build the tree with this array
        param tree_id: id of the current tree
        """

        self.target_name = target_name
        self.loss = loss
        self.max_depth = max_depth
        self.tree_input = tree_input
        self.tree_id = tree_id
        self.id =0
        self.leaf_nodes = []

        #tree_array is used to record the internal nodes
        self.tree_array = []
        self.data = dataset
        self.data._encodeTable()

        #Get lookup tables
        self.lookup_tables = self.data.getLookupTable()

        #if tree_input not [], get the merge --> (feature, feature value)
        if len(tree_input) != 0:
            merge = self.lookup_tables[self.tree_input[0]]
        else:
            merge = self.getRandomElement()

        data = self.data.getTrainData()
        self.root = Node(merge, data.index, self.id, self.loss, 0, None)


    def getRandomElement(self):
        return self.data.getRandomElement()

    def getLeftRemainIndexs(self, df, feature, feature_value):
        """
        Get the remain indexs for the left child node
        """
        if df[feature].dtype != 'object':
            return list(df[df[feature] < feature_value].index)
        else:
            return list(df[df[feature] != feature_value].index)
    
    def getRightRemainIndexs(self, df, feature, feature_value):
        """
        Get the remain indexs for the right child node
        """
        if df[feature].dtype != 'object':
            return list(df[df[feature] >= feature_value].index)
        else:
            return list(df[df[feature] == feature_value].index)

    def _build_tree(self, data):
        queue = [self.root]
        while (len(queue) != 0):         
            current_node = queue.pop(0)
            current_data = data.loc[current_node.remain_indexs]
            #print(len(current_data), current_node.depth)

            #Internel node
            if current_node.depth < self.max_depth - 1:
                self.tree_array.append(current_node)  

                #If current node's depth eaual to the last node's depth, use the last node rules (oblivous tree)
                if current_node.parent == None:

                    #If the length of tree_input is 0, at init process, randomly assign feature and feature_val
                    if len(self.tree_input) == 0:
                        merge = self.getRandomElement()
                    else:
                        merge = self.lookup_tables[self.tree_input[current_node.depth + 1]]

                elif current_node.parent != None and current_node.depth != self.tree_array[-2].depth: 
                    if len(self.tree_input) == 0 or current_node.depth == self.max_depth - 2:
                        merge = self.getRandomElement()
                    else:
                        merge = self.lookup_tables[self.tree_input[current_node.depth + 1]]

                feature = current_node.merge[0]
                feature_value = current_node.merge[1]
                left_index = self.getLeftRemainIndexs(current_data, feature, feature_value)
                right_index = self.getRightRemainIndexs(current_data, feature, feature_value)
                self.id += 1
                if current_node.left == None:
                    current_node.left = Node(merge, left_index, self.id, self.loss, current_node.depth + 1, current_node)
                    queue.append(current_node.left)

                if current_node.right == None:
                    current_node.right= Node(merge, right_index, self.id, self.loss, current_node.depth + 1, current_node)
                    queue.append(current_node.right)
            
            #leaf node
            else:
                if len(self.target_name.split('_')) == 3:
                    label_name = 'label_' + self.target_name.split('_')[1]
                else:
                    label_name = 'label'

                current_node.update_predict_value(current_data, self.target_name, label_name)
                self.leaf_nodes.append(current_node)


    def predict(self, df):
        """
        Extract the rules from the oblivious tree, then manipulate the bits to get predict results of the DataFrame
        """

        # Extract the rules
        rules = [self.tree_array[2**i].merge for i in range(self.max_depth - 1)]
        predict_values = []
        print(rules) 
        for i in range(len(df)):
            str = ''
            for j in rules:              
                if df[j[0]].dtype != 'object':
                    if df.loc[i][j[0]] >= j[1]:
                        str += '1'
                    else:
                        str += '0'
                else: 
                    if df.loc[i][j[0]] == j[1]:
                        str += '1'
                    else:
                        str += '0'
            predict_values.append(self.leaf_nodes[int(str, 2)].predict_value)
        return predict_values

    def predictInstance(self, instance):
        """
        Get the predict result for an DataFrame instance
        """

        df = self.data.getData()
        rules = [self.tree_array[2**i].merge for i in range(self.max_depth - 1)]
        str = ''
        for rule in rules:
            if df[rule[0]].dtype != 'object':
                if instance[rule[0]] >= rule[1]:
                    str += '1'
                else:
                    str += '0'
            else:
                if instance[rule[0]] == rule[1]:
                    str += '1'
                else:
                    str += '0'
        return self.leaf_nodes[int(str, 2)].predict_value
            

    def getTreeArray(self):
        """
        Encode the oblivous tree as an array made by (feature, feature_val) --> key (only one each level)
        """
        return [self.data.getIndex(self.tree_array[2**i - 1].merge) for i in range(self.max_depth - 1)]
    
    def getTreeArrayAll(self):
        """
        Encode the oblivous tree as an array made by (feature, feature_val) --> key (all nodes at each level)
        """
        return [self.data.getIndex(node.merge) for node in self.tree_array]

    def getTreeNodes(self):
        """
        Encode the oblivous tree as an array made by nodes --> Class Node()
        """
        return [node for node in self.tree_array]
    
    def getLeafNodes(self):
        """
        Get the predict value for every leaf nodes
        """
        return [node.predict_value for node in self.leaf_nodes]

    def getNodesNums(self):
        """
        Get the number of remain indexs of each node
        """
        return sum([len(node.remain_indexs) for node in self.leaf_nodes])

    def checkNodesNums(self):
        """
        Check the correctness 
        """
        return sum([len(node.remain_indexs) for node in self.leaf_nodes]) == self.data.getLength()
