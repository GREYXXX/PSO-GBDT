import numpy as np

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

        #Get lookup tables
        self.lookup_tables = self.data.get_lookup_table()

        #Get the merge --> (feature, feature value) or raise error
        if len(tree_input) == self.max_depth:
            merge = self.lookup_tables[self.tree_input[0]]
        else:
            print("tree_input is {}, depth is {}".format(len(tree_input), self.max_depth))
            raise ValueError("Input tree error")

        data = self.data.get_train_data()
        self.root = Node(merge, data.index, self.id, self.loss, 0, None)


    def get_random_element(self):
        return self.data.get_random_element()

    def get_left_remain_indexs(self, df, feature, feature_value):
        """
        Get the remain indexs for the left child node
        """

        if df[feature].dtype != 'object':
            return df[df[feature] < feature_value].index
        else:
            return df[df[feature] != feature_value].index
        
    
    def get_right_remain_indexs(self, df, feature, feature_value):
        """
        Get the remain indexs for the right child node
        """
        if df[feature].dtype != 'object':
            return df[df[feature] >= feature_value].index
        else:
            return df[df[feature] == feature_value].index


    def build_tree(self, data):
        queue = [self.root]
        while (len(queue) != 0):         
            current_node = queue.pop(0)
            current_data = data.loc[current_node.remain_indexs]
            self.tree_array.append(current_node) 

            #Internel node
            if current_node.depth < self.max_depth - 1: 

                #If current node's depth eaual to the last node's depth, use the last node rules (oblivous tree)
                if current_node.parent == None:
                    merge = self.lookup_tables[self.tree_input[current_node.depth + 1]]

                elif current_node.parent != None and current_node.depth != self.tree_array[-2].depth: 
                    merge = self.lookup_tables[self.tree_input[current_node.depth + 1]]

                feature = current_node.merge[0]
                feature_value = current_node.merge[1]
                left_index = self.get_left_remain_indexs(current_data, feature, feature_value)
                right_index = self.get_right_remain_indexs(current_data, feature, feature_value)
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
        # Extract the leaf values
        leaf_values = np.array([self.leaf_nodes[i].predict_value for i in range(len(self.leaf_nodes))])
        idxs = np.zeros(len(df)).astype(int)
        cur_depth = 0
        for rule in rules:
            if df[rule[0]].dtype != "object":
                condlist   = [df[rule[0]] >= rule[1], df[rule[0]] < rule[1]] 
                choicelist = [2**(len(rules) - cur_depth - 1), 0]

            else:
                condlist   = [df[rule[0]] == rule[1], df[rule[0]] != rule[1]] 
                choicelist = [2**(len(rules) - cur_depth - 1), 0]
            
            idxs = idxs + np.select(condlist, choicelist)
            cur_depth += 1 
        # values = np.array([self.leaf_nodes[i].predict_value for i in idxs])
    
        return leaf_values[idxs]


    def get_rules(self):
        return [self.tree_array[2**i].merge for i in range(self.max_depth - 1)]

    def predict_instance(self, instance, rules):
        """
        Get the predict result for an DataFrame instance
        """

        df = self.data.get_data()
        # str = ''
        idx = 0
        for i in range(len(rules)):
            if df[rules[i][0]].dtype != 'object':
                if instance[rules[i][0]] >= rules[i][1]:
                    idx += 2**(len(rules) - i - 1)
                else:
                    idx += 0
            else:
                if instance[rules[i][0]] == rules[i][1]:
                    idx += 2**(len(rules) - i - 1)
                else:
                    idx += 0

        return self.leaf_nodes[idx].predict_value
            

    def get_tree_array(self):
        """
        Encode the oblivous tree as an array made by (feature, feature_val) --> key (only one each level)
        """
        return [self.data.get_index(self.tree_array[2**i - 1].merge) for i in range(self.max_depth)]
    
    def get_tree_array_all(self):
        """
        Encode the oblivous tree as an array made by (feature, feature_val) --> key (all nodes at each level)
        """
        return [self.data.get_index(node.merge) for node in self.tree_array]

    def get_tree_nodes(self):
        """
        Encode the oblivous tree as an array made by nodes --> Class Node()
        """
        return [node for node in self.tree_array]
    
    def get_leaf_nodes(self):
        """
        Get the predict value for every leaf nodes
        """
        return [node.predict_value for node in self.leaf_nodes]

    def get_nodes_nums(self):
        """
        Get the number of remain indexs of each node
        """
        return sum([len(node.remain_indexs) for node in self.leaf_nodes])

    def check_nodes_nums(self):
        """
        Check the correctness 
        """
        return sum([len(node.remain_indexs) for node in self.leaf_nodes]) == self.data.getLength()