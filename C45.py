from functools import reduce
from collections import defaultdict
import math
import numpy as np


class C45:
    def __init__(self):
        pass

    def fit(self, X, Y, max_depth, min_samples_leaf = 1):
        self.max_depth =  max_depth
        self.min_samples_leaf = min_samples_leaf
        self.root = self._create_tree(X, Y)

    def predict(self, X):
        return self.root.eval(X)

    def export_graphviz(self):
        #At least print from top to down and left to right
        pass

    def _information_content(self, Y, class_value):
        # Y is a Series of labels
        # i(y) con y P(y = clase) dado Y
        class_value_count = len(filter((lambda  x: x == class_value), Y))
        total_count = len(Y)
        Py = class_value_count / total_count
        return Py, math.log(1/Py,2)
        pass 

    def _entropy(self, Y):
        # Y is a Series of labels
        # returns the entropy of a set S in terms of the labels Y, I(S)
        classes = Y.unique()
        entropy = 0
        for c in classes:
            prob, info = self._information_content(Y, c)
            entropy = entropy + prob * info
        return entropy

        pass

    def _split_expected_entropy(self, Y, subsets_Y):
        #Y is a df of labels before split
        #subsets_Y is a list of dfs of labels after split
        #returns the expected entropy of splitting
        S_count = len(Y)
        expected_entropy = 0
        for s in subsets_Y:
            S_v_count = len(s)
            expected_entropy = expected_entropy + S_v_count/S_count * self._entropy(s)

        return expected_entropy
        pass

    def _discrete_feature_filter(self, X, Y, X_feature):
        # X is a df of features
        # Y is a df of labels
        # X_feature is the column name in X 
        # returns a list of [ (feature_value, subset_x, subset_y), (...), ... ] 
        # where the subset_x are all X's rows where X == feature_value 
        # and subset_y are all Y's rows where X == feature_value
        rows = list(zip(X, Y))
        res = defaultdict(list)
        
        for x, y in rows: 
            res[x[X_feature]].append((x,y))

        return [(k, v[0], v[1]) for k,v in res.items()]
        pass

    def _discrete_feature_subsets_Y(self, X, Y, X_feature):
        # X is a df of features
        # Y is a Series of labels
        # X_feature is the column name in X
        # returns a list of subsets of Y, one subset of each value of X_feature
        return map( (lambda x: x[2]), self._discrete_feature_filter(X, Y, X_feature) )
        pass

    def _continuous_feature_filter(self, X, Y, X_feature, cutoff):
        # X is a df of features
        # Y is a df of labels
        # X_feature is the column name in X 
        # returns a list of 2 subsets [(True, subset_x1, subset_y1), (False, subset_x2, subset_y2)] )
        # where one subset_x are all X's rows where X_feature > cutoff (False) and other subset_x where all X's rows <= cutoff (True)
        # and subset_y same conditions to subset_x
        rows = list(zip(X, Y))
        
        false_subset = filter((lambda x: x[0][X_feature] > cutoff), rows)
        true_subset = filter((lambda x: x[0][X_feature] <= cutoff), rows)
        
        return [(True, true_subset[0], true_subset[1]), (False, false_subset[0], false_subset[1])]
        pass
    
    
    def _continuous_feature_subsets_Y(self, X, Y, X_feature, cutoff):
        # X is a df of features
        # Y is a Series of labels
        # X_feature is the column name in X 
        # returns a list of the two subsets of Y splitted by the cutoff

        rows = list(zip(X, Y))
        
        greater_than_cutoff = filter((lambda x: x[0][X_feature] > cutoff), rows)
        less_or_equal_than_cutoff = filter((lambda x: x[0][X_feature] <= cutoff), rows)
        
        return [greater_than_cutoff, less_or_equal_than_cutoff]
        pass

    def _gain_criterion(self, Y, Y_entropy, subsets_Y):
        # Y is a Series of labels
        # Y_entropy is the entropy of the node's training set
        # subset_Y is a list of Series of Y for each attribute value or cutoff split
        # Returns the gain = entropy Y - splits expected entropy
        S_count = len(Y)
        gain = Y_entropy
        for s in subsets_Y:
            S_v_count = len(s)
            gain = gain - S_v_count/S_count * self._entropy(s)
        
        return gain
        
        pass

    def _gain(self, X, Y, X_feature, X_f_type):
        # X is a df of features/features
        # Y is a Series of labels
        # X_feature is the column name in X 
        # Returns (gain, cutoff or None)
        #print(str(X_feature) + "-" + str(X_f_type))
        Y_entropy = self._entropy(Y)
        if np.issubdtype(X_f_type, int):
            subsets_Y = self._discrete_feature_subsets_Y(X, Y, X_feature)
            return self._gain_criterion(Y, Y_entropy, subsets_Y), None
        elif np.issubdtype(X_f_type, float):
            cutoff, gain = self._continuous_feature_gain_of_best_cutoff(X, Y, X_feature, Y_entropy)
            return gain, cutoff
        pass

    def _continuous_feature_gain_of_best_cutoff(self, X, Y, X_feature, Y_entropy):
        # X is a df of features
        # Y is a df of labels
        # X_feature is the column name in X 
        # Calls _continuous__feature_filter for all cutoff taken from uniques X[X_feature]
        # Returns cutoff with best gain and it's gain

        cutoffs = X[X_feature].unique()
        
        cutoff_gain = int('-inf')
        best_cutoff = None

        S_count = len(Y)

        for c in cutoffs:
            partitions = self._continuous_feature_filter(X,Y,X_feature, c)
            current_gain = Y_entropy
            
            for p in partitions:
                Sv_count = len(p[2])
                current_gain = current_gain - Sv_count/S_count * self._entropy(p[2])

            if current_gain > cutoff_gain:
                cutoff_gain = current_gain
                best_cutoff = c

        return best_cutoff, cutoff_gain 
        
    
    def _best_feature(self, X, Y):
        # X is a df of features
        # Y is a Series of labels
        # Returns best feature, cutoff (if cont) or None (if discrete)
        features_and_types = list(zip(X.columns, X.dtypes))
        features = X.columns

        best_feature = None
        max_gain = int('-inf')
        cutoff = None

        for ft in features_and_types:
            current_gain, current_cutoff = self._gain(X,Y,ft[0], ft[1])
            if current_gain > max_gain:
                max_gain = current_gain
                cutoff = current_cutoff

        return best_feature, cutoff
        pass

    def _most_represented_class(self, Y):
        # Receives a train set X of features and a train set Y of labels
        # An creates a decision tree recursively  
        index = Y.value_counts().idxmax()
        return index

    def _create_tree(self, X, Y, depth = 0):
        # Receives a train set X of features/attributes and a train set Y of labels
        # An creates a decision tree recursively  

        #TODO add more cases

        if (len(X.columns) == 0):
            return LeafNode(self._most_represented_class(Y))
        elif (len(Y.unique()) == 1):
            classs = Y.unique()[0]
            return LeafNode(classs)
        else:
            best_question_feature, cutoff = self._best_feature(X, Y)

            if cutoff is None: #discrete
                node = DiscreteNode(best_question_feature)
                partitions = self._discrete_feature_subsets_Y(X, Y, best_question_feature)
                for p in partitions:
                    node.add_children(p[2],self._create_tree(p[0],p[1]))

            else: #cont
                node = ContinuousNode(best_question_feature, cutoff)
                partitions = self._continuous_feature_filter(X, Y, best_question_feature, cutoff)
                for p in partitions:
                    node.add_children(p[2],self._create_tree(p[0],p[1]))

            return node

class LeafNode:

    def __init__(self, classs):
        self.classs = classs

    def eval(self, data):
        #Takes a df and predicts using the classs
        pass

class DiscreteNode:

    def __init__(self, feature):
        self.feature = feature
        self.child_nodes = dict()

    def eval(self, data):
        #Takes a df and predicts using the child_nodes 
        pass

    def add_children(self, value, node):
        self.child_nodes[value] = node

class ContinuousNode:

    def __init__(self, feature, cutoff):
        self.feature = feature
        self.cutoff = cutoff
        self.child_nodes = dict()

    def eval(self, data):
        #Takes a df and predicts using the child_nodes 
        pass
    
    def add_children(self, value, node):
        self.child_nodes[value] = node
