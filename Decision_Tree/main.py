import numpy as np
import pandas as pd


class AMS:
    def __init__(self, data):
        self.data = data

    # gini impurity for each attribute
    def gini_impurity(self, y):
        y_unique = np.unique(y)
        sum_p = 0
        for i in y_unique:
            p = (len(y[y == i]) / len(y))**2
            sum_p += p
        gini = 1 - sum_p
        return gini

    # gini impurity for each feature
    def gini_for_feature(self, y, feature):
        # get unique value of feature
        feature_unique = np.unique(feature)
        # get index of each unique value of feature
        feature_unique_index = {}
        for i in feature_unique:
            feature_unique_index[i] = np.where(feature == i)

        ginis_list = []
        for key, value in feature_unique_index.items():
            # caculate gini for each unique value of feature
            gini = self.gini_impurity(y.iloc[value])
            # append gini to ginis_list
            ginis_list.append(gini * (len(value[0]) / len(y)))
        # return sum of ginis_list
        return sum(ginis_list), feature_unique

    def choose_best_feature(self, y):
        # get all feature
        features = self.data.columns.tolist()
        # get gini for each feature
        ginis = [self.gini_for_feature(y, self.data[feature])
                 for feature in features]
        # get gini for each feature
        ginis_list = [i[0] for i in ginis]
        # get smallest gini and value of feature
        return features[np.argmin(ginis_list)], ginis[np.argmin(ginis_list)][1]


class Node:
    def __init__(self, feature=None, value=None, left=None, right=None, label=None):
        self.feature = feature  # Feature to split on
        self.value = value      # Possible values of that feature
        self.left = left        # Left child node
        self.right = right      # Right child node
        self.label = label      # Label if this is a leaf node


class DecisionTree:
    def __init__(self, max_depth=None):
        self.root = None
        self.max_depth = max_depth

    # def build_tree(self, data, y):
    #     meseuse = AMS(data)
    #     feature, value = meseuse.choose_best_feature(y)
    #     self.root = Node(feature, value)
    #     data = data.drop(columns=[feature])

    # def fit(self, data, y):
    #     self.build_tree(data, y)
    #     if self.root.left is not None:
    #         self.fit(data, y)
    #     if self.root.right is not None:
    #         self.fit(data, y)

    def build_tree(self, data, y, depth=0):
        # Nếu tất cả các nhãn giống nhau, trả về nút lá
        if len(np.unique(y)) == 1:
            return Node(label=np.unique(y)[0])

        # Dừng lại nếu đạt độ sâu tối đa hoặc không còn đặc trưng để chia
        if depth == self.max_depth or len(data.columns) == 0:
            most_common_label = np.argmax(np.bincount(y))
            return Node(label=most_common_label)

        # Chọn đặc trưng tốt nhất và chia dữ liệu
        ams = AMS(data)
        feature, values = ams.choose_best_feature(y)
        node = Node(feature=feature, value=values)

        # Chia dữ liệu thành hai tập con dựa trên giá trị của đặc trưng
        left_indices = data[feature] == values[0]
        right_indices = data[feature] != values[0]

        # Đệ quy xây dựng các nút con
        if len(data[left_indices]) > 0:
            node.left = self.build_tree(data[left_indices].drop(
                columns=[feature]), y[left_indices], depth + 1)
        if len(data[right_indices]) > 0:
            node.right = self.build_tree(data[right_indices].drop(
                columns=[feature]), y[right_indices], depth + 1)

        return node

    def fit(self, data, y):
        self.root = self.build_tree(data, y)

    def _predict_one(self, x, node):
        # Nếu là nút lá, trả về nhãn
        if node.label is not None:
            return node.label

        # Duyệt cây dựa trên giá trị của đặc trưng
        if x[node.feature] == node.value[0]:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def predict(self, X):
        return [self._predict_one(x, self.root) for _, x in X.iterrows()]

    def _visualize_tree(self, node, depth=0):
        if node is not None:
            # In đặc trưng và giá trị phân chia tại mỗi nút
            if node.label is not None:  # Nếu là nút lá
                print(f"{'|  ' * depth}--> Predict: {node.label}")
            else:
                print(f"{'|  ' * depth}{node.feature} == {node.value[0]}?")
                # Đệ quy với nút con bên trái
                self._visualize_tree(node.left, depth + 1)
                # Đệ quy với nút con bên phải
                print(f"{'|  ' * depth}{node.feature} != {node.value[0]}?")
                self._visualize_tree(node.right, depth + 1)

    def visualize(self):
        self._visualize_tree(self.root)


data = {
    'love_math': ['yes', 'yes', 'no', 'no', 'yes', 'yes', 'no'],
    'love_art': ['yes', 'no', 'yes', 'yes', 'yes', 'no', 'no'],
    'love_english': ['no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes'],
    'love_ai': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'no']}

df = pd.DataFrame(data, columns=data.keys())
y = df['love_ai']
df = df.drop(columns=['love_ai'])
# print(type(df['love_math'].unique()))
# print(type(y))

#
meseuse = AMS(df)
# print(meseuse.gini_impurity(df['love_math']))
# print(meseuse.gini_for_feature(y, df['love_math']))
# print(meseuse.choose_best_feature(y)[1])
# arr = np.array([1, 2, 3, 4, 5, 6, 7])
# print(arr[[2,3,4]])


tree = DecisionTree()
tree.fit(df, y)
tree.visualize()

test = pd.DataFrame([['yes', 'yes', 'no']], columns=df.columns)
result = tree.predict(test)
print(result)


attribute_names =  ['age', 'income','student', 'credit_rate']
class_name = 'default'
data1 ={
    'age' : ['youth', 'youth', 'middle_age', 'senior', 'senior', 'senior','middle_age', 'youth', 'youth', 'senior', 'youth', 'middle_age','middle_age', 'senior'],
    'income' : ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium','low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student' : ['no','no','no','no','yes','yes','yes','no','yes','yes','yes','no','yes','no'],
    'credit_rate' : ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair','excellent', 'excellent', 'fair', 'excellent'],
    'default' : ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes','yes', 'yes', 'yes', 'no']
}
df1 = pd.DataFrame (data1, columns=data1.keys())
print(df1)

y = df1['default']
X = df1.drop(columns=['default'])

model = DecisionTree(max_depth=3)
model.fit(X, y)
model.visualize()
