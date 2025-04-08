'''
   开发人员: guojunlong 00020540
   功能:RF、Adaboost、GDBT、XGBoost算法的工程化
   时间: 2021.12.10
   要求:
'''
import os
import joblib
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier, plot_importance, to_graphviz
from model_engineering import DTActivityClassify, RF_engineering, GDBT_engineering, AdaBoost_engineering, XGBoost_engineering, lightGBM_engineering, catBoost_engineering, include_str

NORM_SCORE = 10000
index_str = ""
threshold_str = "{"

def plot_tree_fig(estimator, pngFileName):
    plt.figure(figsize=(24, 24), dpi=360)
    plot_tree(estimator, filled=True, rounded=True)
    plt.savefig(pngFileName)
    #plt.show()
    plt.close()

def treeStr2treeDict(treeStr_list):
    import re
    tree_dict = {}
    pattern = re.compile(r'\d+.\d+|\d+')

    for str_value in treeStr_list:
        match_value_list = pattern.findall(str_value)
        if 'value>' in str_value:
            node_index = int(match_value_list[0])
            feature_index = int(match_value_list[1])
            threshold = math.floor(eval(match_value_list[2]))
            tree_dict[node_index] = {
                "feature": feature_index,
                "threshold": threshold,
                "left": -1,
                "right": -1
            }
        elif 'val = ' in str_value:
            node_index = int(match_value_list[0])
            value = eval(match_value_list[1])
            tree_dict[node_index] = {
                "value": value
            }
        elif '->' in str_value:
            node_index = int(match_value_list[0])
            child_index = int(match_value_list[1])
            if '[label=No]' in str_value:
                tree_dict[node_index]["left"] = child_index
            else:
                tree_dict[node_index]["right"] = child_index
        else:
            raise Exception("node is miss match!!!")

    return tree_dict

def catBoostFile2treeDict(catboost_model):
    tree_dict_list = []
    tree_split_feature_index = catboost_model.tree_split_feature_index
    tree_split_border = catboost_model.tree_split_border
    float_feature_borders = catboost_model.float_feature_borders
    float_features_index = catboost_model.float_features_index
    tree_depth_list = catboost_model.tree_depth
    leaf_values = catboost_model.leaf_values

    for tree_index, tree_depth in enumerate(tree_depth_list):
        tree_dict = {}
        for tree_depth_index in range(tree_depth + 1):
            for node_index in range(2 ** tree_depth_index - 1, 2 ** (tree_depth_index + 1) - 1):
                if tree_depth_index < tree_depth:
                    index = tree_depth * (tree_index + 1) - tree_depth_index - 1
                    tree_dict[node_index] = {
                        "feature": float_features_index[tree_split_feature_index[index]],
                        "threshold": float_feature_borders[tree_split_feature_index[index]][
                            tree_split_border[index] - 1],
                        "left": 2 * node_index + 1,
                        "right": 2 * node_index + 2
                    }
                else:
                    leaf_index = tree_index * (2 ** tree_depth) + node_index - (2 ** tree_depth - 1)
                    tree_dict[node_index] = {
                        "value": leaf_values[leaf_index]
                    }
        tree_dict_list.append(tree_dict)

    return tree_dict_list

def PreOrderXGBoost(tree_dict, node_of_tree):
    global index_str
    global threshold_str

    if 'leaf' in tree_dict.keys():#到达叶子节点
        threshold = int(round(NORM_SCORE*tree_dict['leaf']))
        index = -1
        index_str = index_str + "{},".format(index)
        threshold_str = threshold_str + "{},".format(threshold)
        node_of_tree +=1
        return node_of_tree

    index = int(tree_dict['split'].replace("f", ""))
    threshold = math.floor(tree_dict['split_condition'])#向下取证整
    index_str = index_str + "{},".format(index)
    threshold_str = threshold_str + "{},".format(threshold)
    node_of_tree +=1

    for children in tree_dict['children']:
        if tree_dict['yes'] == children['nodeid']:
            tree_dict_left = children
        elif tree_dict['no'] == children['nodeid']:
            tree_dict_right = children
        else:
            raise Exception("nodeid is miss match!!!")

    node_of_tree = PreOrderXGBoost(tree_dict_left, node_of_tree)
    node_of_tree = PreOrderXGBoost(tree_dict_right, node_of_tree)
    return node_of_tree

def PreOrderLightGBM(tree_dict, node_of_tree):
    global index_str
    global threshold_str

    if 'leaf_index' in tree_dict.keys():#到达叶子节点
        threshold = int(round(NORM_SCORE*tree_dict['leaf_value']))
        index_str = index_str + "-1,"
        threshold_str = threshold_str + "{},".format(threshold)
        node_of_tree +=1
        return node_of_tree

    index = tree_dict['split_feature']
    threshold = math.floor(tree_dict['threshold'])#向下取整
    index_str = index_str + "{},".format(index)
    threshold_str = threshold_str + "{},".format(threshold)
    node_of_tree +=1

    tree_dict_left = tree_dict['left_child']
    tree_dict_right = tree_dict['right_child']

    node_of_tree = PreOrderLightGBM(tree_dict_left, node_of_tree)
    node_of_tree = PreOrderLightGBM(tree_dict_right, node_of_tree)
    return node_of_tree

def PreOrderCatBoost(tree_dict, node_index):
    global index_str
    global threshold_str

    if 'value' in tree_dict[node_index].keys():#到达叶子节点
        threshold = int(round(NORM_SCORE*tree_dict[node_index]['value']))
        index_str = index_str + "-1,"
        threshold_str = threshold_str + "{},".format(threshold)
        return

    index = tree_dict[node_index]['feature']
    threshold = math.floor(tree_dict[node_index]['threshold'])#向下取证整
    index_str = index_str + "{},".format(index)
    threshold_str = threshold_str + "{},".format(threshold)

    left_node_index = tree_dict[node_index]['left']
    right_node_index = tree_dict[node_index]['right']

    PreOrderCatBoost(tree_dict, left_node_index)
    PreOrderCatBoost(tree_dict, right_node_index)
    return

#前序遍历
def PreOrder(node, stack, dtc, tree_type, learning_rate = None):
    global index_str
    global threshold_str

    index = dtc.tree_.feature[node]
    if(-2 == index):
        index = -1
    index_str = index_str + "{},".format(index)

    threshold = dtc.tree_.threshold[node]
    if (-2 == threshold) and (-1 == index):
        if ("Adaboost" == tree_type) or ("RF" == tree_type):
            value = dtc.tree_.value[node]
            proba = value[:, :dtc.n_classes_]
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer
            threshold = int(round(NORM_SCORE*proba[0, 1]))#四舍五入取整
            if "Adaboost" == tree_type:
                if NORM_SCORE == threshold:
                    threshold = 9999
        elif "GDBT" == tree_type:
            if learning_rate is None:
                raise Exception("learning_rate is None")
            value = dtc.tree_.value[node][0, 0]
            threshold = int(round(NORM_SCORE * learning_rate * value))  # 四舍五入取整,乘以学习率，在推理阶段不再需要乘以学习率
        else:
            raise Exception("tree_type:{} is err".format(tree_type))
    else:
        threshold = math.floor(threshold) #向下取证整
    threshold_str = threshold_str + "{},".format(threshold)

    children_left = dtc.tree_.children_left[node]
    children_right = dtc.tree_.children_right[node]

    if (-1 == children_right):
        if(0 == len(stack)):
            return
        node_value = stack.pop()
        PreOrder(node_value, stack, dtc, tree_type, learning_rate)
    else:
        stack.append(children_right)
        PreOrder(children_left, stack, dtc, tree_type, learning_rate)

def extractAllDTParametersAdaboost(adaboost_info, targetFileName):
    global index_str
    global threshold_str
    model_params = adaboost_info['params']
    className2model_dict = adaboost_info['model']
    first_class_name = list(className2model_dict.keys())[0]

    ### 7. plot tree
    pngFileName = targetFileName.replace(".c", ".png")
    plot_tree_fig(className2model_dict[first_class_name].estimators_[0], pngFileName)

    all_str = "/**** \n Adaboost params:{}\n ****/\n\n".format(model_params)
    all_str = all_str + "{}\n".format(include_str)
    tree_num_define_str = "NUM_OF_ADABOOST_TREE"
    node_num_define_str = "NUM_OF_ADABOOST_NODE"

    num_of_tree = className2model_dict[first_class_name].n_estimators
    max_node_of_tree = max([className2model_dict[classId].estimators_[index].tree_.node_count for classId in className2model_dict.keys() for index in range(num_of_tree)])
    all_str = all_str + "#define {} {}\n".format(tree_num_define_str, num_of_tree)
    all_str = all_str + "#define {} {}\n".format(node_num_define_str, max_node_of_tree)

    if "SAMME.R" == model_params["algorithm"]:
        all_str = all_str + "unsigned char g_sammeRFlag = 1; //0: algorithm is SAMME, 1: algorithm is SAMME.R, Other values are considered to be 1\n\n"
    else:
        all_str = all_str + "unsigned char g_sammeRFlag = 0; //0: algorithm is SAMME, 1: algorithm is SAMME.R, Other values are considered to be 1\n\n"

    for className in className2model_dict.keys():
        weight_str = "float const g_AdaBoostTree{}Weight[{}] = ".format(className, tree_num_define_str) + "{"
        index_str = "signed char const g_AdaBoostTree{}Index[{}][{}] = ".format(className, tree_num_define_str, node_num_define_str) + "{\n"
        threshold_str = "short const g_AdaBoostTree{}Value[{}][{}] = ".format(className, tree_num_define_str, node_num_define_str) + "{\n"
        for index in range(num_of_tree):
            estimator = className2model_dict[className].estimators_[index]
            node_of_tree = estimator.tree_.node_count

            stack = []
            node = 0
            index_str = index_str + "    {"
            threshold_str = threshold_str + "    {"
            PreOrder(node, stack, estimator, tree_type="Adaboost")
            weight_str = weight_str + "{:.4f},".format(className2model_dict[className].estimator_weights_[index])

            for _ in range(max_node_of_tree-node_of_tree):
                index_str = index_str + "-1,"
                threshold_str = threshold_str + "1,"
            index_str = index_str + "},\n"
            threshold_str = threshold_str + "},\n"

        weight_str = weight_str[0:-1] + "};\n"
        index_str = index_str[0:-2] + "\n};\n"
        threshold_str = threshold_str[0:-2] + "\n};\n"
        all_str = all_str + weight_str + index_str + threshold_str
    print(all_str, file=open(targetFileName, 'w'))
    print(DTActivityClassify, file=open(targetFileName, 'a'))
    print(AdaBoost_engineering, file=open(targetFileName, 'a'))

def extractAllDTParametersRF(model_info, targetFileName):
    global index_str
    global threshold_str
    model_params = model_info['params']
    className2model_dict = model_info['model']
    first_class_name = list(className2model_dict.keys())[0]

    ### 7. plot tree
    #pngFileName = targetFileName.replace(".c", ".png")
    #plot_tree_fig(className2model_dict[first_class_name].estimators_[0], pngFileName)

    all_str = "/**** \n RandomForestClassifier params:{}\n ****/\n\n".format(model_params)
    all_str = all_str + "{}\n".format(include_str)
    tree_num_define_str = "NUM_OF_RANDOM_FOREST_TREE"
    node_num_define_str = "NUM_OF_RANDOM_FOREST_NODE"

    num_of_tree = className2model_dict[first_class_name].n_estimators
    max_node_of_tree = max([className2model_dict[classId].estimators_[index].tree_.node_count for classId in className2model_dict.keys() for index in range(num_of_tree)])
    all_str = all_str + "#define {} {}\n".format(tree_num_define_str, num_of_tree)
    all_str = all_str + "#define {} {}\n".format(node_num_define_str, max_node_of_tree)

    for className in className2model_dict.keys():
        index_str = "signed char const g_RandomForestTree{}Index[{}][{}] = ".format(className, tree_num_define_str, node_num_define_str) + "{\n"
        threshold_str = "short const g_RandomForestTree{}Value[{}][{}] = ".format(className, tree_num_define_str, node_num_define_str) + "{\n"
        for index in range(num_of_tree):
            estimator = className2model_dict[className].estimators_[index]
            node_of_tree = estimator.tree_.node_count

            stack = []
            node = 0
            index_str = index_str + "    {"
            threshold_str = threshold_str + "    {"
            PreOrder(node, stack, estimator, tree_type="RF")

            for _ in range(max_node_of_tree-node_of_tree):
                index_str = index_str + "-1,"
                threshold_str = threshold_str + "0,"
            index_str = index_str + "},\n"
            threshold_str = threshold_str + "},\n"

        index_str = index_str[0:-2] + "\n};\n"
        threshold_str = threshold_str[0:-2] + "\n};\n"
        all_str = all_str + index_str + threshold_str
    print(all_str, file=open(targetFileName, 'w'))
    print(DTActivityClassify, file=open(targetFileName, 'a'))
    print(RF_engineering, file=open(targetFileName, 'a'))


def extractAllDTParametersGDBT(model_info, targetFileName):
    global index_str
    global threshold_str
    model_params = model_info['params']
    className2model_dict = model_info['model']
    first_class_name = list(className2model_dict.keys())[0]

    ### plot tree
    pngFileName = targetFileName.replace(".c", ".png")
    plot_tree_fig(className2model_dict[first_class_name].estimators_[0, 0], pngFileName)

    all_str = "/**** \n GDBT params:{}\n ****/\n\n".format(model_params)
    all_str = all_str + "{}\n".format(include_str)
    tree_num_define_str = "NUM_OF_GDBT_TREE"
    node_num_define_str = "NUM_OF_GDBT_NODE"

    num_of_tree = className2model_dict[first_class_name].n_estimators
    max_node_of_tree = max([className2model_dict[classId].estimators_[index][0].tree_.node_count for classId in className2model_dict.keys() for index in range(num_of_tree)])
    all_str = all_str + "#define {} {}\n".format(tree_num_define_str, num_of_tree)
    all_str = all_str + "#define {} {}\n".format(node_num_define_str, max_node_of_tree)

    for className in className2model_dict.keys():
        learning_rate = className2model_dict[className].learning_rate
        proba_pos_class = className2model_dict[className].init_.class_prior_[1]
        eps = np.finfo(np.float32).eps
        proba_pos_class = np.clip(proba_pos_class, eps, 1 - eps)
        f0_value = np.log(proba_pos_class / (1 - proba_pos_class)) # log(x / (1 - x)) is the inverse of the sigmoid (expit) function
        f0_value = int(round(NORM_SCORE * f0_value))

        F0_str = "short const g_GDBTTree{}F0Value = {};\n".format(className, f0_value)
        index_str = "signed char const g_GDBTTree{}Index[{}][{}] = ".format(className, tree_num_define_str, node_num_define_str) + "{\n"
        threshold_str = "short const g_GDBTTree{}Value[{}][{}] = ".format(className, tree_num_define_str, node_num_define_str) + "{\n"
        for index in range(num_of_tree):
            estimator = className2model_dict[className].estimators_[index][0]
            node_of_tree = estimator.tree_.node_count

            stack = []
            node = 0
            index_str = index_str + "    {"
            threshold_str = threshold_str + "    {"
            PreOrder(node, stack, estimator, tree_type="GDBT", learning_rate = learning_rate)

            for _ in range(max_node_of_tree-node_of_tree):
                index_str = index_str + "-1,"
                threshold_str = threshold_str + "0,"
            index_str = index_str + "},\n"
            threshold_str = threshold_str + "},\n"

        index_str = index_str[0:-2] + "\n};\n"
        threshold_str = threshold_str[0:-2] + "\n};\n"
        all_str = all_str + F0_str + index_str + threshold_str
    print(all_str, file=open(targetFileName, 'w'))
    print(DTActivityClassify, file=open(targetFileName, 'a'))
    print(GDBT_engineering, file=open(targetFileName, 'a'))

def extractAllDTParametersXGBoost(model_info, targetFileName):
    global index_str
    global threshold_str
    model_params = model_info['params']
    className2model_dict = model_info['model']
    first_class_name = list(className2model_dict.keys())[0]

    ### plot tree
    pngFileName = targetFileName.replace(".c", "")
    # digraph = to_graphviz(className2model_dict[first_class_name], num_trees=0)
    # digraph.format = 'png'
    # digraph.view(pngFileName)

    all_str = "/**** \n XGBoost params:{}\n ****/\n\n".format(model_params)
    all_str = all_str + "{}\n".format(include_str)
    tree_num_define_str = "NUM_OF_XGBOOST_TREE"
    node_num_define_str = "NUM_OF_XGBOOST_NODE"

    num_of_tree = className2model_dict[first_class_name].n_estimators
    max_node_of_tree = 2**(className2model_dict[first_class_name].max_depth + 1) - 1
    all_str = all_str + "#define {} {}\n".format(tree_num_define_str, num_of_tree)
    all_str = all_str + "#define {} {}\n".format(node_num_define_str, max_node_of_tree)
    for className in className2model_dict.keys():
        index_str = "signed char const g_XGBoostTree{}Index[{}][{}] = ".format(className, tree_num_define_str, node_num_define_str) + "{\n"
        threshold_str = "short const g_XGBoostTree{}Value[{}][{}] = ".format(className, tree_num_define_str, node_num_define_str) + "{\n"

        tree_list = className2model_dict[first_class_name].get_booster().get_dump(dump_format='json')
        for tree_str in tree_list:
            node_of_tree = 0
            tree_dict = eval(tree_str)
            index_str = index_str + "    {"
            threshold_str = threshold_str + "    {"
            node_of_tree = PreOrderXGBoost(tree_dict, node_of_tree)

            for _ in range(max_node_of_tree - node_of_tree):
                index_str = index_str + "-1,"
                threshold_str = threshold_str + "0,"
            index_str = index_str + "},\n"
            threshold_str = threshold_str + "},\n"

        index_str = index_str[0:-2] + "\n};\n"
        threshold_str = threshold_str[0:-2] + "\n};\n"
        all_str = all_str + index_str + threshold_str
    print(all_str, file=open(targetFileName, 'w'))
    print(DTActivityClassify, file=open(targetFileName, 'a'))
    print(XGBoost_engineering, file=open(targetFileName, 'a'))

def extractAllDTParametersLightGBM(model_info, targetFileName):
    from lightgbm import create_tree_digraph
    global index_str
    global threshold_str
    model_params = model_info['params']
    className2model_dict = model_info['model']
    first_class_name = list(className2model_dict.keys())[0]

    ### plot tree
    pngFileName = targetFileName.replace(".c", "")
    digraph = create_tree_digraph(className2model_dict[first_class_name], orientation='vertical')
    digraph.format = 'png'
    digraph.view(pngFileName)

    all_str = "/**** \n lightGBM params:{}\n ****/\n\n".format(model_params)
    all_str = all_str + "{}\n".format(include_str)
    tree_num_define_str = "NUM_OF_LIGHTGBM_TREE"
    node_num_define_str = "NUM_OF_LIGHTGBM_NODE"

    num_of_tree = className2model_dict[first_class_name].n_estimators
    max_node_of_tree = 2*max([className2model_dict[classId].booster_.dump_model()['tree_info'][index]['num_leaves'] for classId in className2model_dict.keys() for index in range(num_of_tree)])-1
    all_str = all_str + "#define {} {}\n".format(tree_num_define_str, num_of_tree)
    all_str = all_str + "#define {} {}\n".format(node_num_define_str, max_node_of_tree)
    for className in className2model_dict.keys():
        index_str = "signed char const g_lightGBMTree{}Index[{}][{}] = ".format(className, tree_num_define_str, node_num_define_str) + "{\n"
        threshold_str = "short const g_lightGBMTree{}Value[{}][{}] = ".format(className, tree_num_define_str, node_num_define_str) + "{\n"

        tree_list = className2model_dict[className].booster_.dump_model()['tree_info']
        for tree_info in tree_list:
            node_of_tree = 0
            tree_dict = tree_info['tree_structure']
            index_str = index_str + "    {"
            threshold_str = threshold_str + "    {"
            node_of_tree = PreOrderLightGBM(tree_dict, node_of_tree)

            for _ in range(max_node_of_tree - node_of_tree):
                index_str = index_str + "-1,"
                threshold_str = threshold_str + "0,"
            index_str = index_str + "},\n"
            threshold_str = threshold_str + "},\n"

        index_str = index_str[0:-2] + "\n};\n"
        threshold_str = threshold_str[0:-2] + "\n};\n"
        all_str = all_str + index_str + threshold_str
    print(all_str, file=open(targetFileName, 'w'))
    print(DTActivityClassify, file=open(targetFileName, 'a'))
    print(lightGBM_engineering, file=open(targetFileName, 'a'))

def extractAllDTParametersCatBoost(model_info, targetFileName):
    global index_str
    global threshold_str
    model_params = model_info['params']
    className2model_dict = model_info['model']
    first_class_name = list(className2model_dict.keys())[0]

    ### plot tree
    # pngFileName = targetFileName.replace(".c", "")
    # digraph = className2model_dict[first_class_name].plot_tree(1)
    # digraph.format = 'png'
    # digraph.view(pngFileName)

    all_str = "/**** \n catboost params:{}\n ****/\n\n".format(model_params)
    all_str = all_str + "{}\n".format(include_str)
    tree_num_define_str = "NUM_OF_CATBOOST_TREE"
    node_num_define_str = "NUM_OF_CATBOOST_NODE"

    num_of_tree = className2model_dict[first_class_name].tree_count_
    node_of_tree = 2**(className2model_dict[first_class_name].get_all_params()['depth'] + 1) - 1
    all_str = all_str + "#define {} {}\n".format(tree_num_define_str, num_of_tree)
    all_str = all_str + "#define {} {}\n".format(node_num_define_str, node_of_tree)
    for className in className2model_dict.keys():
        index_str = "signed char const g_catBoostTree{}Index[{}][{}] = ".format(className, tree_num_define_str, node_num_define_str) + "{\n"
        threshold_str = "short const g_catBoostTree{}Value[{}][{}] = ".format(className, tree_num_define_str, node_num_define_str) + "{\n"

        className2model_dict[className].save_model('catboost_model_temp.py', format="python", export_parameters=None)
        #import catboost_model_temp
        catboost_model_temp = "sdfds"
        import importlib
        importlib.reload(catboost_model_temp)
        tree_dict_list = catBoostFile2treeDict(catboost_model_temp.catboost_model)
        os.remove('catboost_model_temp.py')
        for tree_dict in tree_dict_list:
            node_index = 0
            index_str = index_str + "    {"
            threshold_str = threshold_str + "    {"
            PreOrderCatBoost(tree_dict, node_index)

            index_str = index_str + "},\n"
            threshold_str = threshold_str + "},\n"

        index_str = index_str[0:-2] + "\n};\n"
        threshold_str = threshold_str[0:-2] + "\n};\n"
        all_str = all_str + index_str + threshold_str
    print(all_str, file=open(targetFileName, 'w'))
    print(DTActivityClassify, file=open(targetFileName, 'a'))
    print(catBoost_engineering, file=open(targetFileName, 'a'))

modelType2Func_dict = {
    "RF": extractAllDTParametersRF,
    "Adaboost": extractAllDTParametersAdaboost,
    "GDBT": extractAllDTParametersGDBT,
    "XGBoost": extractAllDTParametersXGBoost,
    "lightGBM": extractAllDTParametersLightGBM,
    "catBoost": extractAllDTParametersCatBoost
}

def extractAllDTParameters1(model_info, targetFileName, model_type):
    """
    :param model_info: type dict, must has keys: "params" and "model"
           model_info["model"] = {"className_1" : Classifier_1,
                                  "className_2" : Classifier_2,
                                    .......
                                  "className_n" : Classifier_n
                                  };
           Classifier_n can be AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
    :param targetFileName: path to save .c
    :param model_type: must be "RF"、"Adaboost"、"GDBT"；
                       "RF" is RandomForestClassifier，
                       "Adaboost" is AdaBoostClassifier，
                       "GDBT" is GradientBoostingClassifier,
                       "XGBoost" is XGBClassifier,
                       "lightGBM" is LGBMClassifier,
                       "catBoost" is CatBoostClassifier
    :return:
    """
    if model_type not in modelType2Func_dict.keys():
        print("err: {} is not supported model_type".format(model_type))
        return
    if not isinstance(model_info, dict):
        print("err: model_info is not dict")
        return
    if "params" not in model_info.keys() or "model" not in model_info.keys():
        print("err: model_info not has keys:'params' or 'model'")
        return
    if not isinstance(model_info["model"], dict):
        print("err: model_info['model'] is not dict")
        return

    modelType2Func_dict[model_type](model_info, targetFileName)


if __name__ == '__main__':
    ### 7. load model
    model_name_lightGBM = r'C:\Users\g00020540\Desktop\treeModemengineering\lightGBM_model_dict_2022_04_14_16_09_09_f1_0.9240788723376584.model'
    model_name = model_name_lightGBM
    model_info = joblib.load(model_name)
    className2model_dict = model_info['model']

    ### 8. extract the Decision Tree parameters for C platform
    targetFileName = model_name.replace(".model", ".c")
    extractAllDTParameters1(model_info, targetFileName, model_type="lightGBM")
    #extractAllDTParametersGDBT(GBDT_info, targetFileName)

    print("finish")
