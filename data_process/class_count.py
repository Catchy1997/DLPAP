import pandas as pd
import os
from tqdm import tqdm

def csv_dir(filepath, path_list):
    for i in tqdm(os.listdir(filepath), ncols=50):
        path = os.path.join(filepath, i)
        if os.path.isdir(path):
            csv_dir(path, path_list)
        if path.endswith(".xlsx"):
            path_list.append(path)
    
    return path_list

# 整合所有csv
def sum_patent(name, filepath):
    path_list = []    
    path_list = csv_dir(filepath, path_list)
    print(name + " - csv文件数量：" + str(len(path_list)))

    patent_file_sum = pd.DataFrame()
    for path in path_list:
        patent_file = pd.read_excel(path, encoding='utf-8')
        patent_file_sum = patent_file_sum.append(patent_file)
    print(name + " - 涉及专利数量：" + str(len(patent_file_sum)))

    return patent_file_sum

def count_class(name, patent_file_sum):
    filename = "E:/Pythonworkspace/patent/process_data/class/" + name + ".csv"

    # 查看分类的情况
    object_result = patent_file_sum.describe(include=['object'])
    print(object_result)
    
    # 统计
    class_con = patent_file_sum.groupby('cpc_class').describe()
    class_count = class_con.loc[:,'application_id']['count'] # 计数，每一类有多少个专利
    
    # 按索引排序 sort_index方法
    # class_count.sort_index(ascending=False)
    # 按值排序 sort_values方法
    sort_count = class_count.sort_values(ascending=False)
    index_list = list(sort_count.index)
    value_list = list(sort_count.values)
    
    df = pd.DataFrame({'cpc_class':index_list,'count':value_list})
    if os.path.exists(filename):
        df.to_csv(filename, header=0, mode='a', index=False, sep=',')
    else:
        df.to_csv(filename, mode='a', index=False, sep=',')

if __name__ == '__main__':
    # 统计每一年专利的分类
    name_list = ["2007", "2008", "2009", "2010"]
    for name in tqdm(name_list, ncols=70):
        filepath = "E:/Pythonworkspace/patent/patent_data/Application/" + name + "/"
        patent_file_sum = sum_patent(name, filepath)
        count_class(name, patent_file_sum)

    # # 2.得到某一类的专利——语料
    # filepath = "E:/Pythonworkspace/patent/patent_data/暂时不用/corpus"
    # patent_file_sum = sum_patent(filepath)
    # class_name = 'G-06-F-17'
    # after_class = patent_file_sum[patent_file_sum['cpc_class'] == class_name]
    # print("语料文本数量：" + str(len(after_class)))
    # # 存储该类专利
    # after_class.to_excel("E:/Pythonworkspace/patent/process_data/sample3_"+class_name+"/2007_2009.xlsx")

    # 3.得到某一类的专利——样本
    # filepath = "E:/Pythonworkspace/patent/patent_data/暂时不用/sample"
    # patent_file_sum = sum_patent(filepath)
    # class_name = 'G-06-F-17'
    # after_class = patent_file_sum[patent_file_sum['cpc_class'] == class_name]
    # print("样本文本数量：" + str(len(after_class)))
    # # 存储该类专利
    # after_class.to_excel("E:/Pythonworkspace/patent/process_data/sample3_"+class_name+"/2010.xlsx")