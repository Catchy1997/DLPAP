import pandas as pd
import os
from tqdm import tqdm

def csv_dir(filepath, path_list):
    for i in tqdm(os.listdir(filepath), ncols=50):
        path = os.path.join(filepath, i)
        if os.path.isdir(path):
            csv_dir(path, path_list)
        if path.endswith(".csv"):
            path_list.append(path)
    
    return path_list

# 整合所有csv
def sum_patent(name, filepath):
    path_list = []    
    path_list = csv_dir(filepath, path_list)
    print(name + " - csv文件数量：" + str(len(path_list)))

    patent_file_sum = pd.DataFrame()
    for path in path_list:
        patent_file = pd.read_csv(path, encoding='utf-8')
        patent_file_sum = patent_file_sum.append(patent_file)
    print(name + " - 涉及专利数量：" + str(len(patent_file_sum)))

    return patent_file_sum

if __name__ == '__main__':
    # 得到某一类的专利
    class_name = 'G-06-F-17'

    name_list = ["2010"]
    for name in tqdm(name_list, ncols=70):
        filepath = "E:/Pythonworkspace/patent/patent_data/Application/" + name + "/"
        
        patent_file_sum = sum_patent(name, filepath)
        after_class = patent_file_sum[patent_file_sum['cpc_class'] == class_name]
        # print("语料文本数量：" + str(len(after_class)))
        # 存储该类专利
        after_class.to_excel("E:/Pythonworkspace/patent/process_data/"+class_name+"/class/"+name+".xlsx")