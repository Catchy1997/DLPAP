import os, re
import pandas as pd
from tqdm import tqdm

# 建立assignee字典
def build_assignee_dic(start_year, end_year):
    assignee_index_dic = {}
    assignee_file_sum = pd.DataFrame()

    for i in range(start_year,end_year+1):
        csv_filepath = "E:/Pythonworkspace/patent/process_data/index/assignee_file_"+str(i)+".csv"
        assignee_file = pd.read_csv(csv_filepath, header=None, encoding='utf-8')
        application_id_list = assignee_file.iloc[:,0]
        assignee_list = assignee_file.iloc[:,1]
        print("number of assigneees in "+ str(i) + ": " + str(len(assignee_list)))

        for i,assignee in enumerate(tqdm(assignee_list, ncols=60)):
            if assignee not in assignee_index_dic:
                assignee_index_dic[assignee] = 1
            else:
                assignee_index_dic[assignee] =  assignee_index_dic[assignee] + 1
        
        assignee_file_sum = assignee_file_sum.append(assignee_file)

    return assignee_index_dic, assignee_file_sum

def get_assignee(pa_path):
    assignee_name_list = []

    with open(pa_path,'r',encoding='utf-8') as f:
        items = f.readlines()
        
    for item in items:
        assignee_name = item.split('\t')[-1].strip('\n')
        assignee_name_list.append(assignee_name)

    return assignee_name_list

def search_assignee(path, assignee_index_dic, assignee_file_sum, start_year, end_year):
    assignee_name_list = get_assignee(path+"PA.txt")
    assignee_name_list = set(assignee_name_list)
    assignee_name_list = list(assignee_name_list)
    print("assignee numbers of sample patents: " + str(len(assignee_name_list)))
    
    for assignee_name in tqdm(assignee_name_list, ncols=50):
        if assignee_name in assignee_index_dic:
            targets = assignee_file_sum.loc[assignee_file_sum.iloc[:,1]==assignee_name]
            for target in targets.iloc[:, 0]:
                with open(path+"A-P-"+start_year+"-"+end_year+".txt", 'a') as f:
                    f.write(str(target))
                    f.write('\t')
                    f.write(assignee_name)
                    f.write('\n')

# 建立inventor字典
def build_inventor_dic(start_year, end_year):
    inventor_index_dic = {}
    inventor_file_sum = pd.DataFrame()

    for i in range(start_year,end_year+1):
        csv_filepath = "E:/Pythonworkspace/patent/process_data/index/inventor_file_"+str(i)+".csv"
        inventor_file = pd.read_csv(csv_filepath, header=None, encoding='utf-8')  
        application_id_list = inventor_file.iloc[:,0]
        first_name_list = inventor_file.iloc[:,1]
        last_name_list = inventor_file.iloc[:,2]
        print("number of inventor in "+ str(i) + ": " + str(len(application_id_list)))
        
        for i in tqdm(range(len(application_id_list)), ncols=60):
            inventor = str(first_name_list[i]) + "+" + str(last_name_list[i])
            if inventor not in inventor_index_dic:
                inventor_index_dic[inventor] = 1
            else:
                inventor_index_dic[inventor] =  inventor_index_dic[inventor] + 1
        
        inventor_file_sum = inventor_file_sum.append(inventor_file)

    return inventor_index_dic, inventor_file_sum

def get_inventor(path):
    inventor_name_list = []
    first_name_list = []
    last_name_list = []

    with open(path,'r',encoding='utf-8') as f:
        items = f.readlines()
        
    for item in items:
        inventor_name = item.split('\t')[-1].strip('\n')
        first_name = inventor_name.split('+')[0]
        last_name = inventor_name.split('+')[-1]
        inventor_name = first_name + "+" + last_name
        inventor_name_list.append(inventor_name)
        first_name_list.append(first_name)
        last_name_list.append(last_name)

    return inventor_name_list, first_name_list, last_name_list

def search_inventor(path, inventor_index_dic, inventor_file_sum, start_year, end_year):
    inventor_name_list, first_name_list, last_name_list = get_inventor(path+"PI+.txt")
    inventor_name_list = set(inventor_name_list)
    inventor_name_list = list(inventor_name_list)
    print("inventor number of sample patents: " + str(len(inventor_name_list)))

    for i, inventor_name in enumerate(tqdm(inventor_name_list, ncols=50)):
        if inventor_name in inventor_index_dic:
            targets = inventor_file_sum.loc[(inventor_file_sum.iloc[:,1]==first_name_list[i])&(inventor_file_sum.iloc[:,2]==last_name_list[i])]
            for target in targets.iloc[:, 0]:
                with open(path+"I-P-"+start_year+"-"+end_year+".txt", 'a') as f:
                    f.write(str(target))
                    f.write('\t')
                    f.write(inventor_name)
                    f.write('\n')

if __name__ == '__main__':
    year = "2012"
    location_path = "E:/Pythonworkspace/patent/patent_data/Application/"

    sample_num_list = [20000]
    for sample_num in sample_num_list:
        path = "E:/Pythonworkspace/patent/process_data/sample" + str(sample_num) + "/"

        # dynamic history
        start_years = ["2011", "2010", "2009"]
        end_year = "2011"

        # assignee history
        for start_year in start_years:
            print("start="+start_year+", end="+end_year)

            assignee_index_dic, assignee_file_sum = build_assignee_dic(int(start_year),int(end_year))
            print("sum number of assignee:" + str(len(assignee_index_dic)))

            search_assignee(path, assignee_index_dic, assignee_file_sum, start_year, end_year)

        # inventor history
        for start_year in start_years:
            print("start="+start_year+", end="+end_year)

            inventor_index_dic, inventor_file_sum = build_inventor_dic(int(start_year),int(end_year))
            print("sum number of inventor:" + str(len(inventor_index_dic)))

            search_inventor(path, inventor_index_dic, inventor_file_sum, start_year, end_year)