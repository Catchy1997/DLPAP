import os, re
import pandas as pd
from tqdm import tqdm

# 生成节点的网络特征
def get_patent_data(i, path, network_path):
    content = []
    with open(path,'r',encoding='utf-8') as f:
        items = f.readlines()

    appl_content = ""
    for item in items:
        if re.findall(r'<application-reference appl-type="utility">', item):
            appl_content = ""
        appl_content = appl_content + item
        if re.findall(r'</application-reference>', item):
            break
    appl_no = re.findall(r'<doc-number>([\d]{8})</doc-number>', appl_content)
    application_id = appl_no[0]

    # 生成patent_dict.txt
    with open(network_path+"patent_dict.txt", 'a') as f:
        f.write(str(i))
        f.write("\t")
        f.write(application_id)
        f.write("\n")
        f.close()
        
    # inventor
    invention_content = ""
    for item in items:
        if re.findall(r'<applicants>', item):
            invention_content = ""
        invention_content = invention_content + item
        if re.findall(r'</applicants>', item):
            break        
    first_name = re.findall(r'<first-name>(.+)</first-name>', invention_content)
    last_name = re.findall(r'<last-name>(.+)</last-name>', invention_content)
    if len(first_name) != len(last_name):
        if len(first_name) < len(last_name):
            last_name = last_name[0:len(first_name)]
        if len(first_name) > len(last_name):
            first_name = first_name[0:len(last_name)]
    for i in range(0, len(last_name)):
        inventor = str(first_name[i]) + "-" + str(last_name[i])
        # 生成PI-.txt
        with open(network_path+"PI-.txt", 'a') as f:
            f.write(application_id)
            f.write("\t")
            f.write(inventor)
            f.write("\n")
            f.close()

        inventor_ = str(first_name[i]) + "+" + str(last_name[i])
        # 生成PI+.txt
        with open(network_path+"PI+.txt", 'a') as f:
            f.write(application_id)
            f.write("\t")
            f.write(inventor_)
            f.write("\n")
            f.close()
        
    # assignee
    assignee_content = ""
    assignee_list = []
    for item in items:
        if re.findall(r'<assignee>', item):
            assignee_content = ""
        assignee_content = assignee_content + item
        if re.findall(r'</assignee>', item):
            assignee_list.append(assignee_content)
    if len(assignee_list) > 0:
        for assignee in assignee_list:
            assignee = re.findall(r'<orgname>(.+)</orgname>', assignee)
            if len(assignee) > 0:
                # 生成PA.txt
                with open(network_path+"PA.txt", 'a') as f:
                    f.write(application_id)
                    f.write("\t")
                    f.write(assignee[0])
                    f.write("\n")
                    f.close()
    else:
        address_content = ""
        for item in items:
            if re.findall(r'<correspondence-address>', item):
                address_content = ""
            address_content = address_content + item
            if re.findall(r'</correspondence-address>', item):
                break
        assignee_list = re.findall(r'<name>(.+)</name>', str(items))
        for assignee in assignee_list:
            # 生成PA.txt
            with open(network_path+"PA.txt", 'a') as f:
                f.write(application_id)
                f.write("\t")
                f.write(assignee)
                f.write("\n")
                f.close()
        
    # 记录parent专利
    parent_content = ""
    parent_list = []
    for item in items:
        if re.findall(r'<parent-doc>', item):
            parent_content = ""
        parent_content = parent_content + item
        if re.findall(r'</parent-doc>', item):
            parent_list.append(parent_content)
    for parent in parent_list:
        if re.findall(r'<doc-number>(.+)</doc-number>', parent):
            doc_number = re.findall(r'<doc-number>(.+)</doc-number>', parent)
        else:
            doc_number = "NULL"
        if re.findall(r'<date>(.+)</date>', parent):
            doc_date = re.findall(r'<date>(.+)</date>', parent)
        else:
            doc_date = "NULL"
        if re.findall(r'<parent-status>(.+)</parent-status>', parent):
            doc_status = re.findall(r'<parent-status>(.+)</parent-status>', parent)
        else:
            doc_status = "NULL"
        # 生成citation 
        with open(network_path+"citations.txt", 'a') as f:
            f.write(application_id)
            f.write("\t")
            f.write(str(doc_number[0]))
            f.write("\t")
            f.write(doc_date[0])
            f.write("\t")
            f.write(doc_status[0])
            f.write("\n")

# 建立assignee字典
def build_assignee_dic():
    assignee_index_dic = {}
    assignee_file_sum = pd.DataFrame()

    for i in range(2007,2010):
        csv_filepath = "E:/Pythonworkspace/patent/process_data/index/assignee_file_"+str(i)+".csv"
        assignee_file = pd.read_csv(csv_filepath, header=None, encoding='utf-8')
        application_id_list = assignee_file.iloc[:,0]
        assignee_list = assignee_file.iloc[:,1]
        print("number of assigneees in "+ str(i) + ": " + str(len(assignee_list)))

        for i,assignee in enumerate(assignee_list):
            if assignee not in assignee_index_dic:
                assignee_index_dic[assignee] = 1
            else:
                assignee_index_dic[assignee] =  assignee_index_dic[assignee] + 1
        
        assignee_file_sum = assignee_file_sum.append(assignee_file)

    return assignee_index_dic, assignee_file_sum

def get_assignee(path):
    assignee_name_list = []

    with open(path,'r',encoding='utf-8') as f:
        items = f.readlines()
        
    for item in items:
        assignee_name = item.split('\t')[-1].strip('\n')
        assignee_name_list.append(assignee_name)

    return assignee_name_list

# 建立inventor字典
def build_inventor_dic():
    inventor_index_dic = {}
    inventor_file_sum = pd.DataFrame()

    for i in range(2007,2010):
        csv_filepath = "E:/Pythonworkspace/patent/process_data/index/inventor_file_"+str(i)+".csv"
        inventor_file = pd.read_csv(csv_filepath, header=None, encoding='utf-8')    
        application_id_list = inventor_file.iloc[:,0]
        first_name_list = inventor_file.iloc[:,1]
        last_name_list = inventor_file.iloc[:,2]
        print("number of inventor in "+ str(i) + ": " + str(len(application_id_list)))
        
        for i in range(0,len(application_id_list)):
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

def search(path, inventor_index_dic, inventor_file_sum, assignee_index_dic=0, assignee_file_sum=0):
    inventor_name_list, first_name_list, last_name_list = get_inventor(path+"PI+.txt")
    inventor_name_list = set(inventor_name_list)
    print("inventor number: " + str(len(inventor_name_list)))

    for i, inventor_name in enumerate(inventor_name_list):
        if inventor_name in inventor_index_dic:
            targets = inventor_file_sum.loc[(inventor_file_sum.iloc[:,1]==first_name_list[i])&(inventor_file_sum.iloc[:,2]==last_name_list[i])]
            for target in targets.iloc[:, 0]:
                with open(path+"I-GP.txt", 'a') as f:
                    f.write(str(target))
                    f.write('\t')
                    f.write(inventor_name)
                    f.write('\n')
        print("\r", "---- 处理到第" + str(i) + "个inventor", end="", flush=True)

    # assignee_name_list = get_assignee(path+"PA.txt")
    # assignee_name_list = set(assignee_name_list)
    # print("assignee number: " + str(len(assignee_name_list)))
    
    # for j,assignee_name in enumerate(assignee_name_list):
    #     if assignee_name in assignee_index_dic:
    #         targets_1 = assignee_file_sum.loc[assignee_file_sum.iloc[:,1]==assignee_name]
    #         for target in targets_1.iloc[:, 0]:
    #             with open(path+"A-GP.txt", 'a') as f:
    #                 f.write(str(target))
    #                 f.write('\t')
    #                 f.write(assignee_name)
    #                 f.write('\n')
    #     print("\r", "---- 处理到第" + str(j) + "个assignee", end="", flush=True)

def excel_dir(filepath, path_list):
    for i in os.listdir(filepath):
        path = os.path.join(filepath, i)
        if os.path.isdir(path):
            excel_dir(path, path_list)
        if path.endswith(".xlsx"):
            path_list.append(path)
    
    return path_list

def get_granted(network_path):
    path_list = []
    print("read all excel....")
    filepath = "E:/Pythonworkspace/patent/patent_data/暂时不用/corpus"
    path_list = excel_dir(filepath, path_list)
    print("finish!!!")

    patent_file_sum = pd.DataFrame()
    for path in tqdm(path_list, ncols=30):
        patent_file = pd.read_excel(path, encoding='utf-8')
        patent_file_sum = patent_file_sum.append(patent_file)

    inventor_data = pd.read_excel(network_path+"I-GP.xlsx", encoding='utf-8')
    patent_list = inventor_data['application_id']
    patent_list = set(patent_list)
    patent_frame = pd.DataFrame()
    for patent in tqdm(patent_list, ncols=30):
        patent_line = patent_file_sum[patent_file_sum['application_id'] == patent]
        patent_line = patent_line[['application_id','result']]
        if os.path.exists(network_path+"I-GP-granted.csv"):
            patent_line.to_csv(network_path+"I-GP-granted.csv", header=0, mode='a', index=False, sep=',')
        else:
            patent_line.to_csv(network_path+"I-GP-granted.csv", mode='a', index=False, sep=',')

    # assignee_data = pd.read_excel(network_path+"A-GP.xlsx", encoding='utf-8')
    # patent_list = assignee_data['application_id']
    # patent_list = set(patent_list)
    # patent_frame = pd.DataFrame()
    # for patent in tqdm(patent_list, ncols=30):
    #     patent_line = patent_file_sum[patent_file_sum['application_id'] == patent]
    #     patent_line = patent_line[['application_id','result']]
    #     if os.path.exists(network_path+"A-GP-granted.csv"):
    #         patent_line.to_csv(network_path+"A-GP-granted.csv", header=0, mode='a', index=False, sep=',')
    #     else:
    #         patent_line.to_csv(network_path+"A-GP-granted.csv", mode='a', index=False, sep=',')

if __name__ == '__main__':
    # location_path = "E:/Pythonworkspace/patent/patent_data/Application/"
    # network_path = "E:/Pythonworkspace/patent/process_data/G-06-F-17/network/"

    # name_list = ["2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018"]
    # for name in name_list:
    #     print("years: " + name)
    #     excel_path = "E:/Pythonworkspace/patent/process_data/G-06-F-17/class/"+ name +".xlsx"
    #     text_filepath = text_path + name + ".csv"
        
    #     data = pd.read_excel(excel_path, encoding='utf-8')
    #     location_list = data['location'].tolist()
    #     for i,location in enumerate(tqdm(location_list, ncols=60)):
    #         xml = location_path + location            
    #         # 得到节点网络
    #         get_patent_data(i, xml, network_path)
    
    # 3.查询历史专利
    # path = "E:/Pythonworkspace/patent/process_data/sample3_G-06-F-17/network/"

    # assignee_index_dic, assignee_file_sum = build_assignee_dic()
    # print("sum number of assignee:" + str(len(assignee_index_dic)))

    # inventor_index_dic, inventor_file_sum = build_inventor_dic()
    # print("sum number of inventor:" + str(len(inventor_index_dic)))
    
    # search(path, inventor_index_dic, inventor_file_sum)
    # search(path, inventor_index_dic, inventor_file_sum, assignee_index_dic, assignee_file_sum)
    
    # 4.获取历史专利的授权结果
    # get_granted(path)