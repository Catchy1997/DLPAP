import os, re
import pandas as pd
from tqdm import tqdm

# 生成节点的网络特征
def get_network_data(i, path, network_path):
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
        inventor_ = str(first_name[i]) + "+" + str(last_name[i])
        # 生成PI.txt
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
        # 生成P-P.txt 
        with open(network_path+"P-P.txt", 'a') as f:
            f.write(application_id)
            f.write("\t")
            f.write(str(doc_number[0]))
            f.write("\t")
            f.write(doc_date[0])
            f.write("\t")
            f.write(doc_status[0])
            f.write("\n")

if __name__ == '__main__':
    year = "2012"
    location_path = "E:/Pythonworkspace/patent/patent_data/Application/"

    sample_num_list = [10000, 20000]
    for sample_num in sample_num_list:
        save_path = "E:/Pythonworkspace/patent/process_data/sample" + str(sample_num) + "/"
        
        excel_path = "E:/Pythonworkspace/patent/process_data/sample" + str(sample_num) + "/sample.xlsx"
        data = pd.read_excel(excel_path, encoding='utf-8')
        location_list = data['location'].tolist()
        for i,location in enumerate(tqdm(location_list, ncols=60)):
            xml = location_path + location            
            # 得到节点网络
            get_network_data(i, xml, save_path)