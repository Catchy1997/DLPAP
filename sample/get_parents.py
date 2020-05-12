import os, re
import pandas as pd
from tqdm import tqdm

def get_parent(path, parent_file):
    application_id_list = []
    cited_id_list = []
    doc_date_list = []
    doc_status_list = []

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
        application_id_list.append(application_id)
        cited_id_list.append(doc_number[0])
        doc_date_list.append(doc_date[0])
        doc_status_list.append(doc_status[0])
        
    dataframe = pd.DataFrame({'application_id':application_id_list,'cited_id':cited_id_list,'date':doc_date_list,'status':doc_status_list})
    if os.path.exists(parent_file):
        dataframe.to_csv(parent_file, header=0, mode='a', index=False, sep=',')
    else:
        dataframe.to_csv(parent_file, mode='a', index=False, sep=',')

if __name__ == '__main__':
    location_path = "E:/Pythonworkspace/patent/patent_data/Application/"

    name_list = ["2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018"]
    for name in name_list:
        print("years: " + name)
        excel_path = "E:/Pythonworkspace/patent/process_data/G-06-F-17/class/"+ name +".xlsx"
        parent_file = "E:/Pythonworkspace/patent/process_data/G-06-F-17/network/citation/" + name +".csv"

        data = pd.read_excel(excel_path, encoding='utf-8')
        location_list = data['location'].tolist()
        for i,location in enumerate(tqdm(location_list, ncols=60)):
            xml = location_path + location            
            # add parent
            get_parent(xml, parent_file)