import re
import datetime
import os
import pandas as pd
from tqdm import tqdm

def print_dir(filepath):
    path_list = []
    for i in os.listdir(filepath):
        path = os.path.join(filepath, i)
        if os.path.isdir(path):
            print_dir(path)
        if path.endswith(".xml"):
            path_list.append(path)

    return path_list

def get_inventor_and_assignee(path_list, inventor_file, assignee_file):
    for path in tqdm(path_list, ncols=50):
        with open(path, 'r', encoding='utf-8') as f:
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
        application_id_list = []

        if len(first_name) != len(last_name):
            if len(first_name) < len(last_name):
                last_name = last_name[0:len(first_name)]
                for i in range(0, len(first_name)):
                    application_id_list.append(application_id)
            if len(first_name) > len(last_name):
                first_name = first_name[0:len(last_name)]
                for i in range(0, len(last_name)):
                    application_id_list.append(application_id)
        else:
            for i in range(0, len(first_name)):
                application_id_list.append(application_id)

        dataframe = pd.DataFrame({'application_id':application_id_list,'first_name':first_name,'last_name':last_name})
        # if count == 0:
        #     dataframe.to_csv(inventor_file, mode='a', index=False, sep=',')
        # else:
        #     dataframe.to_csv(inventor_file, header=0, mode='a', index=False, sep=',')
        dataframe.to_csv(inventor_file, header=0, mode='a', index=False, sep=',')

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
                    pass
                else:
                    address_content = ""
                    for item in items:
                        if re.findall(r'<correspondence-address>', item):
                            address_content = ""
                        address_content = address_content + item
                        if re.findall(r'</correspondence-address>', item):
                            break
                    assignee = re.findall(r'<name>(.+)</name>', str(items))
        else:
            address_content = ""
            for item in items:
                if re.findall(r'<correspondence-address>', item):
                    address_content = ""
                address_content = address_content + item
                if re.findall(r'</correspondence-address>', item):
                    break
            assignee = re.findall(r'<name>(.+)</name>', str(items))

        if len(assignee) > 0:
            dataframe = pd.DataFrame({'application_id':appl_no,'assignee':assignee})
            dataframe.to_csv(assignee_file, header=0, mode='a', index=False, sep=',')
        # if count == 0:
        #     dataframe.to_csv(assignee_file, mode='a', index=False, sep=',')
        # else:
        #     dataframe.to_csv(assignee_file, header=0, mode='a', index=False, sep=',')

if __name__ == '__main__':
    abs_filepath = "E:/Pythonworkspace/patent/"

    name_list = ["2018"]
    for name in name_list:
        print("years: " + name)
        filepath = abs_filepath + "patent_data/Application/" + name + "/"
        count = 0
        for i in os.listdir(filepath):
            count = count + 1
            if count == -1:
                print(str(i))
            else:
                path = os.path.join(filepath, i)

                path_list = print_dir(path)
                inventor_file = abs_filepath + "process_data/index/inventor_file_" + name + ".csv"
                assignee_file = abs_filepath + "process_data/index/assignee_file_" + name + ".csv"
                get_inventor_and_assignee(path_list, inventor_file, assignee_file)