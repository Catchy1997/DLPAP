import pandas as pd
import os, re
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

if __name__ == '__main__':
    name_list = ["2007", "2008", "2009" ,"2010"]
    for name in name_list:
        print("year: " + name)
        location_id_dic = {}

        filepath = "E:/Pythonworkspace/patent/patent_data/Application/" + name + "/"
        patent_file_sum = sum_patent(name, filepath)
        for index in tqdm(range(0, len(patent_file_sum)), ncols=70):
            location_id = re.findall(r'US([\d]{11})', patent_file_sum.iloc[index]['location'])
            location_id = location_id[0]
            application_id = patent_file_sum.iloc[index]['application_id']
            if not location_id in location_id_dic:
                location_id_dic[location_id] = application_id

        citation_file = "E:/Pythonworkspace/patent/process_data/citations_app/" + name + ".csv"
        citations = pd.read_csv(citation_file, encoding='utf-8')
        save_file = "E:/Pythonworkspace/patent/process_data/citations_app/" + name + "_appID.csv"
        for index in tqdm(range(0, len(citations)), ncols=70):
            location = citations['application_id'].iloc[index]
            if location in location_id_dic.keys():
                application_id = location_id_dic[location]
                category = citations['category'].iloc[index]
                if len(re.findall(r"applicant", category)) > 0:
                    patent_id = citations['patent_id'].iloc[index]
                    dataframe = pd.DataFrame({'application_id':[application_id],'category':[category],'patent_id':[patent_id]})
                    if os.path.exists(save_file):
                        dataframe.to_csv(save_file, header=0, mode='a', index=False, sep=',')
                    else:
                        dataframe.to_csv(save_file, mode='a', index=False, sep=',')