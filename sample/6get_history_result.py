import os, re
import pandas as pd
from tqdm import tqdm

def excel_dir(filepath, path_list):
    for i in os.listdir(filepath):
        path = os.path.join(filepath, i)
        if os.path.isdir(path):
            excel_dir(path, path_list)
        if path.endswith(".xlsx"):
        	if len(re.findall(r'match', path)) > 0:
        		path_list.append(path)
        if path.endswith(".csv"):
        	if len(re.findall(r'match', path)) > 0:
        		path_list.append(path)
    
    return path_list

def get_label(location_path, start_year, end_year, network_path):
    path_list = []
    for year in tqdm(range(int(start_year), int(end_year)+1), ncols=50):
        filepath = location_path + str(year)
        path_list = excel_dir(filepath, path_list)

    patent_file_sum = pd.DataFrame()
    for path in tqdm(path_list, ncols=50):
        if path.endswith(".xlsx"):
            patent_file = pd.read_excel(path, encoding='utf-8')
        if path.endswith(".csv"):
            patent_file = pd.read_csv(path, encoding='utf-8')
        patent_file_sum = patent_file_sum.append(patent_file)

    # inventor
    ip_filepath = network_path + "network/" + start_year + "-" + end_year + "/I-P-" + start_year +"-" + end_year + ".xlsx"
    label_file = network_path + "network/" + start_year + "-" + end_year + "/I-P-label.csv"
    assignee_data = pd.read_excel(ip_filepath, encoding='utf-8')
    patent_list = assignee_data['application_id']
    patent_list = set(patent_list)
    
    patent_frame = pd.DataFrame()
    for patent in tqdm(patent_list, ncols=50):
        patent_line = patent_file_sum[patent_file_sum['application_id'] == patent]
        patent_line = patent_line[['application_id','result']]
        if os.path.exists(label_file):
            patent_line.to_csv(label_file, header=0, mode='a', index=False, sep=',')
        else:
            patent_line.to_csv(label_file, mode='a', index=False, sep=',')

    # assignee
    ap_filepath = network_path + "network/" + start_year + "-" + end_year + "/A-P-" + start_year +"-" + end_year + ".xlsx"
    label_file = network_path + "network/" + start_year + "-" + end_year + "/A-P-label.csv"
    assignee_data = pd.read_excel(ap_filepath, encoding='utf-8')
    patent_list = assignee_data['application_id']
    patent_list = set(patent_list)
    
    patent_frame = pd.DataFrame()
    for patent in tqdm(patent_list, ncols=50):
        patent_line = patent_file_sum[patent_file_sum['application_id'] == patent]
        patent_line = patent_line[['application_id','result']]
        if os.path.exists(label_file):
            patent_line.to_csv(label_file, header=0, mode='a', index=False, sep=',')
        else:
            patent_line.to_csv(label_file, mode='a', index=False, sep=',')

if __name__ == '__main__':
    year = "2012"
    location_path = "E:/Pythonworkspace/patent/patent_data/Application/"

    sample_num_list = [20000]
    for sample_num in sample_num_list:
        network_path = "E:/Pythonworkspace/patent/process_data/sample" + str(sample_num) + "/"

        # dynamic history
        start_years = ["2009", "2010", "2011"]
        end_year = "2011"

        for start_year in start_years:
            print("start="+start_year+", end="+end_year)
            get_label(location_path, start_year, end_year, network_path)