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

def sum_patent(year, filepath):
    path_list = []    
    path_list = csv_dir(filepath, path_list)
    print(year + " - csv文件数量：" + str(len(path_list)))

    patent_file_sum = pd.DataFrame()
    for path in path_list:
        patent_file = pd.read_csv(path, encoding='utf-8')
        patent_file_sum = patent_file_sum.append(patent_file)
    print(year + " - 涉及专利数量：" + str(len(patent_file_sum)))

    return patent_file_sum

if __name__ == '__main__':
	year = "2012"
	location_path = "E:/Pythonworkspace/patent/patent_data/Application/" + year + "/"
	patent_file_sum = sum_patent(year, location_path)

	sample_num_list = [5000, 10000, 20000]
	save_path = "E:/Pythonworkspace/patent/process_data/"
	for sample_num in sample_num_list:
		print("sample number: " + str(sample_num))

		result_1 = patent_file_sum[patent_file_sum['result'] == 1]
		result_0 = patent_file_sum[patent_file_sum['result'] == 0]

		df1 = result_1.sample(int(sample_num)//2)
		df0 = result_0.sample(int(sample_num)//2)
		df = pd.concat([df1, df0])

		filepath = save_path + "sample" + str(sample_num) + "/sample.xlsx"
		df.to_excel(filepath)