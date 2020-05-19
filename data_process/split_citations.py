import re, os
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
	file = "E:/Pythonworkspace/patent/patent_data/PatentsView/usapplicationcitation.tsv/usapplicationcitation.tsv"

	index = 0
	with open(file, 'r', encoding='utf-8') as f:
		for line in tqdm(f, ncols=60):
			index = index + 1
			if index > 20065000:
				date = line.split('\t')[3]
				year = re.findall(r"([\d]{4})-[\d]{2}-[\d]{2}", date)
				if year:
					patent_id = line.split('\t')[1]
					number = line.split('\t')[-4]
					country = line.split('\t')[-3]
					category = line.split('\t')[-2]

					filepath = "E:/Pythonworkspace/patent/process_data/citations_app/" + year[0] + ".csv"
						
					dataframe = pd.DataFrame({'patent_id':[patent_id],'application_id':[number],'date':[date],'category':[category],'country':[country]})
					if os.path.exists(filepath):
						dataframe.to_csv(filepath, header=0, mode='a', index=False, sep=',')
					else:
						dataframe.to_csv(filepath, mode='a', index=False, sep=',')