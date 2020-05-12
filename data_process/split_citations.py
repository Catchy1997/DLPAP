import re, os
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    file = "E:/Pythonworkspace/patent/patent_data/PatentsView/uspatentcitation.tsv/uspatentcitation.tsv"

    with open(file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, ncols=50):
            uuid = line.split()[0]
            patent_id = line.split()[1]
            citation_id = line.split()[2]
            date = line.split()[3]
            name = line.split()[4]
            kind = line.split()[5]
            country = line.split()[6]
            category = line.split()[7]
            sequence = line.split()[8]
            
            year = re.findall(r"([\d]{4})-[\d]{2}-[\d]{2}", date)        
            if year:
                filepath = "E:/Pythonworkspace/patent/process_data/citations/" + year[0] + ".csv"
                
                dataframe = pd.DataFrame({'patent_id':[patent_id],'citation_id':[citation_id],'date':[date],'category':[category],'country    ':[country    ]})
                if os.path.exists(filepath):
                    dataframe.to_csv(filepath, header=0, mode='a', index=False, sep=',')
                else:
                    dataframe.to_csv(filepath, mode='a', index=False, sep=',')