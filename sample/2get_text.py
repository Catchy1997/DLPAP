import os, re
import pandas as pd
from tqdm import tqdm

# 获取文本内容
def get_content(location, xml, text_filepath):
    with open(xml,'r',encoding='utf-8') as f:
        items = f.readlines()

    # application_id
    appl_content = ""
    for item in items:
        if re.findall(r'<application-reference appl-type="utility">', item):
            appl_content = ""
        appl_content = appl_content + item
        if re.findall(r'</application-reference>', item):
            break
    appl_no = re.findall(r'<doc-number>([\d]{8})</doc-number>', appl_content)
    application_id = appl_no[0]
                
    # title
    title = re.findall(r'<invention-title id="[\w]+">(.+)</invention-title>', str(items))
    title = title[0]

    # abstract
    abstract = re.findall(r'<abstract id="abstract">([\s\S]*?)</abstract>', str(items))
    abstract_list = re.findall(r'<p\b[^>]*>([\s\S]*?)</p>', str(abstract))
    if len(abstract_list) == 1:
        abstract = abstract_list[0]
    elif len(abstract_list) == 0:
        abstract = ""
    else:
        abstract = ""
        for abst in abstract_list:
            abstract = abstract + " "+ abst
    p = re.compile(r'<\b[^>]*>|</\b[^>]*>', re.S)
    abstract = p.sub('', abstract)
    p = re.compile(r',', re.S)
    abstract = p.sub(';', abstract)

    # # 描述的内容
    # description = re.findall(r'<description id="description">([\s\S]*?)</description>', str(items))
    # p = re.compile(r'<\b[^>]*>|</\b[^>]*>|<\?\b[^>]*?>', re.S)
    # description = p.sub('', description[0])
    # description_list.append(description)

    # claims
    claims_content = ""
    claims_list = re.findall(r'<claim-text>([\s\S]*?)</claim-text>', str(items))
    for claim_text in claims_list:
        claims_content = claims_content + " " + str(claim_text)
    p = re.compile(r'<maths \b[^>]*>([\s\S]*?)</maths>|<\?in-line-formulae description="In-line Formulae" \b[^>]*\?>', re.S)
    claims_content = p.sub('', claims_content)
    p = re.compile(r'\'|<\b[^>]*>|</\b[^>]*>|\n|\\n', re.S)
    claims_content = p.sub('', claims_content)
    p = re.compile(r',', re.S)
    claims_content = p.sub(';', claims_content)

    dataframe = pd.DataFrame({'location':[location],'application_id':[application_id],'title':[title],'abstract':[abstract],'claims':[claims_content]})
    if os.path.exists(text_filepath):
        dataframe.to_csv(text_filepath, header=0, mode='a', index=False, sep=',')
    else:
        dataframe.to_csv(text_filepath, mode='a', index=False, sep=',')

if __name__ == '__main__':
    year = "2012"
    location_path = "E:/Pythonworkspace/patent/patent_data/Application/"

    sample_num_list = [5000, 10000, 20000]
    for sample_num in sample_num_list:
        save_path = "E:/Pythonworkspace/patent/process_data/sample" + str(sample_num) + "/text.csv"

        excel_path = "E:/Pythonworkspace/patent/process_data/sample" + str(sample_num) + "/sample.xlsx"
        data = pd.read_excel(excel_path, encoding='utf-8')
        location_list = data['location'].tolist()
        for i,location in enumerate(tqdm(location_list, ncols=50)):
            xml = location_path + location       
            # 得到文本
            get_content(location, xml, save_path)