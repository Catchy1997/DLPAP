import os, re
import pandas as pd
from tqdm import tqdm

# 获取文本内容
def get_content(xml, text_filepath):
    application_id_list = []
    title_list = []
    abstract_text_list = []
    claims_content_list = []
    description_list = []
    
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
    application_id_list.append(application_id)
                
    # title
    title = re.findall(r'<invention-title id="[\w]+">(.+)</invention-title>', str(items))
    title = title[0]
    title_list.append(title)

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
    abstract_text_list.append(abstract)

    # 描述的内容
    description = re.findall(r'<description id="description">([\s\S]*?)</description>', str(items))
    p = re.compile(r'<\b[^>]*>|</\b[^>]*>|<\?\b[^>]*?>', re.S)
    description = p.sub('', description[0])
    description_list.append(description)

    # claims
    claims_content = ""
    claims_list = re.findall(r'<claim-text>([\s\S]*?)</claim-text>', str(items))
    for claim_text in claims_list:
        claims_content = claims_content + " " + str(claim_text)
    p = re.compile(r'<maths \b[^>]*>([\s\S]*?)</maths>|<\?in-line-formulae description="In-line Formulae" \b[^>]*\?>', re.S)
    claims_content = p.sub('', claims_content)
    p = re.compile(r'\'|<\b[^>]*>|</\b[^>]*>|\\n', re.S)
    claims_content = p.sub('', claims_content)
    claims_content_list.append(claims_content)

    dataframe = pd.DataFrame({'application_id':application_id_list,'title':title_list,'abstract':abstract_text_list,'claims':claims_content_list,'description':description_list})
    if os.path.exists(text_filepath):
    	dataframe.to_csv(text_filepath, header=0, mode='a', index=False, sep=',')
    else:
        dataframe.to_csv(text_filepath, mode='a', index=False, sep=',')

if __name__ == '__main__':
    location_path = "E:/Pythonworkspace/patent/patent_data/Application/"
    network_path = "E:/Pythonworkspace/patent/process_data/G-06-F-17/network/"
    text_path = "E:/Pythonworkspace/patent/process_data/G-06-F-17/text/"

    name_list = ["2006", "2007", "2008", "2009", "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018"]
    for name in name_list:
        print("years: " + name)
        excel_path = "E:/Pythonworkspace/patent/process_data/G-06-F-17/class/"+ name +".xlsx"
        text_filepath = text_path + name + ".csv"
        
        data = pd.read_excel(excel_path, encoding='utf-8')
        location_list = data['location'].tolist()
        for i,location in enumerate(tqdm(location_list, ncols=60)):
            xml = location_path + location            
            # 得到文本
            get_content(xml, text_filepath)