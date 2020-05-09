import os, re
import pandas as pd
import zipfile
from tqdm import tqdm

def unzip(path, zfile):
    file_path = path + os.sep + zfile
    desdir = path + os.sep + zfile[:zfile.index('.zip')]
    srcfile = zipfile.ZipFile(file_path)
    for filename in srcfile.namelist():
        srcfile.extract(filename, desdir)
        if filename.endswith('.zip'):
            # if zipfile.is_zipfile(filename):
            path = desdir
            zfile = filename
            unzip(path, zfile)

def zip_dir(zip_filepath):
    for i in os.listdir(zip_filepath):
        path = os.path.join(zip_filepath, i)
        if os.path.isdir(path):
            zip_dir(path)
        if path.endswith(".zip"):
            unzip(zip_filepath, path.split('/')[-1])

def split_xml(xml_path):
    path_list = []
    for i in os.listdir(xml_path):
        path = os.path.join(xml_path, i)
        if os.path.isdir(path):
            split_xml(path)
        if path.endswith(".xml"):
            try:
                with open(path, 'r') as f:
                    items = f.readlines()

                copus = []
                line = 0
                for item in tqdm(items, ncols=50):
                    if re.findall(r'<\?xml version="1\.0" encoding="UTF-8"\?>', item):
                        if line != 0:
                            copus.append(item_content)
                        item_content = ""
                    item_content = item_content + item
                    line = line + 1
                copus.append(item_content)

                for item in copus:              
                    document_identifier = re.findall(r'file="[\w]+-[\w]+.XML"', item)
                    file1 = path.split('\\')[0]
                    if document_identifier:
                        document_identifier = re.findall(r'US[\w]+', document_identifier[0])
                        filename = file1 + "/" + document_identifier[0] + ".xml"
                        try:
                            with open(filename, 'a') as f:
                                f.write(item)
                                f.write("\n")
                        except:
                            print(filename)
                    else:
                        document_identifier = re.findall(r'US[\w]+-[\w]+-[\w]+.TIF', item)
                        if document_identifier:
                            document_identifier = re.findall(r'US[\w]+', document_identifier[0])
                            filename = file1 + "/" + document_identifier[0] + ".xml"
                            try:
                                with open(filename, 'a') as f:
                                    f.write(item)
                                    f.write("\n")
                            except:
                                print(filename)
            except:
                pass

def filter(xml_path):
    for i in os.listdir(xml_path):
        path = os.path.join(xml_path, i)
        if os.path.isdir(path):
            filter(path)
        if path.endswith(".xml"):
            with open(path, 'r', encoding='utf-8') as f:
                items = f.readlines()

            filename = path.split('/')[-2]+"/"+path.split('/')[-1].split('\\')[0]+"/"+path.split('/')[-1].split('\\')[-1]

            copus = []
            item_content = ""
            for item in items:
                if re.findall(r'<application-reference appl-type="utility">', item):
                    item_content = ""
                item_content = item_content + item
                if re.findall(r'</application-reference>', item):
                    break
            copus.append(item_content)

            appl_no1 = re.findall(r'<doc-number>[\d]{8}</doc-number>', str(copus))
            result_file = path.split('\\')[0]

            if len(appl_no1) > 1:
                with open(result_file+"/fail_to_get_appNo.txt", 'a') as f:
                    f.write("!!!!!!!!!!error:have too much app_no!")
                    f.write("\t")
                    f.write(filename)
                    f.write("\n")
            appl_no2 = re.findall(r'[\d]{8}', appl_no1[0])

            application_id = appl_no2[0]

            if application_id:
                with open(result_file+"/results.txt", 'a') as f:
                    f.write(filename)
                    f.write("\t")
                    f.write(application_id)
                    f.write("\n")
            else:
                with open(result_file+"/fail_to_get_appNo.txt", 'a') as f:
                    f.write(filename)
                    f.write("\n")

def print_txt(year, xml_path, app_no):
    txt_list = []
    for i in os.listdir(xml_path):
        path = os.path.join(xml_path, i)
        if os.path.isdir(path):
            print_txt(year, path, app_no)
        if path.endswith(".txt"):
            if path.split('\\')[-1][0] == "r":
                txt_list.append(path)

    for txt in txt_list:
        match(year, txt, app_no)

def match(year, txt, app_no):
    result_txt = txt.split('\\')[0]+"/"
    with open(txt, 'r') as f1:
        items = f1.readlines()
    
    for item in items:
        filename = item.split('\t')[0]
        app_id = item.split('\t')[1]
        p = re.compile(r'\n', re.S)
        app_id = p.sub('',app_id)
            
        if int(app_id) in app_no:
            result = 1
            with open(result_txt+"match_final.txt", 'a') as f:
                f.write(filename)
                f.write("\t")
                f.write(app_id)
                f.write("\t")
                f.write(str(result))
                f.write("\t")
                f.write(str(year))
                f.write("\n")
        else:
            result = 0

def no_match(filepath):
    txt_list = []
    for i in os.listdir(filepath):
        path = os.path.join(filepath, i)
        if os.path.isdir(path):
            no_match(path)
        if path.endswith(".txt"):
            if path.split('\\')[-1][0] == "r":
                txt_list.append(path)

    for txt in txt_list:
        p = re.compile(r"results", re.S)
        match_file = p.sub("match_final", txt)
        if os.path.exists(match_file):
            with open(match_file, 'r') as f:
                results = f.readlines()
            
            application_id_list = []
            for result in results:
                application_id = result.split('\t')[1]
                application_id_list.append(application_id)
            application_id_set = set(application_id_list)

            with open(txt, 'r') as f1:
                items = f1.readlines()
            
            for item in items:
                p1 = re.compile(r'\n', re.S)
                item = p1.sub('', item)

                appl_id = item.split('\t')[-1]
                
                if appl_id not in application_id_set:
                    result = 0
                    with open(match_file, 'a') as f:
                        f.write(item.split('\t')[0])
                        f.write("\t")
                        f.write(appl_id)
                        f.write("\t")
                        f.write(str(result))
                        f.write("\n")
        else:
            print(match_file)

def get_class(path):
    with open(path,'r',encoding='utf-8') as f:
        items = f.readlines()
        
    class_content = ""
    for item in items:
        if re.findall(r'<classification-ipcr>', item):
            class_content = ""
        class_content = class_content + item
        if re.findall(r'</classification-ipcr>', item):
            break
    section = re.findall(r'<section>([\w]+)</section>', class_content)
    if section:
        pass
    else:
        section = "0"
    class_ = re.findall(r'<class>([\d]+)</class>', class_content)
    if class_:
        pass
    else:
        class_ = "0"
    subclass = re.findall(r'<subclass>([\w]+)</subclass>', class_content)
    if subclass:
        pass
    else:
        subclass = "0"
    group = re.findall(r'<main-group>([\d]+)</main-group>', class_content)
    if group:
        pass
    else:
        group = "0"
    cpc_class = section[0] + "-" + class_[0] + "-" + subclass[0] + "-" + group[0]

    return cpc_class

def info_to_excel(filepath):
    for i in os.listdir(filepath):
        path = os.path.join(filepath, i)
        if os.path.isdir(path):
            info_to_excel(path)
        if path.endswith(".txt"):
            if path.split('\\')[-1] == "match_final.txt":
                with open(path, 'r') as f:
                    lines = f.readlines()

                file = path.split('\\')[0]
                location_list = []
                application_id_list = []
                result_list = []
                cpc_class_list = []

                for line in tqdm(lines, ncols=40):
                    location = line.split('\t')[0]
                    location_list.append(location)
                    application_id = line.split('\t')[1]
                    application_id_list.append(application_id)
                    result = line.split('\t')[-1].strip('\n')
                    if len(result) == 4:
                        result = 1
                    result_list.append(result)

                    cpc_class = get_class(file+"/"+location.split('/')[-1])
                    cpc_class_list.append(cpc_class)

                df = pd.DataFrame({'application_id':application_id_list,'location':location_list,'cpc_class':cpc_class_list,'result':result_list})
                if os.path.exists(file+"/match_final.csv"):
                    df.to_csv(file+"/match_final.csv", header=0, mode='a', index=False, sep=',')
                else:
                    df.to_csv(file+"/match_final.csv", mode='a', index=False, sep=',')

if __name__ == '__main__':
	abs_filepath = "E:/Pythonworkspace/patent/patent_data/"

    # 解压application的zip文件
    name_list = ["2001", "2002", "2003", "2004", "2005", "2006", "2011", "2012", "2013", "2014"]
    for name in tqdm(name_list, ncols=70):
        zip_filepath = abs_filepath + "ApplicationZip/" + name + "/"
        zip_dir(zip_filepath)

    # 每个patent拆分成一个xml
    name_list = ["2018"]
    for name in tqdm(name_list, ncols=70):
        zip_filepath = abs_filepath + "ApplicationZip/" + name + "/"
        split_xml(zip_filepath)

    # 条件过滤：
    # - appl-type="utility"
    # - number of Application-ID = 1
    name_list = ["2012"]
    for name in tqdm(name_list, ncols=70):
        xml_path = abs_filepath + "ApplicationZip/" + name + "/"
        filter(xml_path)
    
    # match result=1
    name_list = ["2013","2014", "2015", "2016", "2017", "2018"]
    for name in name_list:
        print("years: " + name)
        xml_path = abs_filepath + "ApplicationZip/" + name + "/"
        for i in tqdm(range(2001,2020), ncols=70):
            filename = abs_filepath + "/GrantedSum/" + str(i) + '.xlsx'
            data = pd.read_excel(filename, encoding='utf-8')
            app_no = data.iloc[:,3].tolist()
            app_no = set(app_no)
            print_txt(i, xml_path, app_no)
    
    # match result=0
    name_list = ["2012"]
    for name in tqdm(name_list, ncols=70):
        print("years: " + name)
        xml_path = abs_filepath + "ApplicationZip/" + name + "/"
        no_match(xml_path)

    # 获取专利分类号
    name_list = ["2015", "2016", "2017", "2018"]
    for name in tqdm(name_list, ncols=70):
        xml_path = abs_filepath + "ApplicationZip/" + name + "/"
        info_to_excel(xml_path)