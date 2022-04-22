import os,re
import xlrd
import json
# 根据模板建立文件夹，创建.md文件，修改索引文件sidebars.js
rewritemd=0; #标志，是否重写已存在的md文档

filedir=os.getcwd()
curdirname=filedir.split('\\')[-1]
filedir="/".join(filedir.split('\\'))
print(curdirname);
print(filedir)



dic=dict();#除去intro的文件字典，字符串内容不含"doc"，用于建文件夹,字典套字典带列表
jsonlist=[curdirname+"/abstract/doc",dict()]; #输入sidebars.js的内容,json格式


wb = xlrd.open_workbook("../数模网站架构.xlsx")
sheet = wb.sheet_by_name('1模型')


def adddot(s):   #网站左侧目录加点
    num=re.findall("\d+",s);
    if len(num)  ==1:
        num=num[0];
        return num+'.'+s.split(num)[-1]
    return s; #没有数字的直接返回


theme1= '';
theme2= '';
for i in range(1,sheet.nrows ):
    if sheet.cell(i,1).value:
        theme2=sheet.cell(i,1).value;
        theme2d= adddot(theme2); #网站左侧目录加点
        if sheet.cell(i,0).value:
            theme1=sheet.cell(i,0).value;
            theme1d= adddot(theme1);  #网站左侧目录加点
            dic[theme1]=dict();
            jsonlist[-1][theme1d]=[dict()];
        jsonlist[-1][theme1d][-1][theme2d]=[];
        dic[theme1][theme2]=[];
        
    mdname=sheet.cell(i,2).value;#0子文件夹名称
    rdirname=theme1+'/'+theme2+'/'+mdname  #1模块内相对路径
    dirname= filedir+'/'+rdirname; #2绝对路径，用于建立文件夹
    
    dic[theme1][theme2].append(dirname);
    jsonlist[-1][theme1d][-1][theme2d].append(curdirname+'/'+rdirname+'/'+'doc')#3网站内相对路径，用于索引


#修改索引文件
print(json.dumps(jsonlist,ensure_ascii=False, sort_keys=False, indent=4, separators=(', ', ': ')))

path_bar='../../sidebars.js'

if os.path.exists(path_bar):
    with open(path_bar,'r',encoding='utf-8') as f: 
        text=f.read();
        text= text.split('=',1)[1]
        text=text.strip().strip(';')
        sidebarsdic=eval(text)
        sidebarsdic['模型']=jsonlist        
    with open(path_bar,'w',encoding='utf-8') as f: 
        f.write('module.exports = ')
        f.write(json.dumps(sidebarsdic,ensure_ascii=False, sort_keys=False, indent=4, separators=(', ', ': ')))



#建立文件夹，生成markdown模板文档
# 确定内容md文档
sample_txt="";
with open("format.md","r",encoding='utf-8') as f:
    sample_txt= f.read()

for key in dic.keys(): # keys可以作为1级文件夹名称
    dir1=filedir+'/'+key; #1级目录
    if not os.path.exists(dir1):
        os.makedirs(dir1, mode=0o777)
    for key2 in dic[key].keys():  
        dir2=dir1+'/'+key2; #2级目录
        if not os.path.exists(dir2):
            os.makedirs(dir2, mode=0o777)
    
        for path in dic[key][key2]:  
            path=path.strip();
            if not os.path.exists(path):
                os.makedirs(path, mode=0o777)
            txtfile=os.path.join(path, path.split('/')[-1]+'.md')
            if rewritemd or (not os.path.exists(txtfile))  : #是否会把其中的md重写
                with open(txtfile,"w",encoding='utf-8') as f:
                    f.write("---\n\
id: doc\n\
title: %s   \n\
---           \n%s"%(path.split('/')[-1],sample_txt));



    #os.rmdir(path)

#"""
