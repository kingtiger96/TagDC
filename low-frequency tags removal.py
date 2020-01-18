#coding=utf-8
from collections import Counter
text = " "
Tags_dict = {}
Tags_list = []

"""读取文件"""
for i in range(1,50001):
    text += open("/data/lican/StackOverflowsmall/Tags/Tags/Tags%d.txt"%(i)).read().lower()
    text +=" "

print(Counter(r).most_common())
"""统计词频"""
frequency = {}
for word in text.split():
    if word not in frequency:
        frequency[word] = 1
    else:
        frequency[word] += 1
print(len(frequency))

"""筛选"""
Tags_dict = {}
Tags_dict = {k: v for k, v in frequency.items() if v > 50}
print(Tags_dict)
dict1 = []
for k,v in Tags_dict.items():
     dict1.append(k+":"+str(v))
filename4 = "/data0/docker/lican/English/Tags/Tags_dict.txt" 
with open(filename4,'w',encoding = 'utf-8',errors = 'ignore') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！

    f.write(" ".join(dict1))
     
Tags_list = Tags_dict.keys()
print("tags:", len(Tags_list))

filename = "/data0/docker/lican/English/Tags/Tags_list.txt"  
with open(filename,'w',encoding = 'utf-8',errors = 'ignore') as f1: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
    f1.write(" ".join(Tags_list))


"""提取出筛选后标签的文档序号"""
savelist = []
for i in range(1,105109):
    tags = open("/data0/docker/lican/English/Tags/Tags(qujiankuohao)/Tags%d.txt"%(i),encoding = 'utf-8',errors = 'ignore').read().split()
 
    
    for word in tags:
        if word in Tags_list:
            savelist.append(i)
            
"""去除重复序号"""
savelist1 = list(set(savelist))
savalist_length = len(savelist1)

"""重新保存筛选后的文档"""
count = 0
for i in savelist1:
    tags = open("/data0/docker/lican/English/Tags/Tags(qujiankuohao)/Tags%d.txt"%(i),encoding = 'utf-8',errors = 'ignore').read()
    title = open("/data0/docker/lican/English/Title/Title(raw)/Title%d.txt"%(i),'r',encoding = 'utf-8',errors = 'ignore').read()
    body = open("/data0/docker/lican/English/Body/Bodybeautifulsoup/Body%d.txt"%(i),'r',encoding = 'utf-8',errors = 'ignore').read()
    count = count+1
    print(count)
    filename1 = "/data0/docker/lican/English/Tags/Tags(choose)/Tag%ld.txt"%(count)      
    with open(filename1,'w',encoding = 'utf-8',errors = 'ignore') as f1: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        f1.write(tags)
        
    filename2 = "/data0/docker/lican/English/Title/Title(choose)/Title%ld.txt"%(count)      
    with open(filename2,'w',encoding = 'utf-8',errors = 'ignore') as f2: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        f2.write(title)
        
    filename3 = "/data0/docker/lican/English/Body/Body(choose)/Body%ld.txt"%(count)      
    with open(filename3,'w',encoding = 'utf-8',errors = 'ignore') as f3: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        f3.write(body)
print(len(frequency))
print("tags:", len(Tags_list))
print("document", savalist_length)
