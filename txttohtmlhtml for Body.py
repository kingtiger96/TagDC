# coding=gbk
for i in range(1,11203033):
    body = open("/data0/docker/lican/Cooking/Body/Body(raw)/Body%d.txt"%(i),'r').read()
    print(body)
    
    filename = "/data0/docker/lican/Cooking/Body/Body(html)/Body%ld.html"%i  
    print(i)
 
    with open(filename,'w') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        f.write(body)
     
       
 


