# -*- coding: utf-8 -*-

import sys

reload(sys)

sys.setdefaultencoding('utf-8')
from bs4 import BeautifulSoup

for i in range(1,105109):
        
    path = '/data0/docker/lican/English/Body/Body(raw)/Body%ld.html'%i
    with open(path, 'r') as f:
        Soup = BeautifulSoup(f.read(), 'lxml')
        bodys = Soup.select('p')
    Body = []
    for body in bodys:    
        Body.append(body.getText())
    print(i)
    B = " ".join(Body)
    print(B.encode('utf8'))
    

    
    
    filename = "/data0/docker/lican/English/Body/Bodybeautifulsoup/Body%ld.txt"%i     
    with open(filename,'w') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        f.write(B.encode('utf8'))
     
    