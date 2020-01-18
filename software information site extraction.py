# 解析 xml 文件对象
import xml.sax
 
# 继承 xml.sax.ContentHandler 对象
# 实现原理，当开始解析标签的时候调用startElement方法
# 解析标签内容的时候调用characters方法
# 解析完标签之后调用endElement方法
Tags = []
Titles = []
Bodys = []

class BookHandler(xml.sax.ContentHandler):
    # 构造函数
    
    def __init__(self):
        self.CurrentData = ""
        self.name = ""
        self.price = ""
        self.author = ""
 
    # 元素开始事件处理
    def startElement(self,tag,attributes):
        self.CurrentData = tag
        if tag == "row":
            if int(attributes["PostTypeId"])==1:
                Tags.append(attributes["Tags"])
                Titles.append(attributes["Title"])
                Bodys.append(attributes["Body"])
                
               
               
 
    # 元素内容处理事件
    def characters(self,content):
        if self.CurrentData == "name":
            self.name = content
        elif self.CurrentData == "price":
            self.price = content
        elif self.CurrentData == "author":
            self.author = content

 
    # 元素结束事件处理
    def endElement(self,tag):
        if self.CurrentData == "name":
            print("name:"+self.name)
        elif self.CurrentData == "price":
            print("price:"+self.price)
        elif self.CurrentData == "author":
            print("author:"+self.author)
        self.CurrentData = ""

 
# main 方法
if( __name__ == "__main__" ):
 
    # 创建一个XMLReader
    parser = xml.sax.make_parser()
 
    # 关闭命名空间
    parser.setFeature(xml.sax.handler.feature_namespaces,0)
 
    # 重写 ContextHandler,即自定义的类赋值给Handler
    Handler = BookHandler()
 
    # 设置解析器
    parser.setContentHandler(Handler)
 
    # 开始解析 xml 文件
    parser.parse("/data/lican/Posts.xml")
    for i in range(0, 1000000):
        print(i+1)
        filename1 = "/data/lican/StackOverflowsmall/Tags/TagsRaw/Tags%ld.txt"%(i+1)  
        with open(filename1,'w',encoding = 'utf-8',errors = 'ignore') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
            f.write(Tags[i])
        filename2 = "/data/lican/StackOverflowsmall/Tags/TitleRaw/Title%ld.txt"%(i+1)  
        with open(filename2,'w',encoding = 'utf-8',errors = 'ignore') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
            f.write(Titles[i])
        filename3 = "/data/lican/StackOverflowsmall/Tags/BodyRaw/Body%ld.html"%(i+1)  
        with open(filename3,'w',encoding = 'utf-8',errors = 'ignore') as f: # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
       	    f.write(Bodys[i])
    
 

        