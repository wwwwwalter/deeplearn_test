list=['a','b','c','d']
for index,item in enumerate(list):
    print(index,item)

l = [index*value for index,value in enumerate(list)]
print(l)