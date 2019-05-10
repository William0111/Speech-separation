
#基础巩固

# condition = 1
# while condition<10:
#     print(condition)
#     condition = condition + 1

# ex_list = [1,2,4,5,6,3,7,8,3,12]
# for i in ex_list:
#     print(i)
#     print()


# for i in range(1,10,2):#1到9而不是1到10,2是步长
#     print(i)

# x=7
# y=7
# z=5
#
# if x!=y:  #记住if和else都有冒号在后面
#     print("x不等于y")
# #elif是第一个判断不满足后开始执行，elif可以不只有一个
# elif z>x:
#     print(z)
# #现在else是前面所有的判断都不满足后再执行
# else:
#     print('x等于y')


#def function_add(a,b):
#     c=a+b
#     print('the a+b=',c)
#     return c
#
# print(function_add(4,3))

# length = 100
# price=None
#
#
# def sale_pen(color,is_old):
#     global price#在函数里面也能定义全局变量，
#     # 结果是函数运行前后全局变量会改变
#     price = 6
#
#     print('length',length,
#         'price',price,
#           'color',color)
# print("Past price",price)
# sale_pen('bule',True)
# print("Now price",price)

# My_book = 'This is my own book.\n I definitely want to write a book'
# print(My_book)
#
# my_file = open('my book','w')  #w是写，r是读
# my_file.write(My_book)
# my_file.close()
#
# append_My_book='\n This is added into Mybook'
#
# my_file = open('my book','a')  #a是append是加上去
# my_file.write(append_My_book)
# my_file.write(My_book)
# my_file.close()
#
# file=open('my book', 'r')
# c = file.readlines()
# print(c)

# class Williams_calculater:
#     name = 'Williams calculater'
#     price = 'priceless'
#     def __init__(self,name,price,length):
#         self.price = price
#         self.name = name
#         self.l = length
#
#     def add(self,x,y):
#         result = x+y
#         print(result)
#     def minus(self,x,y):
#         result = x-y
#         print(result)
#     def times(self,x,y):
#         result = x*y
#         print(result)
#
# print(Williams_calculater.name)
# print(Williams_calculater.times(8,9,9))
#
# W=Williams_calculater("Wc",89,90)
# print(Williams_calculater.name)
# print(W.name)

# a_input = input('Pls input:')
#
# print(a_input)
# if a_input ==str(1):
#     print("Good Luck")
# else:
#     print("Good Luck next time")
#
# a_tuple = (1,2,4,52,2421,42)
# another_tuple = 1,2,5,2,3,1,4,242
#
# a_list = [1.3,424,2,31,31,]
#
# for content in a_list:
#     print(content)
#
# for index in range(len(a_list)):
#     print('index=',index,'n in list=',a_list[index])
#     a_list[1]


a_list = [1.3,424,2,31,31,]
a_list.append(0)
print(a_list)

a_list.insert(2,555)
print(a_list)

a_list.remove(2)
print(a_list)


