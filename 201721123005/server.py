# from urllib.parse import urlencode,quote,unquote
# import time
# import re
from main_model import Model

vocab_file = './couplet/vocabs'
model_dir = '.\models\output_couplet'

m = Model(
    None, None, None, None, vocab_file,
    num_units=1024, layers=4, dropout=0.2,
    batch_size=32, learning_rate=0.0001,
    output_dir=model_dir,
    restore_model=True, init_train=False, init_infer=True)
output = m.infer(' '.join("日久天长"))
output = ''.join(output.split(' '))
print(output)

# def Get_text(in_str):
#     if len(in_str) == 0 or len(in_str) > 50:
#         output = u'您的输入太长了'
#     else:
#         output = m.infer(' '.join(in_str))
#         output = ''.join(output.split(' '))
#     print('上联：%s；下联：%s' % (in_str, output))
#     # return jsonify({'output': output})
#     return  output
#
#
# def Get_couplet(text):  #校对
#     if text:
#         patten = re.compile('''"XialianCandidates":\["(.*?)"+\]''', re.S)
#         data2 = re.findall(patten, text)
#         return data2
#     else:
#         return ["结果不符合题目，请重新设计！"]
#
#
#
# def get_xialian():
#     shanglian  = entry.get()  #用户输入上联
#     xialians = Get_couplet(Get_text(shanglian))#获取下联
#     scr1 = scrolledtext.ScrolledText(root, width=5, height=10, font=("隶书", 18))#设置滚动窗口的参数
#     scr1.place(x=600, y=150)  # 滚动文本框在页面的位置
#     # scr1.insert(END, '上联：\t\t')
#     scr2 = scrolledtext.ScrolledText(root, width=5, height=10, font=("隶书", 18))  # 设置滚动窗口的参数
#     scr2.place(x=800, y=150)  # 滚动文本框在页面的位置
#     # scr2.insert(END, '下联：\n')
#
#     for Xialian in xialians:
#         xialians = Xialian.split(',')
#         for xialian in xialians:
#             scr1.insert(END,shanglian+'\t\n')
#             scr2.insert(END,xialian.replace('"','')+'\n')
#
#
# from tkinter import *
# import tkinter as tk
# from PIL import Image, ImageTk
# from tkinter import scrolledtext
#
# if __name__ == '__main__':
#     root = Tk()
#     root.title("对对联")
#     root.geometry("1200x500")
#     root.geometry("+100+150")
#
#     # 做成背景的装饰
#     pic1 = Image.open('Background.jpg').resize((1200, 500))  #加载图片并调整大小至窗口大小
#     pic = ImageTk.PhotoImage(pic1)
#     render = Label(root, image=pic, compound=tk.CENTER, justify=tk.LEFT)
#     render.place(x=0, y=0)
#
#     #标签 and 输入框
#     label = Label(root, font=('微软雅黑', 20), fg='black')
#     label.grid(row=0,column=0,padx=20, pady=20, ipadx=20,ipady=20)
#     entry = Entry(root, font=('宋体', 25),width = 15)
#     entry.grid(row=1, column=2,padx=10, pady=10, ipadx=10, ipady=10)
#
#     #按钮
#     button = Button(root,text='对下联',font=('微软雅黑', 17), bg="#A6917C",command=get_xialian)
#     button.grid(row=1, column=3, padx=10, pady=5, ipadx=10, ipady=2)
#
#     root.mainloop()
#
#
