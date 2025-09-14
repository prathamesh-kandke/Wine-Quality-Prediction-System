import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tkinter import *
import pickle

from PIL import Image, ImageTk


win=Tk()

# Load the background image
bg_image = Image.open("image.png")  # Replace with your actual image file
bg_image = bg_image.resize((1588, 1600), Image.LANCZOS)  # Resize to match the window size
bg_photo = ImageTk.PhotoImage(bg_image)


# Create a Canvas to place the background image
canvas = Canvas(win, width=1500, height=1600)
canvas.pack(fill="both", expand=True)
canvas.create_image(0, 0, image=bg_photo, anchor="nw")  # Set image as background



def show():
    fixed_acidity = float(var1.get())
    volatile_acidity = float(var2.get())
    citric_acid = float(var3.get())
    residual_sugar = float(var4.get())
    chlorides = float(var5.get())
    free_sulfur_dioxide = float(var6.get())
    total_sulfur_dioxide = float(var7.get())
    density = float(var8.get())
    pH = float(var9.get())
    sulphates = float(var10.get())
    alcohol = float(var11.get())

    # print(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol)

    new_df = pd.DataFrame({'fixed acidity': [fixed_acidity],
                           'volatile acidity': [volatile_acidity],
                           'citric acid': [citric_acid],
                           'residual sugar': [residual_sugar],
                           'chlorides': [chlorides],
                           'free sulfur dioxide': [free_sulfur_dioxide],
                           'total sulfur dioxide': [total_sulfur_dioxide],
                           'density': [density],
                           'pH': [pH],
                           'sulphates': [sulphates],
                           'alcohol': [alcohol]
                           })

    with open('wine_sc', 'rb') as f:
        sc = pickle.load(f)
    with open('wine_model', 'rb') as f:
        model1 = pickle.load(f)

    new_df_sc = sc.transform(new_df)

    # prediction
    result=model1.predict(new_df_sc)
    #print(result)
    if result[0]==1:
        z='Good Quality Wine'
    else:
        z='Average Quality Wine'
    l13.config(text=z)


win.title('Wine Quality Prediction System')
win.geometry('900x1000')
win.config(bg='#FFFAF0')


# main label (l1)
l1 = Label(win, text='Wine Quality Prediction System'.upper(), bg='white', fg='black', bd=5, relief='ridge',
         font=('times new roman', 19, 'bold'))
l1.place(x=130, y=40)

# l2 - fixed acidity
l2 = Label(win, text='Fixed Acidity', bg='white', fg='black', bd=5, relief='ridge', width=20,
         font=('times new roman', 14, 'bold'))
l2.place(x=100, y=120)

var1 = StringVar()
e2 = Entry(win, bg='white', fg='black', bd=5, relief='ridge', width=30, textvariable=var1 ,font=('times new roman', 14))
e2.place(x=350, y=120)


# l3 - volatile acidity
l3 = Label(win, text='Volatile Acadity', bg='white', fg='black', bd=5, relief='ridge', width=20,
         font=('times new roman', 14, 'bold'))
l3.place(x=100, y=160)

var2 = StringVar()
e3 = Entry(win, bg='white', fg='black', bd=5, relief='ridge', width=30, textvariable=var2 ,font=('times new roman', 14))
e3.place(x=350, y=160)


# l4 - citric acid
l4 = Label(win, text='Citric Acid', bg='white', fg='black', bd=5, relief='ridge', width=20,
         font=('times new roman', 14, 'bold'))
l4.place(x=100, y=200)

var3 = StringVar()
e4 = Entry(win, bg='white', fg='black', bd=5, relief='ridge', width=30, textvariable=var3 ,font=('times new roman', 14))
e4.place(x=350, y=200)


# l5 - residual sugar
l5 = Label(win, text='Residual Sugar', bg='white', fg='black', bd=5, relief='ridge', width=20,
         font=('times new roman', 14, 'bold'))
l5.place(x=100, y=240)

var4 = StringVar()
e5 = Entry(win, bg='white', fg='black', bd=5, relief='ridge', width=30, textvariable=var4 ,font=('times new roman', 14))
e5.place(x=350, y=240)


# l6 - chlorides
l6 = Label(win, text='Chlorides', bg='white', fg='black', bd=5, relief='ridge', width=20,
         font=('times new roman', 14, 'bold'))
l6.place(x=100, y=280)

var5 = StringVar()
e6 = Entry(win, bg='white', fg='black', bd=5, relief='ridge', width=30, textvariable=var5 ,font=('times new roman', 14))
e6.place(x=350, y=280)


# l7 - free sulfur dioxide
l7 = Label(win, text='Free Sulphur Dioxide', bg='white', fg='black', bd=5, relief='ridge', width=20,
         font=('times new roman', 14, 'bold'))
l7.place(x=100, y=320)

var6 = StringVar()
e7 = Entry(win, bg='white', fg='black', bd=5, relief='ridge', width=30, textvariable=var6 ,font=('times new roman', 14))
e7.place(x=350, y=320)


# l8 - total sulfur dioxide
l8 = Label(win, text='Total Sulphur Dioxide', bg='white', fg='black', bd=5, relief='ridge', width=20,
         font=('times new roman', 14, 'bold'))
l8.place(x=100, y=360)

var7 = StringVar()
e8 = Entry(win, bg='white', fg='black', bd=5, relief='ridge', width=30, textvariable=var7 ,font=('times new roman', 14))
e8.place(x=350, y=360)


# l9 - density
l9 = Label(win, text='Density', bg='white', fg='black', bd=5, relief='ridge', width=20,
         font=('times new roman', 14, 'bold'))
l9.place(x=100, y=400)

var8 = StringVar()
e9 = Entry(win, bg='white', fg='black', bd=5, relief='ridge', width=30, textvariable=var8 ,font=('times new roman', 14))
e9.place(x=350, y=400)


# l10 - pH
l10 = Label(win, text='pH', bg='white', fg='black', bd=5, relief='ridge', width=20,
         font=('times new roman', 14, 'bold'))
l10.place(x=100, y=440)

var9 = StringVar()
e10 = Entry(win, bg='white', fg='black', bd=5, relief='ridge', width=30, textvariable=var9 ,font=('times new roman', 14))
e10.place(x=350, y=440)


# l11 - sulphates
l11 = Label(win, text='Sulphates', bg='white', fg='black', bd=5, relief='ridge', width=20,
         font=('times new roman', 14, 'bold'))
l11.place(x=100, y=480)

var10 = StringVar()
e11 = Entry(win, bg='white', fg='black', bd=5, relief='ridge', width=30, textvariable=var10 ,font=('times new roman', 14))
e11.place(x=350, y=480)


# l12 - alcohol
l12 = Label(win, text='Alcohol', bg='white', fg='black', bd=5, relief='ridge', width=20,
         font=('times new roman', 14, 'bold'))
l12.place(x=100, y=520)

var11 = StringVar()
e12 = Entry(win, bg='white', fg='black', bd=5, relief='ridge', width=30, textvariable=var11 ,font=('times new roman', 14))
e12.place(x=350, y=520)


# button
b1 = Button(win, text='PREDICT QUALITY', width=16, bd=5, fg='black', bg='grey',relief='ridge', font=('times new roman', 12, 'bold'), command=show)
b1.place(x=270, y=600)


# another label l13
l13 = Label(win, text='Quality of Wine is', bg='green', fg='white', bd=5, relief='ridge', width=25, height=3,
         font=('times new roman', 20, 'bold'))
l13.place(x=150, y=670)


""""# Load and display an image
image = Image.open("image.png")  # Replace with your image file
image = image.resize((500, 400))  # Resize if needed
photo = ImageTk.PhotoImage(image)

image_label = Label(win, image=photo, bg='#FFFAF0')
image_label.place(x=750, y=150)  # Adjust position as needed"""








win.mainloop()