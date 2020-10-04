###HEART DISEASE PREDICTION USING  MACHINE LEARNING##
###### DATASET IS COLLECTED FROM UCI REPOSITORY ####
#####DATA SET DESCRIPTION #######
'''Data contains; 

age - age in years 
sex - (1 = male; 0 = female) 
cp - chest pain type 
trestbps - resting blood pressure (in mm Hg on admission to the hospital) 
chol - serum cholestoral in mg/dl 
fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false) 
restecg - resting electrocardiographic results 
thalach - maximum heart rate achieved 
exang - exercise induced angina (1 = yes; 0 = no) 
oldpeak - ST depression induced by exercise relative to rest 
slope - the slope of the peak exercise ST segment 
ca - number of major vessels (0-3) colored by flourosopy 
thal - 3 = normal; 6 = fixed defect; 7 = reversable defect 
target - have disease or not (1=yes, 0=no)
'''
##################  CODING  ####################

import pandas as pd
from numpy import *
###############load dataset ################################################
data=pd.read_csv("f:/project/heart123.csv")
heart=data.values

####################separate attributes and label #############################
x=heart[0:,0:13] ####attributes
y=heart[0:,13:]#  labels
########split data set for training and testing#####
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=5)
y_train=y_train.ravel() 

###################################################

def predict():
    a=float(v1.get())
    b=float(v2.get())
    c=float(v3.get())
    d=float(v4.get())
    e=float(v5.get())
    f=float(v6.get())
    g=float(v7.get())
    h=float(v8.get())
    i=float(v9.get())
    j=float(v10.get())
    k=float(v11.get())
    l=float(v12.get())
    m=float(v13.get())
    
    ######predict by Logistic Regression as it is best
    result=L.predict([[a,b,c,d,e,f,g,h,i,j,k,l,m]])
    if result==1:
        M.showinfo(title="heart Disease prediction",message="You may suffer from heart problem")
    else:
        M.showinfo(title="heart Disease prediction",message="You  dont have heart problem")
    
def reset():
    v1.set("")
    v2.set("")
    v3.set("")
    v4.set("")
    v5.set("")
    v6.set("")
    v7.set("")
    v8.set("")
    v9.set("")
    v10.set("")
    v11.set("")
    v12.set("")
    v13.set("")
def knn():
    global K
    global K_acc
    from sklearn.neighbors import KNeighborsClassifier
    K=KNeighborsClassifier(n_neighbors=5)
    ####Train the model by training dataset
    K.fit(x_train,y_train)
    ######test the model
    y_pred_knn=K.predict(x_test)
    
    #####Find accuracy for KNN
    from sklearn.metrics import accuracy_score
    K_acc=round(accuracy_score(y_pred_knn,y_test)*100,2)
    M.showinfo(title="KNearest Neighbors",message="Accuracy is"+str(K_acc))
def logreg():
    global L
    global L_acc
    from sklearn.linear_model import LogisticRegression
    L=LogisticRegression(solver='liblinear')
    ####Train the model by training dataset
    L.fit(x_train,y_train)
    ######test the model
    y_pred_logreg=L.predict(x_test)
    
    #####Find accuracy for logistic regression
    from sklearn.metrics import accuracy_score
    L_acc=round(accuracy_score(y_pred_logreg,y_test)*100,2)
    M.showinfo(title="Logistic Regression",message="Accuracy is"+str(L_acc))
def naivebayes():
    global N
    global N_acc
    from sklearn.naive_bayes import GaussianNB
    N=GaussianNB()
    ##train the naive bayes model
    N.fit(x_train,y_train)
    #test the model with testing dataset
    y_pred_naive=N.predict(x_test)
    #Find   accuracy
    from sklearn.metrics import accuracy_score
    N_acc=round(accuracy_score(y_pred_naive,y_test)*100,2)
    M.showinfo(title="Naive Bayes",message="Accuracy is"+str(N_acc))
def decisiontree():
    global D
    global D_acc
    from sklearn.tree import DecisionTreeClassifier
    D=DecisionTreeClassifier()
    D.fit(x_train,y_train)
    y_pred_dt=D.predict(x_test)
    
    from sklearn.metrics import accuracy_score
    D_acc=round(accuracy_score(y_pred_dt,y_test)*100,2)
    M.showinfo(title="Decision Tree",message="Accuracy is"+str(D_acc))
###########COMPARE ALL THE 4 MODELS##############
def compare():
    global K_acc;global L_acc;global N_acc;global D_acc
    import matplotlib.pyplot as plt
    clf=['KNN','logistic','naivebayes','decisiontree']
    comp_acc=[K_acc,L_acc,N_acc,D_acc]
    plt.bar(clf,comp_acc,color=['red','orange','blue','green'])
    plt.xlabel("Model comparison")
    plt.ylabel("Accuracy")
    plt.title("KNN vs Logistic vs naivebayes vs decisiontree")
    plt.show()
##############  GRAPHICAL USER INTERFACE############    
from tkinter import *
import tkinter.messagebox as M
w=Tk()
w.title("HEART  DISEASE PREDICTION")
v1=StringVar()
v2=StringVar()
v3=StringVar()
v4=StringVar()
v5=StringVar()
v6=StringVar()
v7=StringVar()
v8=StringVar()
v9=StringVar()
v10=StringVar()
v11=StringVar()
v12=StringVar()
v13=StringVar()
L=Label(w,relief="solid",font=('arial',30,'bold'),text="    HEART DISEASE PREDICTION USING MACHINE LEARNING    ",bg='white',fg='red',)
B1=Button(w,pady=5,bd=5,bg="cyan",command=knn,text='KNearest Neighbors',font=('arial',20,'bold'))
B2=Button(w,pady=5,bd=5,bg="cyan",command=logreg,text='     Logistic Regression',font=('arial',20,'bold'))
B3=Button(w,pady=5,bd=5,bg="cyan",command=naivebayes,text='     Naive Bayes',font=('arial',20,'bold'))
B4=Button(w,pady=5,bd=5,bg="cyan",command=decisiontree,text='Decision Tree Classifier',font=('arial',20,'bold'))


L.grid(row=1,column=1,columnspan=4)
B1.grid(row=2,column=1)
B2.grid(row=2,column=2)
B3.grid(row=2,column=3)
B4.grid(row=2,column=4)

####################################################
L1=Label(w,text='Age    ',font=('arial',20,'bold'))
L2=Label(w,text='Sex    ',font=('arial',20,'bold'))
L3=Label(w,text='Chest pain',font=('arial',20,'bold'))
L4=Label(w,text='Trestbps',font=('arial',20,'bold'))
L5=Label(w,text='cholestoral',font=('arial',20,'bold'))
L6=Label(w,text='Fbs    ',font=('arial',20,'bold'))
L7=Label(w,text='Restecg',font=('arial',20,'bold'))
L8=Label(w,text='Thalach',font=('arial',20,'bold'))
L9=Label(w,text='Exang',font=('arial',20,'bold'))
L10=Label(w,text='Oldpeak',font=('arial',20,'bold'))
L11=Label(w,text='Slope',font=('arial',20,'bold'))
L12=Label(w,text='CA    ',font=('arial',20,'bold'))
L13=Label(w,text='Thal   ',font=('arial',20,'bold'))
E1=Entry(w,bd=5,textvariable=v1,bg='pink',font=('arial',20,'bold'))
E2=Entry(w,bd=5,textvariable=v2,bg='pink',font=('arial',20,'bold'))
E3=Entry(w,bd=5,textvariable=v3,bg='pink',font=('arial',20,'bold'))
E4=Entry(w,bd=5,textvariable=v4,bg='pink',font=('arial',20,'bold'))
E5=Entry(w,bd=5,textvariable=v5,bg='pink',font=('arial',20,'bold'))
E6=Entry(w,bd=5,textvariable=v6,bg='pink',font=('arial',20,'bold'))
E7=Entry(w,bd=5,textvariable=v7,bg='pink',font=('arial',20,'bold'))
E8=Entry(w,bd=5,textvariable=v8,bg='pink',font=('arial',20,'bold'))
E9=Entry(w,bd=5,textvariable=v9,bg='pink',font=('arial',20,'bold'))
E10=Entry(w,bd=5,textvariable=v10,bg='pink',font=('arial',20,'bold'))
E11=Entry(w,bd=5,textvariable=v11,bg='pink',font=('arial',20,'bold'))
E12=Entry(w,bd=5,textvariable=v12,bg='pink',font=('arial',20,'bold'))
E13=Entry(w,bd=5,textvariable=v13,bg='pink',font=('arial',20,'bold'))
#################################################
L1.grid(row=3,column=1)
E1.grid(row=3,column=2)
L2.grid(row=3,column=3)
E2.grid(row=3,column=4)
L3.grid(row=4,column=1)
E3.grid(row=4,column=2)
L4.grid(row=4,column=3)
E4.grid(row=4,column=4)
#######################################
L5.grid(row=5,column=1)
E5.grid(row=5,column=2)
L6.grid(row=5,column=3)
E6.grid(row=5,column=4)
L7.grid(row=6,column=1)
E7.grid(row=6,column=2)
L8.grid(row=6,column=3)
E8.grid(row=6,column=4)
###################################################
L9.grid(row=7,column=1)
E9.grid(row=7,column=2)
L10.grid(row=7,column=3)
E10.grid(row=7,column=4)
L11.grid(row=8,column=1)
E11.grid(row=8,column=2)
L12.grid(row=8,column=3)
E12.grid(row=8,column=4)
L13.grid(row=9,column=1)
E13.grid(row=9,column=2)
########################################
B5=Button(w,bd=5,relief="solid",bg='white',fg="blue",text="   SUBMIT  ",font=('arial',20,'bold'),command=predict)
B6=Button(w,bd=5,relief="solid",bg='white',fg='blue',text="CLEAR DATA",font=('arial',20,'bold'),command=reset)

Bcmp=Button(w,bd=5,relief="solid",bg='white',fg="blue",command=compare,text='MODEL COMPARASION',font=('arial',20,'bold'))
Bcmp.grid(row=10,column=2)
B5.grid(row=10,column=3)
B6.grid(row=10,column=4)
##############################################
Lend=Label(w,relief="solid",font=('arial',30,'bold'),text="    HEART DISEASE PREDICTION USING MACHINE LEARNING    ",bg='white',fg='red',)
Lend.grid(row=11,column=1,columnspan=4)
w.mainloop()


