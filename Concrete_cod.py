# import joblib
# import pandas as pd
# import numpy as np
# from PIL.ImImagePlugin import number
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import MinMaxScaler
# import tkinter as tk
# from tkinter import messagebox
#
# from sqlalchemy import values
#
# CDF = pd.read_csv('../Database_Regressions/Concrete_Data.csv')
#
# columns = ['Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water', 'Superplasticizer'
#     , 'Coarse_Aggregate', 'Fine_Aggregate', 'Age', 'Concrete_compressive_strength']
# CDF.columns = columns
# # print(CDF.columns)
# # print(CDF.head().to_string())
# # print(CDF.shape)
# # print(CDF.info())
# # print(CDF.describe())
# # print(CDF.isna().sum())
#
# # for col in CDF.select_dtypes(include='number').columns:
# #     Q1 = CDF[col].quantile(0.25)
# #     Q3 = CDF[col].quantile(0.75)
# #     IQR = Q3 - Q1
# #
# #     lower_bound = Q1 - 1.5 * IQR
# #     upper_bound = Q3 + 1.5 * IQR
# #
# #     outliers=CDF[(CDF[col]<lower_bound)|(CDF[col]>upper_bound)]
# #
# #     print(f'column:{col}')
# #     print(f'Lower:{lower_bound}')
# #     print(f'Upper:{upper_bound}')
# #     print(f'count_Outliers:{len(outliers)}')
# #
# #     if len(outliers)==0:
# #         print('not_Outlier')
# #     print(100 * '-')
#
# def remove_outlier(df,column):
#     Q1=df[column].quantile(0.25)
#     Q3=df[column].quantile(0.75)
#     IQR=Q3-Q1
#     lower_bound=Q1-1.5*IQR
#     upper_bound=Q3+1.5*IQR
#     return (df[(df[column]>=lower_bound)&(df[column]<=upper_bound)])
# for column in ['Blast_Furnace_Slag','Water','Superplasticizer','Fine_Aggregate',
#                'Age','Concrete_compressive_strength']:
#     CDF=remove_outlier(CDF,column)
# # print(CDF.info())
#     # print(100*'-')
# # for col in CDF.select_dtypes(include='number').columns:
# #     Q1 = CDF[col].quantile(0.25)
# #     Q3 = CDF[col].quantile(0.75)
# #     IQR = Q3 - Q1
# #
# #     lower_bound = Q1 - 1.5 * IQR
# #     upper_bound = Q3 + 1.5 * IQR
# #
# #     outliers=CDF[(CDF[col]<lower_bound)|(CDF[col]>upper_bound)]
# #
# #     print(f'column:{col}')
# #     print(f'Lower:{lower_bound}')
# #     print(f'Upper:{upper_bound}')
# #     print(f'count_Outliers:{len(outliers)}')
# #     if len(outliers)==0:
# #         print('not_Outlier')
# #         print(100 * '-')
# for column in [ 'Fine_Aggregate',
#                'Age', 'Concrete_compressive_strength']:##-->Remove Outliers again
#     CDF = remove_outlier(CDF, column)
#     # print(100 * '-')
# # for col in CDF.select_dtypes(include='number').columns:
# #     Q1 = CDF[col].quantile(0.25)
# #     Q3 = CDF[col].quantile(0.75)
# #     IQR = Q3 - Q1
# #
# #     lower_bound = Q1 - 1.5 * IQR
# #     upper_bound = Q3 + 1.5 * IQR
# #
# #     outliers=CDF[(CDF[col]<lower_bound)|(CDF[col]>upper_bound)]
# #
# #     print(f'column:{col}')
# #     print(f'Lower:{lower_bound}')
# #     print(f'Upper:{upper_bound}')
# #     print(f'count_Outliers:{len(outliers)}')
# # print(CDF.info())
# ##Finaly Remove Outlier-------------------------------------#
# x=CDF.drop(['Concrete_compressive_strength'],axis=1)
# y=CDF['Concrete_compressive_strength']
# x_train,x_test,y_train,y_test=train_test_split(x,y ,test_size=0.2,random_state=42)
# # print(len(x_train))
# # print(len(x_test))
#
# ##
# scaler=MinMaxScaler()
# X_trainScaler=scaler.fit_transform(x_train)
# X_test_scaler=scaler.transform(x_test)
# ##
# Lr_Model=LinearRegression()
# Lr_Model.fit(x_train,y_train)
# ##
# y_prediction=Lr_Model.predict(x_test)
# # print(y_prediction)
# # print(len(y_prediction))
# # print(len(y_test))
# ##
# MSE=mean_squared_error(y_test,y_prediction)
# RMSE=np.sqrt(MSE)
# # print(MSE)
# # print(RMSE)
# ##
# # joblib.dump(Lr_Model,'LinearRegression.Concrete.joblib')
# # joblib.dump(scaler,'scaler.Concrete.joblib')
# ##
# def predict_strength():
#     try:
#         values=[float(entry_cement.get()),
#                 float(entry_slag.get()),
#                 float(entry_flyash.get()),
#                 float(entry_water.get()),
#                 float(entry_super.get()),
#                 float(entry_coarse.get()),
#                 float(entry_fine.get()),
#                 float(entry_Age.get()),
#
#                 ]
#         X=np.array([values])
#         y_pred=Lr_Model.predict(X)[0]
#         messagebox.showinfo(f'predicted strength:{y_pred:.2f}MPa')
#     except ValueError:
#                 messagebox.showinfo('Error','please enter valid numeric values')
#
#
# root=tk.Tk()
# root.geometry('350x450')
# root.title('concrete+compressive+strength')
# root.resizable(False,False)
# labl_font=('Arial',12)
#
# tk.Label(root,text='Cement',font=labl_font).pack()
# entry_cement=tk.Entry(root);entry_cement.pack()
#
# tk.Label(root,text='Blast_Furnace_Slag',font=labl_font).pack()
# entry_slag=tk.Entry(root);entry_slag.pack()
#
# tk.Label(root,text='Fly_Ash',font=labl_font).pack()
# entry_flyash=tk.Entry(root);entry_flyash.pack()
#
# tk.Label(root,text='Water',font=labl_font).pack()
# entry_water=tk.Entry(root);entry_water.pack()
#
# tk.Label(root,text='Superplasticizer',font=labl_font).pack()
# entry_super=tk.Entry(root);entry_super.pack()
#
# tk.Label(root,text='Coarse_Aggregate',font=labl_font).pack()
# entry_coarse=tk.Entry(root);entry_coarse.pack()
#
# tk.Label(root,text='Fine_Aggregate',font=labl_font).pack()
# entry_fine=tk.Entry(root);entry_fine.pack()
#
# tk.Label(root,text='Age',font=labl_font).pack()
# entry_Age=tk.Entry(root);entry_Age.pack()
#
#
#
#
# tk.Button(root,text='Predict Strength',font=('Arial' ,14),bg='lightblue',command=predict_strength).pack(pady=20)
#
#
#
#
# root.mainloop()
#
#
# # ---------------------------------------
# import tkinter as tk
# from tkinter import messagebox
# import numpy as np
# import joblib
#
# # Load model
# model = joblib.load("concrete_model.pkl")
#
# # Prediction function
# def predict_strength():
#     try:
#         values = [
#             float(entry_cement.get()),
#             float(entry_slag.get()),
#             float(entry_flyash.get()),
#             float(entry_water.get()),
#             float(entry_super.get()),
#             float(entry_coarse.get()),
#             float(entry_fine.get()),
#             float(entry_age.get())
#         ]
#
#         X = np.array([values])
#         y_pred = model.predict(X)[0]
#
#         messagebox.showinfo("Result", f"Predicted Strength: {y_pred:.2f} MPa")
#
#     except ValueError:
#         messagebox.showerror("Error", "Please enter valid numeric values")
#
#
# # GUI
# root = tk.Tk()
# root.title("Concrete Strength Predictor")
# root.geometry("350x450")
# root.resizable(False, False)
#
# label_font = ("Arial", 12)
#
# # Labels and entries
# tk.Label(root, text="Cement", font=label_font).pack()
# entry_cement = tk.Entry(root); entry_cement.pack()
#
# tk.Label(root, text="Blast Furnace Slag", font=label_font).pack()
# entry_slag = tk.Entry(root); entry_slag.pack()
#
# tk.Label(root, text="Fly Ash", font=label_font).pack()
# entry_flyash = tk.Entry(root); entry_flyash.pack()
#
# tk.Label(root, text="Water", font=label_font).pack()
# entry_water = tk.Entry(root); entry_water.pack()
#
# tk.Label(root, text="Superplasticizer", font=label_font).pack()
# entry_super = tk.Entry(root); entry_super.pack()
#
# tk.Label(root, text="Coarse Aggregate", font=label_font).pack()
# entry_coarse = tk.Entry(root); entry_coarse.pack()
#
# tk.Label(root, text="Fine Aggregate", font=label_font).pack()
# entry_fine = tk.Entry(root); entry_fine.pack()
#
# tk.Label(root, text="Age (days)", font=label_font).pack()
# entry_age = tk.Entry(root); entry_age.pack()
#
# # Predict button
# tk.Button(root, text="Predict Strength", font=("Arial", 14), bg="lightblue",
#           command=predict_strength).pack(pady=20)
#
# root.mainloop()