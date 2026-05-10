# %%
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# %%
df= pd.read_csv('train_.csv')
df.head(15)

# %%
df.isnull().sum()

# %%
from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Married']=le.fit_transform(df['Married'])
df['Education']=le.fit_transform(df['Education'])
df['Self_Employed']=le.fit_transform(df['Self_Employed'])
df['Loan_Status']=le.fit_transform(df['Loan_Status'])

df


# %%
print(df.dtypes)

# %%
df.fillna(df['Gender'].mode()[0],inplace=True)
df.fillna(df['Married'].mode()[0],inplace=True)
df.fillna(df['Self_Employed'].mode()[0],inplace=True)
df.fillna(df['LoanAmount'].median(),inplace=True)
df.fillna(df['Loan_Amount_Term'].median(),inplace=True)
df.fillna(df['Credit_History'].mode()[0],inplace=True)


# %%
plt.scatter(df['ApplicantIncome'],df['Loan_Status'],c='blue' , linestyle='dashed')

# %%
x= df[['ApplicantIncome','Education','Self_Employed','Credit_History']]
y=df[['Loan_Status']]

x_train,x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=42)

# %%
regr= LogisticRegression()
regr.fit(x_train, y_train)

# %%
predicts= regr.predict(x_test)
predicts

# %%
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test, predicts)
score



# %%
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, predicts)
cm



