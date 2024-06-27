# importing libraries
from datasets import load_dataset, load_dataset_builder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, classification_report
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from skops import hub_utils
import pickle
#from skops.card import Card, metadata_from_config
from pathlib import Path
import streamlit as st 
from tempfile import mkdtemp, mkstemp

# Loading the dataset
dataset_name = "saifhmb/FraudPaymentData"
dataset = load_dataset(dataset_name, split = 'train')
dataset = pd.DataFrame(dataset)

dataset = dataset.dropna()
dataset = dataset.drop(['Time_step','Transaction_Id','Sender_Id', 'Sender_Account','Bene_Id','Bene_Account'], axis = 1) #  deleting high cardinality features 
y = dataset.iloc[:, 5].values
dataset = dataset.drop(['Label'], axis = 1)
dataset = dataset.drop(['Sender_lob', 'Sender_Sector'], axis = 1) # delete column since there is only a single unique value for 'Sender_lob' and 'Sender_sector' is a high cardinal feature

# Encoding the Independent Variables 
categoricalColumns = ['Sender_Country', 'Bene_Country', 'Transaction_Type']
onehot_categorical = OneHotEncoder(handle_unknown='ignore', sparse_output= False)
categorical_transformer = Pipeline(steps = [('onehot', onehot_categorical)])

numericalColumns = dataset.select_dtypes(include = np.number).columns
sc = StandardScaler()
numerical_transformer = Pipeline(steps = [('scale', sc)])
preprocessorForCategoricalColumns = ColumnTransformer(transformers=[('cat', categorical_transformer, categoricalColumns)], remainder ='passthrough')
preprocessorForAllColumns = ColumnTransformer(transformers=[('cat', categorical_transformer, categoricalColumns),('num',numerical_transformer,numericalColumns)],
                                            remainder="passthrough")

# Spliting the datset into Training and Test set
X = dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 42) # random state is 0 or 42

# Train Naive Bayes Model using the Training set
# Handling imbalanced dataset
under_sampler = RandomUnderSampler()
X_under, y_under = under_sampler.fit_resample(X_train, y_train)

classifier = GaussianNB() # select the appropriate algorithm for the problem statement
model = Pipeline(steps = [('preprocessorAll', preprocessorForAllColumns),('classifier', classifier)])
model.fit(X_under, y_under)

# Predicting the Test result
y_pred = model.predict(X_test)

# Making the Confusion Matrix and evaluating performance
cm = confusion_matrix(y_pred, y_test, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.array(['0 - Normal', '1 - Fraudulent']))
disp.plot()
plt.show()
acc = accuracy_score(y_test, y_pred)

# Pickling the model
pickle_out = open("model.pkl", "wb") 
pickle.dump(model, pickle_out) 
pickle_out.close()

# Loading the model to predict on the data
pickle_in = open('model.pkl', 'rb') 
model = pickle.load(pickle_in) 

def welcome(): 
    return 'welcome all'

# defining the function which will make the prediction using the data which the user inputs 
def prediction(Sender_Country, Bene_Country, USD_amount, Transaction_Type):
  X = pd.DataFrame([[Sender_Country, Bene_Country, USD_amount, Transaction_Type]], columns = ['Sender_Country', 'Bene_Country', 'USD_amount', 'Transaction_Type'])
  prediction = model.predict(X)
  print(prediction)
  return prediction

# this is the main function in which we define our webpage 
def main(): 
      # giving the webpage a title 
    st.title("Fraud Detection ML App") 
    st.header("Model Description", divider = "gray")
    multi = '''This is a Gaussian Naive Bayes model trained on a synthetic dataset, containining a large variety of transaction types representing normal activities 
    as well as abnormal/fraudulent activities. The model predicts whether a transaction is normal or fraudulent.
    For more details on the model please refer to the model card at https://huggingface.co/saifhmb/fraud-detection-model
    '''
    st.markdown(multi)
    st.markdown("To determine whether a transaction is normal or fraudulent, please **ENTER** the Sender Country, Beneficiary Country, Amount in USD and Transaction Type :")
    col1, col2 = st.columns(2)
    with col1:
        Sender_Country = st.text_input("Sender Country")
    with col2:  
      Bene_Country = st.text_input("Beneficiary Country")
    
    col3, col4 = st.columns(2)
    with col3:
      USD_amount = st.number_input("Amount in USD")
    with col4:
      Transaction_Type = st.text_input("Transaction Type (Please enter one of the following: make-payment, quick-payment, move-funds, pay-check)")
    result = ""
    if st.button("Predict"):
        result = prediction(Sender_Country, Bene_Country, USD_amount, Transaction_Type)
        if result == 0:
            st.success("The output is {}".format(result) + " This is a NORMAL transaction")
        if result == 1:
            st.success("The output is {}".format(result) + " This is a FRAUDULENT TRANSACTION")
    
if __name__=='__main__': 
    main()          
  
