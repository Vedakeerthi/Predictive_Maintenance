import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pickle 

sc = StandardScaler()
file = open(r'final_model.pkl','rb')
model = pickle.load(file)

@st.cache_resource
def predict(x_train,y_train,x_test,y_test):
    
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)
    y_pred = y_pred.reshape(-1,1)
    
    st.title(f'Actual VS Predicted')

    fig, ax = plt.subplots()
    
    ax.plot(y_test, color="blue", linewidth=2.5, linestyle="-", label="Actual")
    ax.plot(y_pred, color='red', linewidth=2.5, linestyle="-", label="Predicted")
    
    st.pyplot(fig)

    mse = mean_squared_error(y_test,y_pred)
    acc = round(model.score(x_train,y_train),2)*100  
    y_pred = pd.DataFrame(y_pred)
    y_pred = y_pred.rename(columns={0:'y_pred'})
    st.dataframe(y_pred, 1000, 1000)   
    y_pred = y_pred.to_csv(index=False)
                  
    return mse, acc, y_pred
    


def add_remaining_useful_life(df):
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()
    
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life

    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame


def main():
    st.title("Predictive Maintenance using NASA turbofan")
    st.image('Result Images/Turbofan-operation-lbp.png')
    st.header('Drop all the required datasets')

    #Downloading the files
    train_file = st.file_uploader("Drop the training data", type=["csv", "txt"])
    test_file = st.file_uploader("Drop the test data", type=["csv", "txt"])
    y_test = st.file_uploader('Drop the y_test data', type=['csv','txt'])
    
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1,22)] 
    col_names = index_names + setting_names + sensor_names
    
    time_cycle_each = []
    time_cycle_each_test = []
    
    if train_file is not None:
        if train_file.type == "text/plain":
            train_df = pd.read_csv(train_file, sep='\s+',header=None)
        else:
            train_df = pd.read_csv(train_file,header=None)
        
        st.write("Train data dropped successfully!")
        train_df.columns = col_names
        train_data = add_remaining_useful_life(train_df)
        train_data.drop(['setting_3','s_1','s_10','s_18','s_19'],axis=1,inplace=True)
        train_data.drop_duplicates(inplace=True)
    
        for i in range(1,len(train_data.unit_nr.unique())+1):
            time_cycle_each.append(len(train_data.time_cycles[train_data['unit_nr']==i]))
        
        final_train_data = train_data[['s_2', 's_3', 's_4', 's_7', 's_8', 's_11', 's_12', 's_13', 's_15', 's_17', 's_20', 's_21', 'RUL']]
        
        X_train = final_train_data.drop('RUL',axis=1)
        y_train = final_train_data[['RUL']]
        X_train = sc.fit_transform(X_train)
    
        
    if test_file is not None:
        if test_file.type == "text/plain":
            test_df = pd.read_csv(test_file, sep='\s+',header=None)
        else:
            test_df = pd.read_csv(test_file,header=None)
            
        st.write("Test data dropped successfully!")
        test_df.columns = col_names

        test_df.drop(['setting_3','s_1','s_10','s_18','s_19'],axis=1,inplace=True)
        
        

        for i in range(1,len(test_df.unit_nr.unique())+1):
            time_cycle_each_test.append(len(test_df.time_cycles[test_df['unit_nr']==i]))
            
        final_test_data = test_df[['s_2', 's_3', 's_4', 's_7', 's_8', 's_11', 's_12', 's_13', 's_15',
        's_17', 's_20', 's_21']]
            
        X_test = final_test_data
        X_test = sc.fit_transform(X_test)
        
    if y_test is not None:
        if y_test.type == "text/plain":
            y_test = pd.read_csv(y_test, sep='\s+',header=None, names=['RUL'])
        else:
            y_test = pd.read_csv(y_test,header=None, names=['RUL'])
        st.write("y_test data dropped successfully!")
        
        
        y_test = pd.DataFrame({'RUL': np.repeat(y_test['RUL'], time_cycle_each_test)})
        y_test.set_index(i for i in range(y_test.shape[0]))

    
        if st.button('Predict'):
            mse, acc, y_pred = predict(X_train, y_train, X_test, y_test)
            st.download_button(label='Download Predictions',data=y_pred, file_name='y_pred.csv')   
            st.success(f'The model has been executed with loss: {mse}, accuracy: {acc}')  
            
if __name__ == "__main__":
    main()
