import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import lime
import lime.lime_tabular
from sklearn.externals import joblib

def main():
    st.title("Dashboard : Prêt à dépenser")
    data = pd.read_csv('./fulldata.csv')
    featureimp = pd.read_csv('./featureimp.csv')
    gbm = joblib.load('loan_model.pkl')
    data = data.set_index('SK_ID_CURR')
    id_client = st.sidebar.selectbox(
    'Choisissez votre numéro de client',
    data.index)
    if st.checkbox('Variables'):
        select_var = st.radio("Quel variable voulez-vous examiner ?", 
        ('credit_annuity_ratio', 'DAYS_EMPLOYED', 'AMT_ANNUITY', 'credit_goods_price_ratio'))
        sns.distplot(data[data['TARGET_x'] == 0][select_var], label='Rembourse')
        sns.distplot(data[data['TARGET_x'] == 1][select_var], label='Rembouse pas')
        plt.axvline(data.loc[id_client][select_var], color='g', label='Client actuel')
        plt.legend()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif st.checkbox('Prédiction'):
        prob0 = gbm.predict_proba(data.drop(['TARGET_x', 'TARGET_y'], axis = 1).loc[[id_client]])[0][0]
        st.write('La probabilité de remboursé le prêt est', prob0)
        sns.barplot(x="imp", y="col", data=featureimp)
        plt.title('Importance des variables')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        predict_fn_gbm = lambda x: gbmimp.predict_proba(x).astype(float)
        val = data.drop(['TARGET_x', 'TARGET_y'], axis = 1 ).values

        explainer = lime.lime_tabular.LimeTabularExplainer(val,feature_names = data.drop(['TARGET_x', 'TARGET_y'], axis = 1 ).columns ,class_names=['Rembourse','Rembourse pas'],kernel_width=5)
        choosen_instance = data.drop(['TARGET_x', 'TARGET_y'], axis = 1 )[[id_client]].values[0]
        exp = explainer.explain_instance(choosen_instance, predict_fn_gbm,num_features=10)
        exp.show_in_notebook(show_all=False)
        #modifier lime

    

if __name__ == "__main__":
    main()