import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import lime
import lime.lime_tabular
import webbrowser
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
    menu = st.sidebar.radio('Affichage', ['Prédiction', 'Variables', 'Importance des variables'])
    if menu == 'Variables':
        select_var = st.radio("Quel variable voulez-vous examiner ?", 
        ('credit_annuity_ratio', 'DAYS_EMPLOYED', 'AMT_ANNUITY', 'credit_goods_price_ratio'))
        sns.distplot(data[data['TARGET_x'] == 0][select_var], label='Rembourse')
        sns.distplot(data[data['TARGET_x'] == 1][select_var], label='Rembouse pas')
        plt.axvline(data.loc[id_client][select_var], color='g', label='Client actuel')
        plt.legend()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
    elif menu == 'Prédiction':
        prob0 = gbm.predict_proba(data.drop(['TARGET_x', 'TARGET_y'], axis = 1).loc[[id_client]])[0][0]
        st.header('La probabilité de remboursé le prêt est :')
        st.subheader(prob0)
    elif menu == 'Importance des variables':
        #GLOBAL
        st.subheader('Interprétation globale')
        sns.barplot(x="imp", y="col", data=featureimp)
        plt.title('Importance des variables')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()
        #LOCAL
        st.subheader('Interprétation locale')
        predict_fn_gbm = lambda x: gbm.predict_proba(x).astype(float)
        X = data.drop(['TARGET_x', 'TARGET_y'], axis=1)
        val = X.values
        explainer = lime.lime_tabular.LimeTabularExplainer(val, feature_names = X.columns,class_names=['Rembourse','Rembourse pas'],kernel_width=5)
        choosen_instance = X.loc[[id_client]].values[0]
        exp = explainer.explain_instance(choosen_instance, predict_fn_gbm,num_features=10)
        components.html(exp.as_html(), height=800)

    

if __name__ == "__main__":
    main()