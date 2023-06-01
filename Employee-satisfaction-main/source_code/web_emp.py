from unittest import result
from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split
import streamlit as st
import pandas as pd
from sklearn import metrics
import pickle
import numpy as np
dt={'Strongly Disagree':1,"Disagree":2,"Neutral":3,"agree":4,"strongly agree":5}
st.set_page_config('Emp Satisfaction',layout="centered")
with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.header('Emp Satisfaction')
result = []
Ql = [
     'I feel fairly well satisfied with my present job ',
     'Most days I am enthusiastic about my work',
     'Each day of work seems like it will never end',
     'I find real enjoyment in my work',
     'I consider my job rather unpleasant ',
     'It is not always easy for me to perform tasks on time.',
     'When I have a deadline to perform a certain task, I always finish it on time.',
     'I always leave my tasks to the last minute.',
     'Sometimes, I feel disappointed with my performance at work, because I know I could have done better',
     'I felt downhearted and blue during the past few weeks',
     'I felt bothered during the past few weeks',
     'I was emotionally stable and sure of myself during the past few weeks.',
     'I felt cheerful, lighthearted during the past few weeks.',
     'I felt tired, worn out, used up, or exhausted during the past few weeks.',
     'Thoughts of being absent',
     'Discuss with coworkers about non-work issues',
     'Spent work time on personal matters',
     'Thoughts of leaving current job',
     'Put less effort into job than should have'
]
col1, col2, col3, col4,col5 = st.columns([2,2,2,3,2])
with col1:
    Industry_type = st.selectbox(
        'Industry type',
        ('IT','Education','RealEstate')
    )
with col2:
    Age = st.number_input('Age',min_value=20,max_value=100)
    
with col3:
    Qualification = st.selectbox(
        'Qualification',
        ('Diploma','Bachelors','Masters','Professional/ others')
    )
    
with col4:
    Experience = st.selectbox(
        'Experience',
        ('1 to less than 5 years ','2.5 to less than 10 years','Above 10 years ')
    )
    
with col5:
    Gender = st.selectbox(
        "Gender",
        ('Male', 'Female'))
q1 = st.select_slider(
     'I feel fairly well satisfied with my present job',
     options=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree'])
q2 = st.select_slider(
     'Most days I am enthusiastic about my work',
     options=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree'])
q3 = st.select_slider(
     'Each day of work seems like it will never end',
     options=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree'])
q4 = st.select_slider(
     'I find real enjoyment in my work',
     options=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree'])
q5 = st.select_slider(
     'I consider my job rather unpleasant',
     options=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree'])
q6 = st.select_slider(
     'It is not always easy for me to perform tasks on time',
     options=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree'])
q7 = st.select_slider(
     'When I have a deadline to perform a certain task, I always finish it on time',
     options=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree'])
q8 = st.select_slider(
     'I always leave my tasks to the last minute',
     options=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree'])
q9 = st.select_slider(
     'Sometimes, I feel disappointed with my performance at work, because I know I could have done',
     options=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree'])
q10 = st.select_slider(
     'I felt downhearted and blue during the past few weeks',
     options=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree'])
q11 = st.select_slider(
     'I felt bothered during the past few weeks ',
     options=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree'])
q12 = st.select_slider(
     'I was emotionally stable and sure of myself during the past few weeks',
     options=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree'])
q13 = st.select_slider(
     'I felt cheerful, lighthearted during the past few weeks',
     options=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree'])
q14 = st.select_slider(
     'I felt tired, worn out, used up, or exhausted during the past few weeks.',
     options=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree'])
q15 = st.select_slider(
     'Thoughts of being absent',
     options=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree'])
q16 = st.select_slider(
     'Discuss with coworkers about non-work issues',
     options=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree'])
q17 = st.select_slider(
     'Spent work time on personal matters',
     options=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree'])
q18 = st.select_slider(
     'Thoughts of leaving current job',
     options=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree'])
q19 = st.select_slider(
     'Put less effort into job than should have',
     options=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree'])
if st.button('Submit'):
    #result = [[Qualification],[Age],[Experience],[Gender],[q1],[q2],[q3],[q4],[q5],[q6],[q7],[q8],[q9],[q10],[q11],[q12],[q13],[q14],[q15],[q16],[q17],[q18],[q19],]
    result.append(Qualification)
    result.append(Age)
    result.append(Experience)
    result.append(Gender)
    result.append(q1)
    result.append(q2)
    result.append(q3)
    result.append(q4)
    result.append(q5)
    result.append(q6)
    result.append(q7)
    result.append(q8)
    result.append(q9)
    result.append(q10)
    result.append(q11)
    result.append(q12)
    result.append(q13)
    result.append(q14)
    result.append(q15)
    result.append(q16)
    result.append(q17)
    result.append(q18)
    result.append(q19)
    s={"Qualification":Qualification,"Age":Age,"Experience":Experience,"Gender":Gender,"PRESENT JOB FEELING":dt[q1],"ENTHUSIASM":dt[q2],"WORKOVERLD":dt[q3],"ENJOYMNT":dt[q4],"UNPLSNTTASK":dt[q5],"TOUGH PERFORMNCE":dt[q6],"TIME MNGMNT":dt[q7],"DISAPNTMNT":dt[q8],"DOWNHRTED":dt[q9],"BOTHRED":dt[q10],"EMOSNAL STABLTY":dt[q11],"CHEERUL":dt[q12],"TIRED":dt[q13],"ABSNT MIND":dt[q14],"DISCUSS CO-WORKER":dt[q15],"PERSNL MTTR":dt[q16],"THOUGHT OF LEAVING":dt[q17],"LESS EFFORT":dt[q18]}
    #print(s)
    dataframe=pd.DataFrame(s,index=[1])
    #dataframe.to_excel("website.xlsx")
    class_value=y=dataframe["PRESENT JOB FEELING"].values
    features=x=dataframe[dataframe.columns[5:]].values
    print("value",y,x)
    loaded_model = pickle.load(open("finalized_model.sav", 'rb'))
    result1 = loaded_model.predict(x)
    final_pred=metrics.accuracy_score(y,result1)
    final_pred=int(final_pred)
    print(metrics.accuracy_score(y,result1))
    list_val=['Strongly Disagree', 'Disagree', 'Neutral', 'agree', 'strongly agree']
    st.write(list_val[final_pred-1])

