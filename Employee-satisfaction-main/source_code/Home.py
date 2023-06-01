import random
import firebase_admin
import streamlit as st
import pyrebase
firebaseConfig = {
  'apiKey': "AIzaSyA7BDKnsUet5agv3kACvZ0b_7x4ZYlTtFE",
  'authDomain': "earn-money-7cfba.firebaseapp.com",
  'databaseURL': "https://earn-money-7cfba.firebaseio.com",
  'projectId': "earn-money-7cfba",
  'storageBucket': "earn-money-7cfba.appspot.com",
  'messagingSenderId': "1081799124430",
  'appId': "1:1081799124430:web:87531474fd47dd39a6379f",
  'measurementId': "G-NWPYZX8MTP"
}
firebase=pyrebase.initialize_app(firebaseConfig)
auth=firebase.auth()
db=firebase.database()

