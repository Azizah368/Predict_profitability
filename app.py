import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack

# Memuat model terlatih
model = pickle.load(open('gradient_boosting_model.pkl', 'rb'))

# Memuat TF-IDF dan Label Encoder
tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
label_encoder_restaurant = pickle.load(open('label_encoder_restaurant.pkl', 'rb'))
label_encoder_menu_category = pickle.load(open('label_encoder_menu_category.pkl', 'rb'))
label_encoder_menu_item = pickle.load(open('label_encoder_menu_item.pkl', 'rb'))

st.title('Prediksi Profitabilitas Menu Restoran')

# Input dari pengguna
restaurant_id = st.text_input('Restaurant ID')
menu_category = st.text_input('Menu Category')
menu_item = st.text_input('Menu Item')
price = st.number_input('Price', min_value=0.0, step=0.01)
ingredients = st.text_area('Ingredients (separate by comma)')

if st.button('Predict'):
    # Menyiapkan fitur untuk prediksi
    restaurant_id_encoded = label_encoder_restaurant.transform([restaurant_id])[0]
    menu_category_encoded = label_encoder_menu_category.transform([menu_category])[0]
    menu_item_encoded = label_encoder_menu_item.transform([menu_item])[0]
    
    ingredients_tfidf = tfidf.transform([ingredients])
    
    features = hstack([pd.DataFrame([[restaurant_id_encoded, menu_category_encoded, menu_item_encoded, price]], columns=['RestaurantID', 'MenuCategory', 'MenuItem', 'Price']).values, ingredients_tfidf])
    
    # Melakukan prediksi
    prediction = model.predict(features)
    prediction_label = 'Profitable' if prediction[0] == 1 else 'Not Profitable'
    
    st.write(f'The predicted profitability is: {prediction_label}')
