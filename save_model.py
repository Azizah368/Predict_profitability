import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

# Melatih model (gunakan data asli atau model yang telah dilatih)
# model = GradientBoostingClassifier().fit(X_train, y_train)

# Menyimpan model
with open('gradient_boosting_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Menyimpan TF-IDF dan Label Encoders
tfidf = TfidfVectorizer(stop_words='english')
with open('tfidf_vectorizer.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidf, tfidf_file)

# Buat dan simpan Label Encoders
label_encoder_restaurant = LabelEncoder().fit(data['RestaurantID'])
label_encoder_menu_category = LabelEncoder().fit(data['MenuCategory'])
label_encoder_menu_item = LabelEncoder().fit(data['MenuItem'])

with open('label_encoder_restaurant.pkl', 'wb') as le_restaurant_file:
    pickle.dump(label_encoder_restaurant, le_restaurant_file)
with open('label_encoder_menu_category.pkl', 'wb') as le_menu_category_file:
    pickle.dump(label_encoder_menu_category, le_menu_category_file)
with open('label_encoder_menu_item.pkl', 'wb') as le_menu_item_file:
    pickle.dump(label_encoder_menu_item, le_menu_item_file)
