import os
import pickle
import joblib
import numpy as np
from django.conf import settings
import sys
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

similarity_path = os.path.join(settings.BASE_DIR, 'artifacts/similarity.pkl')
movies_path = os.path.join(settings.BASE_DIR, 'artifacts/movie_list.pkl')
feature_matrix_path = os.path.join(settings.BASE_DIR, 'artifacts/feature_matrix.pkl')
movie_details_path = os.path.join(settings.BASE_DIR, 'artifacts/movies_details.pkl')
user_likes_path = os.path.join(settings.BASE_DIR, 'artifacts/user_likes.pkl')
comment_model_path = os.path.join(settings.BASE_DIR, 'artifacts/model.pkl')

movies = pickle.load(open(movies_path, 'rb'))
similarity = pickle.load(open(similarity_path, 'rb'))
feature_matrix = pickle.load(open(feature_matrix_path, 'rb'))
movie_details = pickle.load(open(movie_details_path, 'rb'))
user_likes = pickle.load(open(user_likes_path, 'rb'))
comment_model = joblib.load(open(comment_model_path, 'rb'))


def recommend(movie_name):
    try:
        index = movies[movies['Movie'] == str(movie_name)].index[0]
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
        recommended_movie_names = []

        for i in distances[1:5]:
            recommended_movie_names.append(movies.iloc[i[0]].Movie)

        return recommended_movie_names

    except IndexError:
        return []


def generate_user_profile(user):
        try:
            liked_movie_ids = user_likes[(user_likes["user_id"] == user)]["movie_id"]
            liked_movie_indices = movie_details[movie_details["movie_id"].isin(liked_movie_ids)].index

            if len(liked_movie_indices) > 0:
                user_profile = feature_matrix[liked_movie_indices].mean(axis=0)
            else:
                user_profile = None
            return user_profile

        except Exception as e:
            raise Exception(e)
        

def recommend_movie(user, user_profile):
    global feature_matrix
    try:
        if user_profile is None:
            return "User has no liked movies to base recommendations on."
        
        user_profile = user_profile.toarray() if hasattr(user_profile, "toarray") else np.asarray(user_profile).reshape(1, -1)
        feature_matrix = feature_matrix.toarray() if hasattr(feature_matrix, "toarray") else np.asarray(feature_matrix)
        
        similarities = cosine_similarity(user_profile, feature_matrix).flatten()
        movie_details["similarity"] = similarities

        rated_movie_ids = user_likes[user_likes["user_id"] == user]["movie_id"]
        recommendations = movie_details[~movie_details["movie_id"].isin(rated_movie_ids)]

        movies = recommendations.sort_values(by="similarity", ascending=False)[["movie_id", "name_x", "similarity"]]
        movies = movies[movies['similarity'] > 0.25]

        return np.array(movies['movie_id'])
    
    except Exception as e:
        raise Exception(e)


def clean_text(text):
    text = str(text)
    text = re.sub(r'\w*\d\w*', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.lower()
    text = re.sub("\n", " ", text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    
    return text.strip()


classifier = comment_model['toxic_classifier']
tfidf = comment_model['tfidf']

def toxicity_prediction(sample_script):
  temp = tfidf.transform([sample_script]).toarray()
  return classifier.predict_proba(temp)[:,1]