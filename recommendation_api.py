from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data
import pickle

app = FastAPI()

# Load dataset
books = pd.read_csv('data/Books.csv')
ratings = pd.read_csv('data/Ratings.csv')

# Encode users and books
user_encoder = LabelEncoder()
book_encoder = LabelEncoder()

ratings['User-ID'] = user_encoder.fit_transform(ratings['User-ID'])
ratings['ISBN'] = book_encoder.fit_transform(ratings['ISBN'])

num_users = len(ratings['User-ID'].unique())
num_books = len(ratings['ISBN'].unique())

# Prepare graph data
user_indices = torch.tensor(ratings['User-ID'].values, dtype=torch.long)
book_indices = torch.tensor(ratings['ISBN'].values + num_users, dtype=torch.long)
edge_index = torch.stack([user_indices, book_indices], dim=0)
edge_attr = torch.tensor(ratings['Book-Rating'].values / ratings['Book-Rating'].max(), dtype=torch.float)
x = torch.ones((num_users + num_books, 1))
data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# Load pre-trained model
from model import BookGNN

input_dim = 1  # Initial feature size
hidden_dim = 64
output_dim = 16  # Embedding size
model = BookGNN(input_dim, hidden_dim, output_dim)
model.load_state_dict(torch.load('models/book_gnn_model-1731667954.pth'))
model.eval()

# Store new user data for periodic training
new_user_interactions = []

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Combine relevant columns to create a 'metadata' field for each book
books['metadata'] = (
    books['Book-Title'].fillna('') + ' ' +
    books['Book-Author'].fillna('') + ' ' +
    books['Publisher'].fillna('')
)

# TF-IDF Vectorizer to encode metadata
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(books['metadata'])

def content_based_recommendation(user_preferences, top_n=10):
    # Join user preferences into a single string
    user_query = ' '.join(user_preferences)
    
    # Transform user query to match TF-IDF matrix
    query_vector = tfidf_vectorizer.transform([user_query])
    
    # Calculate cosine similarity between the query and book metadata
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Get the indices of the most similar books
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    
    # Return the recommended books
    return books.iloc[top_indices][['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher']].to_dict(orient='records')



# Popularity-based recommendations
# def get_popular_books(top_n=10):
#     popular_books = ratings.groupby('ISBN')['Book-Rating'].mean().sort_values(ascending=False)
#     top_books = popular_books.head(top_n).index
#     return books[books['ISBN'].isin(top_books)].to_dict(orient='records')

def get_popular_books(top_n=10):
    popular_books = pickle.load(open('data/PopularBookRecommendation.pkl', 'rb'))
    # Retrieve 10 random books from the popular books list
    
    return popular_books.sample(top_n).to_dict(orient='records')


# Graph-based recommendations
def graph_based_recommendation(user_preferences, top_n=10):
    preference_indices = book_encoder.transform(user_preferences)
    user_feature = torch.ones((1, 1))  # Dummy feature for the new user

    # Temporary graph update
    new_user_index = data.x.size(0)
    new_user_edge_index = torch.tensor([[new_user_index] * len(preference_indices),
                                        preference_indices + num_users], dtype=torch.long)
    updated_edge_index = torch.cat([data.edge_index, new_user_edge_index], dim=1)

    with torch.no_grad():
        updated_x = torch.cat([data.x, user_feature], dim=0)
        updated_out = model(updated_x, updated_edge_index, data.edge_attr)

    new_user_embedding = updated_out[new_user_index]
    scores = torch.matmul(new_user_embedding, updated_out[num_users:].t()).squeeze()
    _, top_book_indices = torch.topk(scores, top_n)
    recommended_books = book_encoder.inverse_transform(top_book_indices.numpy())
    return books[books['ISBN'].isin(recommended_books)].to_dict(orient='records')


# Add user interactions for incremental training
def add_new_user_interaction(user_id, preferences):
    global new_user_interactions
    for pref in preferences:
        isbn = book_encoder.transform([pref])[0]
        new_user_interactions.append({"User-ID": user_id, "ISBN": isbn, "Book-Rating": 5.0})


# Incremental training
def retrain_model():
    global data, model, new_user_interactions
    if not new_user_interactions:
        return  # No new interactions to train on
    
    # Convert new interactions to DataFrame
    new_interactions_df = pd.DataFrame(new_user_interactions)
    new_interactions_df['User-ID'] = user_encoder.transform(new_interactions_df['User-ID'])
    new_interactions_df['ISBN'] = book_encoder.transform(new_interactions_df['ISBN'])

    # Append to existing ratings
    updated_ratings = pd.concat([ratings, new_interactions_df], ignore_index=True)

    # Update graph data
    user_indices = torch.tensor(updated_ratings['User-ID'].values, dtype=torch.long)
    book_indices = torch.tensor(updated_ratings['ISBN'].values + num_users, dtype=torch.long)
    edge_index = torch.stack([user_indices, book_indices], dim=0)
    edge_attr = torch.tensor(updated_ratings['Book-Rating'].values / updated_ratings['Book-Rating'].max(), dtype=torch.float)
    x = torch.ones((num_users + num_books, 1))
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    # Retrain the model
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(10):  # Adjust epochs as needed
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = torch.nn.functional.mse_loss(out, edge_attr.unsqueeze(1))  # Example loss
        loss.backward()
        optimizer.step()
    model.eval()

    # Clear new interactions
    new_user_interactions = []


# FastAPI endpoint
class RecommendationRequest(BaseModel):
    user_id: int = None
    preferences: list = []

@app.post("/recommend")
def recommend_books(request: RecommendationRequest):
    if request.user_id is None:
        if not request.preferences:
            return {"error": "New users must provide preferences."}
        content_recommendations = content_based_recommendation(request.preferences)
        popular_recommendations = get_popular_books()
        return {"content_based": content_recommendations, "popular": popular_recommendations}
    else:
        add_new_user_interaction(request.user_id, request.preferences)
        return {"graph_based": graph_based_recommendation(request.preferences)}

@app.post("/retrain")
def trigger_retrain():
    retrain_model()
    return {"message": "Model retrained with new user data."}

@app.get("/popular")
def popular_books():
    return get_popular_books()