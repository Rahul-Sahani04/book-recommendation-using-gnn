# Importing required libraries
import numpy as np
import pandas as pd
import pickle

# Loading Dataset
book = pd.read_csv("data/Books.csv")
user = pd.read_csv("data/Users.csv")
rating = pd.read_csv("data/Ratings.csv")

# Exploratory Data Analysis
print("Books DataFrame:")
print(book.tail())
print("Users DataFrame:")
print(user.tail())
print("Ratings DataFrame:")
print(rating.tail())

print("\nShapes of the DataFrames:")
print(f"Books shape: {book.shape}")
print(f"Users shape: {user.shape}")
print(f"Ratings shape: {rating.shape}")

print("\nMissing values in each DataFrame:")
print(book.isnull().sum())
print(user.isnull().sum())
print(rating.isnull().sum())

print("\nDuplicated entries in each DataFrame:")
print(f"Books duplicates: {book.duplicated().sum()}")
print(f"Users duplicates: {user.duplicated().sum()}")
print(f"Ratings duplicates: {rating.duplicated().sum()}")

print("\nDataFrame Information:")
user.info()
book.info()
rating.info()

# Popularity-Based Recommendation System
# This system suggests books based on their overall popularity and average ratings.

# Merging rating and book dataframes
rating_with_name = rating.merge(book, on="ISBN")

# Calculating the number of ratings and average rating for each book
num_rating_df = (
    rating_with_name.groupby("Book-Title").count()["Book-Rating"].reset_index()
)
num_rating_df.rename(columns={"Book-Rating": "Num_rating"}, inplace=True)

avg_rating_df = (
    rating_with_name.groupby("Book-Title")
    .mean(numeric_only=True)["Book-Rating"]
    .reset_index()
)
avg_rating_df.rename(columns={"Book-Rating": "Avg_rating"}, inplace=True)

# Merging popularity dataframes and filtering top 100 popular books
popular_df = num_rating_df.merge(avg_rating_df, on="Book-Title")
pbr_df = (
    popular_df[popular_df["Num_rating"] >= 300]
    .sort_values("Avg_rating", ascending=False)
    .head(100)
)

# Selecting relevant columns for recommendation
pbr_df = pbr_df.merge(book, on="Book-Title").drop_duplicates("Book-Title")[
    [
        "Book-Title",
        "Book-Author",
        "Publisher",
        "Image-URL-M",
        "Num_rating",
        "Avg_rating",
    ]
]

# Saving the popularity-based recommendation data into a pickle file
pickle.dump(pbr_df, open("data/PopularBookRecommendation.pkl", "wb"))

print(
    "Popularity-based book recommendation system has been created and saved as 'PopularBookRecommendation.pkl'."
)
