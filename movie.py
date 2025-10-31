import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie ratings data
data = {
    'User1': [5, 4, 0, 1, 0],
    'User2': [4, 0, 0, 1, 1],
    'User3': [1, 1, 0, 5, 4],
    'User4': [0, 0, 5, 4, 0]
}
movies = ['Inception', 'Interstellar', 'Avengers', 'Titanic', 'Joker']
df = pd.DataFrame(data, index=movies)

# Compute similarity between movies
similarity_matrix = cosine_similarity(df.T)
similarity_df = pd.DataFrame(similarity_matrix, index=df.columns, columns=df.columns)

# Recommend similar users for User1
print("User Similarities:\n", similarity_df['User1'].sort_values(ascending=False))

# Predict User1’s rating for “Avengers”
user_sim = similarity_df['User1'][1:]  # skip self
ratings = df.loc['Avengers'][1:]
predicted_rating = (user_sim @ ratings) / user_sim.sum()
print(f"\nPredicted rating for User1 on 'Avengers': {predicted_rating:.2f}")
