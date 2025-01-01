from flask import Flask, request, jsonify, render_template
import dill
import pandas as pd

# Define the MovieRecommender class (ensure consistency with the trained model)
class MovieRecommender:
    def __init__(self, movies_df, ratings_df):
        self.movies = movies_df
        self.ratings = ratings_df
        self._prepare_data()

    def _prepare_data(self):
        # Merge movies and ratings data
        self.df = pd.merge(self.ratings, self.movies, how='left', on='movieId')
        self.cosine_sim = self._compute_cosine_sim()

    def _compute_cosine_sim(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel

        # Initialize TF-IDF Vectorizer with stop word removal
        cv = TfidfVectorizer(stop_words='english')
        tfidf_matrix = cv.fit_transform(self.movies['genres'].fillna(''))  # Handle missing genres
        return linear_kernel(tfidf_matrix, tfidf_matrix)

    def get_similar_movies(self, title, top_n=10):
        # Ensure the movie title exists
        if title not in self.movies['title'].values:
            return []

        # Create index mapping and calculate similarity
        indices = pd.Series(self.movies.index, index=self.movies['title']).drop_duplicates()
        idx = indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]  # Exclude the movie itself

        # Return the titles of similar movies
        movie_indices = [i[0] for i in sim_scores]
        return self.movies['title'].iloc[movie_indices].tolist()

# Initialize Flask app
app = Flask(__name__)

# Load the model using dill
try:
    with open('model.pkl', 'rb') as file:
        model = dill.load(file)
except FileNotFoundError:
    model = None
    print("Error: model.pkl not found. Ensure the file exists and is in the correct directory.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Parse JSON request
    data = request.get_json()
    movie_title = data.get("movie_title")

    if not movie_title:
        return jsonify({"error": "Movie title is required"}), 400

    # Generate recommendations
    try:
        recommendations = model.get_similar_movies(movie_title, top_n=10) if model else []
        if not recommendations:
            return jsonify({"error": f"No recommendations found for '{movie_title}'"}), 404
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
