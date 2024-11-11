from flask import Flask, request, jsonify, render_template
import dill
import pandas as pd

# Define the MovieRecommender class (should match the one used to train the model)
class MovieRecommender:
    def __init__(self, movies_df, ratings_df):
        self.movies = movies_df
        self.ratings = ratings_df
        self._prepare_data()

    def _prepare_data(self):
        self.df = pd.merge(self.ratings, self.movies, how='left', on='movieId')
        self.cosine_sim = self._compute_cosine_sim()

    def _compute_cosine_sim(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel

        cv = TfidfVectorizer()
        tfidf_matrix = cv.fit_transform(self.movies['genres'])
        return linear_kernel(tfidf_matrix, tfidf_matrix)

    def get_similar_movies(self, title, top_n=10):
        indices = pd.Series(self.movies.index, index=self.movies['title'])
        idx = indices[title]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]  # Exclude the movie itself
        movie_indices = [i[0] for i in sim_scores]
        return self.movies['title'].iloc[movie_indices]

# Initialize Flask app
app = Flask(__name__)

# Load model using dill
with open('model.pkl', 'rb') as file:
    model = dill.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie_title = data.get("movie_title")

    if not movie_title:
        return jsonify({"error": "Movie title is required"}), 400

    try:
        recommendations = model.get_similar_movies(movie_title, top_n=10)
        recommendations_list = recommendations.tolist() if isinstance(recommendations, pd.Series) else recommendations
        return jsonify({"recommendations": recommendations_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
