"""
Movie Recommendation System
Implements collaborative filtering using cosine similarity
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')


class MovieRecommendationSystem:
    """A movie recommendation system using collaborative filtering and content-based filtering"""
    
    def __init__(self, movies_file=None, ratings_file=None):
        """
        Initialize the recommendation system
        
        Args:
            movies_file: Path to movies CSV file
            ratings_file: Path to ratings CSV file
        """
        self.movies = None
        self.ratings = None
        self.user_movie_matrix = None
        self.cosine_sim = None
        self.movie_indices = None
        
        if movies_file and ratings_file:
            self.load_data(movies_file, ratings_file)
    
    def load_data(self, movies_file, ratings_file):
        """Load movie and rating data from CSV files"""
        try:
            self.movies = pd.read_csv(movies_file)
            self.ratings = pd.read_csv(ratings_file)
            self._prepare_data()
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Using sample data instead...")
            self._create_sample_data()
            self._prepare_data()
    
    def _create_sample_data(self):
        """Create sample movie and rating data"""
        # Sample movies
        movies_data = {
            'movieId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            'title': [
                'The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
                'Pulp Fiction', 'Fight Club', 'Forrest Gump', 'Inception',
                'The Matrix', 'Goodfellas', 'The Lord of the Rings: The Return of the King',
                'The Lord of the Rings: The Fellowship of the Ring', 'Star Wars: Episode IV',
                'The Avengers', 'Interstellar', 'The Lion King'
            ],
            'genres': [
                'Drama', 'Crime|Drama', 'Action|Crime|Drama',
                'Crime|Drama', 'Drama', 'Drama|Romance', 'Action|Sci-Fi|Thriller',
                'Action|Sci-Fi', 'Crime|Drama', 'Action|Adventure|Drama',
                'Action|Adventure|Drama', 'Action|Adventure|Fantasy',
                'Action|Adventure|Sci-Fi', 'Adventure|Drama|Sci-Fi', 'Animation|Adventure|Drama'
            ]
        }
        self.movies = pd.DataFrame(movies_data)
        
        # Sample ratings
        ratings_data = {
            'userId': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 1, 2, 3, 4, 5] * 3,
            'movieId': [1, 2, 3, 1, 4, 5, 2, 3, 6, 4, 7, 8, 5, 9, 10, 11, 12, 13, 14, 15] * 3,
            'rating': [5, 5, 4, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4] * 3
        }
        self.ratings = pd.DataFrame(ratings_data)
    
    def _prepare_data(self):
        """Prepare data for recommendation"""
        # Create user-movie matrix
        self.user_movie_matrix = self.ratings.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        # Prepare content-based features
        self._prepare_content_based()
    
    def _prepare_content_based(self):
        """Prepare content-based similarity matrix"""
        # Combine title and genres for content-based filtering
        if 'genres' in self.movies.columns:
            self.movies['content'] = self.movies['title'].fillna('') + ' ' + self.movies['genres'].fillna('')
        else:
            self.movies['content'] = self.movies['title'].fillna('')
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.movies['content'])
        
        # Calculate cosine similarity
        self.cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        # Create movie index mapping
        self.movie_indices = pd.Series(
            self.movies.index, 
            index=self.movies['title']
        ).drop_duplicates()
    
    def get_collaborative_recommendations(self, user_id, n_recommendations=10):
        """
        Get movie recommendations using collaborative filtering
        
        Args:
            user_id: ID of the user
            n_recommendations: Number of recommendations to return
        
        Returns:
            List of recommended movie titles
        """
        if user_id not in self.user_movie_matrix.index:
            return []
        
        # Get user's ratings
        user_ratings = self.user_movie_matrix.loc[user_id]
        
        # Calculate similarity with other users
        user_similarity = cosine_similarity([user_ratings], self.user_movie_matrix)
        user_similarity = user_similarity[0]
        
        # Get similar users
        similar_users = np.argsort(user_similarity)[::-1][1:11]  # Top 10 similar users
        
        # Get movies rated by similar users
        recommendations = {}
        for similar_user_idx in similar_users:
            similar_user_id = self.user_movie_matrix.index[similar_user_idx]
            similar_user_ratings = self.user_movie_matrix.loc[similar_user_id]
            
            for movie_id in similar_user_ratings.index:
                if user_ratings[movie_id] == 0 and similar_user_ratings[movie_id] > 0:
                    if movie_id not in recommendations:
                        recommendations[movie_id] = 0
                    recommendations[movie_id] += similar_user_ratings[movie_id] * user_similarity[similar_user_idx]
        
        # Sort and get top recommendations
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        recommended_movie_ids = [movie_id for movie_id, score in sorted_recommendations[:n_recommendations]]
        
        # Get movie titles
        recommended_movies = self.movies[self.movies['movieId'].isin(recommended_movie_ids)]['title'].tolist()
        return recommended_movies
    
    def get_content_based_recommendations(self, movie_title, n_recommendations=10):
        """
        Get movie recommendations using content-based filtering
        
        Args:
            movie_title: Title of the movie
            n_recommendations: Number of recommendations to return
        
        Returns:
            List of recommended movie titles
        """
        if movie_title not in self.movie_indices:
            return []
        
        # Get index of the movie
        idx = self.movie_indices[movie_title]
        
        # Get similarity scores
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        
        # Sort by similarity score
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top similar movies (excluding the movie itself)
        sim_scores = sim_scores[1:n_recommendations+1]
        
        # Get movie indices
        movie_indices = [i[0] for i in sim_scores]
        
        # Return recommended movies
        return self.movies['title'].iloc[movie_indices].tolist()
    
    def get_hybrid_recommendations(self, user_id, movie_title=None, n_recommendations=10):
        """
        Get movie recommendations using hybrid approach (collaborative + content-based)
        
        Args:
            user_id: ID of the user
            movie_title: Optional favorite movie title
            n_recommendations: Number of recommendations to return
        
        Returns:
            List of recommended movie titles
        """
        # Get collaborative recommendations
        collaborative_recs = self.get_collaborative_recommendations(user_id, n_recommendations)
        
        # Get content-based recommendations if movie title is provided
        content_recs = []
        if movie_title:
            content_recs = self.get_content_based_recommendations(movie_title, n_recommendations)
        
        # Combine and remove duplicates
        all_recommendations = list(set(collaborative_recs + content_recs))
        
        # Return top N recommendations
        return all_recommendations[:n_recommendations]
    
    def add_rating(self, user_id, movie_id, rating):
        """Add a new rating to the system"""
        new_rating = pd.DataFrame({
            'userId': [user_id],
            'movieId': [movie_id],
            'rating': [rating]
        })
        self.ratings = pd.concat([self.ratings, new_rating], ignore_index=True)
        self._prepare_data()
    
    def get_movie_info(self, movie_title):
        """Get information about a movie"""
        movie_info = self.movies[self.movies['title'] == movie_title]
        if not movie_info.empty:
            return movie_info.iloc[0].to_dict()
        return None

