"""
Quick example script demonstrating the Movie Recommendation System
"""

from recommendation_system import MovieRecommendationSystem


def quick_example():
    """Quick example of using the recommendation system"""
    
    print("Movie Recommendation System - Quick Example\n")
    
    # Initialize the system
    print("1. Initializing system...")
    recommender = MovieRecommendationSystem()
    print("   ✓ System initialized with sample data\n")
    
    # Show available movies
    print("2. Available Movies:")
    print("   " + "-" * 50)
    for idx, movie in recommender.movies.head(10).iterrows():
        print(f"   {movie['movieId']:2d}. {movie['title']}")
    print("   ... and more\n")
    
    # Collaborative filtering example
    print("3. Collaborative Filtering Example:")
    print("   Getting recommendations for User 1...")
    recs = recommender.get_collaborative_recommendations(user_id=1, n_recommendations=3)
    for i, movie in enumerate(recs, 1):
        print(f"   {i}. {movie}")
    print()
    
    # Content-based filtering example
    print("4. Content-Based Filtering Example:")
    print("   Movies similar to 'The Dark Knight'...")
    similar = recommender.get_content_based_recommendations('The Dark Knight', n_recommendations=3)
    for i, movie in enumerate(similar, 1):
        print(f"   {i}. {movie}")
    print()
    
    # Hybrid recommendation example
    print("5. Hybrid Recommendation Example:")
    print("   Combining collaborative + content-based filtering...")
    hybrid = recommender.get_hybrid_recommendations(
        user_id=2, 
        movie_title='The Matrix', 
        n_recommendations=3
    )
    for i, movie in enumerate(hybrid, 1):
        print(f"   {i}. {movie}")
    print()
    
    print("Example completed! Run 'python main.py' for interactive mode.")


if __name__ == "__main__":
    quick_example()

