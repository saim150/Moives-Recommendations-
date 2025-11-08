"""
Main script for Movie Recommendation System
Demonstrates how to use the recommendation system
"""

from recommendation_system import MovieRecommendationSystem
import pandas as pd


def main():
    """Main function to demonstrate the recommendation system"""
    
    print("=" * 60)
    print("Movie Recommendation System")
    print("=" * 60)
    print()
    
    # Initialize the recommendation system with sample data
    print("Initializing recommendation system...")
    recommender = MovieRecommendationSystem()
    
    # Display available movies
    print("\nAvailable Movies:")
    print("-" * 60)
    for idx, movie in recommender.movies.iterrows():
        print(f"{movie['movieId']}. {movie['title']} ({movie['genres']})")
    
    # Example 1: Collaborative Filtering
    print("\n" + "=" * 60)
    print("Example 1: Collaborative Filtering")
    print("=" * 60)
    user_id = 1
    print(f"\nGetting recommendations for User {user_id}...")
    collaborative_recs = recommender.get_collaborative_recommendations(user_id, n_recommendations=5)
    
    if collaborative_recs:
        print(f"\nTop 5 recommendations for User {user_id}:")
        for i, movie in enumerate(collaborative_recs, 1):
            print(f"{i}. {movie}")
    else:
        print(f"No recommendations found for User {user_id}")
    
    # Example 2: Content-Based Filtering
    print("\n" + "=" * 60)
    print("Example 2: Content-Based Filtering")
    print("=" * 60)
    movie_title = "The Dark Knight"
    print(f"\nGetting movies similar to '{movie_title}'...")
    content_recs = recommender.get_content_based_recommendations(movie_title, n_recommendations=5)
    
    if content_recs:
        print(f"\nTop 5 movies similar to '{movie_title}':")
        for i, movie in enumerate(content_recs, 1):
            print(f"{i}. {movie}")
    else:
        print(f"Movie '{movie_title}' not found in the database")
    
    # Example 3: Hybrid Recommendations
    print("\n" + "=" * 60)
    print("Example 3: Hybrid Recommendations")
    print("=" * 60)
    user_id = 2
    favorite_movie = "The Matrix"
    print(f"\nGetting hybrid recommendations for User {user_id} (favorite: {favorite_movie})...")
    hybrid_recs = recommender.get_hybrid_recommendations(
        user_id, 
        movie_title=favorite_movie, 
        n_recommendations=5
    )
    
    if hybrid_recs:
        print(f"\nTop 5 hybrid recommendations:")
        for i, movie in enumerate(hybrid_recs, 1):
            print(f"{i}. {movie}")
    else:
        print("No recommendations found")
    
    # Example 4: Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode")
    print("=" * 60)
    print("\nYou can now interact with the recommendation system.")
    print("Type 'quit' to exit.\n")
    
    while True:
        print("\nOptions:")
        print("1. Get recommendations for a user (collaborative filtering)")
        print("2. Get similar movies (content-based filtering)")
        print("3. Get hybrid recommendations")
        print("4. View movie information")
        print("5. Add a rating")
        print("6. Quit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            try:
                user_id = int(input("Enter user ID: "))
                n_recs = int(input("Enter number of recommendations (default 5): ") or "5")
                recs = recommender.get_collaborative_recommendations(user_id, n_recs)
                if recs:
                    print(f"\nRecommendations for User {user_id}:")
                    for i, movie in enumerate(recs, 1):
                        print(f"{i}. {movie}")
                else:
                    print(f"No recommendations found for User {user_id}")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        
        elif choice == '2':
            movie_title = input("Enter movie title: ").strip()
            try:
                n_recs = int(input("Enter number of recommendations (default 5): ") or "5")
                recs = recommender.get_content_based_recommendations(movie_title, n_recs)
                if recs:
                    print(f"\nMovies similar to '{movie_title}':")
                    for i, movie in enumerate(recs, 1):
                        print(f"{i}. {movie}")
                else:
                    print(f"Movie '{movie_title}' not found in the database")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        
        elif choice == '3':
            try:
                user_id = int(input("Enter user ID: "))
                movie_title = input("Enter favorite movie title (optional, press Enter to skip): ").strip()
                n_recs = int(input("Enter number of recommendations (default 5): ") or "5")
                recs = recommender.get_hybrid_recommendations(
                    user_id, 
                    movie_title if movie_title else None, 
                    n_recs
                )
                if recs:
                    print(f"\nHybrid recommendations:")
                    for i, movie in enumerate(recs, 1):
                        print(f"{i}. {movie}")
                else:
                    print("No recommendations found")
            except ValueError:
                print("Invalid input. Please enter a valid number.")
        
        elif choice == '4':
            movie_title = input("Enter movie title: ").strip()
            info = recommender.get_movie_info(movie_title)
            if info:
                print(f"\nMovie Information:")
                print(f"Title: {info['title']}")
                print(f"Movie ID: {info['movieId']}")
                if 'genres' in info:
                    print(f"Genres: {info['genres']}")
            else:
                print(f"Movie '{movie_title}' not found in the database")
        
        elif choice == '5':
            try:
                user_id = int(input("Enter user ID: "))
                movie_id = int(input("Enter movie ID: "))
                rating = float(input("Enter rating (1-5): "))
                if 1 <= rating <= 5:
                    recommender.add_rating(user_id, movie_id, rating)
                    print(f"Rating added successfully!")
                else:
                    print("Rating must be between 1 and 5")
            except ValueError:
                print("Invalid input. Please enter valid numbers.")
        
        elif choice == '6' or choice.lower() == 'quit':
            print("Thank you for using the Movie Recommendation System!")
            break
        
        else:
            print("Invalid choice. Please enter a number between 1 and 6.")


if __name__ == "__main__":
    main()

