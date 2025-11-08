# Movie Recommendation System

A comprehensive movie recommendation system built with Python that uses collaborative filtering and content-based filtering to recommend movies to users.

## Features

- **Collaborative Filtering**: Recommends movies based on similar users' preferences
- **Content-Based Filtering**: Recommends movies similar to a given movie based on genres and titles
- **Hybrid Recommendations**: Combines both collaborative and content-based filtering for better results
- **Interactive Mode**: Command-line interface for easy interaction
- **Sample Data**: Includes sample movie and rating data for testing

## Installation

1. Clone the repository or download the project files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the main script to start the recommendation system:

```bash
python main.py
```

The script will:
1. Initialize the system with sample data
2. Display available movies
3. Show examples of different recommendation methods
4. Enter interactive mode for custom queries

### Using the Recommendation System in Your Code

```python
from recommendation_system import MovieRecommendationSystem

# Initialize the system
recommender = MovieRecommendationSystem()

# Optionally load your own data
# recommender.load_data('movies.csv', 'ratings.csv')

# Get collaborative filtering recommendations
recommendations = recommender.get_collaborative_recommendations(user_id=1, n_recommendations=5)

# Get content-based recommendations
similar_movies = recommender.get_content_based_recommendations('The Dark Knight', n_recommendations=5)

# Get hybrid recommendations
hybrid_recs = recommender.get_hybrid_recommendations(user_id=1, movie_title='The Matrix', n_recommendations=5)

# Add a new rating
recommender.add_rating(user_id=1, movie_id=1, rating=5)
```

## Data Format

### Movies CSV Format
The movies CSV file should have the following columns:
- `movieId`: Unique identifier for the movie
- `title`: Title of the movie
- `genres`: Genres of the movie (pipe-separated, e.g., "Action|Adventure|Drama")

### Ratings CSV Format
The ratings CSV file should have the following columns:
- `userId`: Unique identifier for the user
- `movieId`: Unique identifier for the movie
- `rating`: Rating given by the user (typically 1-5)

## Example Data

The system includes sample data with 15 popular movies and ratings from 5 users. You can replace this with your own dataset by providing CSV files when initializing the system.

## Algorithms

### Collaborative Filtering
- Uses cosine similarity to find users with similar preferences
- Recommends movies that similar users have rated highly
- Works well when you have sufficient user interaction data

### Content-Based Filtering
- Uses TF-IDF vectorization on movie titles and genres
- Calculates cosine similarity between movies
- Recommends movies similar to a given movie based on content features

### Hybrid Approach
- Combines collaborative and content-based recommendations
- Removes duplicates and returns the top recommendations
- Provides more diverse and accurate recommendations

## Project Structure

```
movies_recommendation_system/
├── recommendation_system.py    # Main recommendation system class
├── main.py                     # Example usage and interactive mode
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Requirements

- Python 3.7+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Future Improvements

- Add more sophisticated recommendation algorithms (SVD, matrix factorization)
- Implement user-based and item-based collaborative filtering options
- Add support for real-time recommendations
- Create a web interface for the recommendation system
- Add movie poster and description support
- Implement rating prediction functionality

## Author

Created as a Python project for GitHub.

## Acknowledgments

- Sample data includes popular movies from various genres
- Algorithms inspired by common recommendation system practices

