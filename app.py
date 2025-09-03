import pickle
import flask
import numpy as np
import pandas as pd
import warnings
import os
import json
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

app = flask.Flask(__name__)
app.secret_key = 'book_recommendation_secret_key_2023'

# Function to safely load pickle files with error handling
def load_pickle_file(filename):
    """Safely load a pickle file with comprehensive error handling"""
    try:
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            print(f"Loaded {filename} successfully")
            return data
        else:
            print(f"File {filename} not found")
            return None
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

# Load the precomputed models with error handling
popular_df_data = load_pickle_file('popular.pkl')
# Fix the DataFrame assignment to avoid ambiguous truth value error
if popular_df_data is not None:
    popular_df = popular_df_data
else:
    popular_df = pd.DataFrame()

pt_data = load_pickle_file('pt.pkl')
if pt_data is not None:
    pt = pt_data
else:
    pt = pd.DataFrame()

# Try to load books.pkl with protocol handling
books_data = load_pickle_file('books.pkl')
if books_data is not None:
    books = books_data
else:
    # Create a minimal books dataframe from pt index if possible
    if isinstance(pt, pd.DataFrame) and len(pt.index) > 0:
        books = pd.DataFrame({'Book-Title': pt.index})
        books['Book-Author'] = 'Unknown'
        books['Image-URL-M'] = 'https://via.placeholder.com/150'
        print("Created minimal books dataframe from pt index")
    else:
        books = pd.DataFrame()
        print("Failed to create books dataframe")

similarity_scores_data = load_pickle_file('similarity_scores.pkl')
if similarity_scores_data is not None:
    similarity_scores = similarity_scores_data
else:
    similarity_scores = np.array([])
    print("Failed to load similarity_scores, using empty array")

# Create a comprehensive book lookup dictionary for faster access
book_lookup = {}

# First, add all books from pt.index with default values
if isinstance(pt, pd.DataFrame) and len(pt.index) > 0:
    for title in pt.index:
        book_lookup[title] = {
            'author': 'Unknown',
            'image': 'https://via.placeholder.com/150',
            'rating': 0
        }

# Then, enrich with information from popular_df where available
# Check if popular_df is a DataFrame and not empty
if isinstance(popular_df, pd.DataFrame) and not popular_df.empty:
    for _, row in popular_df.iterrows():
        title = row['Book-Title']
        # Update the book info if it exists in our lookup
        if title in book_lookup:
            book_lookup[title].update({
                'author': row.get('Book-Author', 'Unknown'),
                'image': row.get('Image-URL-M', 'https://via.placeholder.com/150'),
                'rating': round(row.get('avg_rating', 0), 2)
            })
        else:
            # Add the book if it's not already in lookup
            book_lookup[title] = {
                'author': row.get('Book-Author', 'Unknown'),
                'image': row.get('Image-URL-M', 'https://via.placeholder.com/150'),
                'rating': round(row.get('avg_rating', 0), 2)
            }

# Finally, try to enrich with information from books dataframe where available
# Check if books is a DataFrame and not empty
if isinstance(books, pd.DataFrame) and not books.empty:
    for _, row in books.iterrows():
        title = row['Book-Title']
        # Only update if the book exists in our lookup and has unknown values
        if title in book_lookup:
            current_info = book_lookup[title]
            # Only update if current values are unknown/placeholder
            if current_info.get('author') == 'Unknown' and 'Book-Author' in row:
                book_lookup[title]['author'] = row['Book-Author']
            if current_info.get('image') == 'https://via.placeholder.com/150' and 'Image-URL-M' in row:
                book_lookup[title]['image'] = row['Image-URL-M']

print(f"Created book lookup with {len(book_lookup)} books")

# Initialize user ratings storage
user_ratings_file = 'user_ratings.json'
if not os.path.exists(user_ratings_file):
    with open(user_ratings_file, 'w') as f:
        json.dump({}, f)

def save_user_rating(book_title, rating, review=""):
    """Save user rating for a book"""
    try:
        # Load existing ratings
        if os.path.exists(user_ratings_file):
            with open(user_ratings_file, 'r') as f:
                user_ratings = json.load(f)
        else:
            user_ratings = {}
        
        # Add new rating
        if book_title not in user_ratings:
            user_ratings[book_title] = []
        
        user_ratings[book_title].append({
            'rating': rating,
            'review': review,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save ratings
        with open(user_ratings_file, 'w') as f:
            json.dump(user_ratings, f)
        
        return True
    except Exception as e:
        print(f"Error saving user rating: {e}")
        return False

def get_book_ratings(book_title):
    """Get all ratings for a book"""
    try:
        if os.path.exists(user_ratings_file):
            with open(user_ratings_file, 'r') as f:
                user_ratings = json.load(f)
            return user_ratings.get(book_title, [])
        return []
    except Exception as e:
        print(f"Error getting book ratings: {e}")
        return []

def get_average_rating(book_title):
    """Get average rating for a book"""
    ratings = get_book_ratings(book_title)
    if ratings:
        return sum(r['rating'] for r in ratings) / len(ratings)
    return 0

def get_book_suggestions(query, limit=5):
    """Get book suggestions based on partial match with book titles"""
    if isinstance(pt, pd.DataFrame) and not pt.empty:
        # Get all book titles from pt.index
        book_titles = list(pt.index)
        # Filter titles that contain the query
        suggestions = [title for title in book_titles if query.lower() in title.lower()]
        # Sort by relevance (exact match first, then partial matches)
        exact_matches = [title for title in suggestions if query.lower() == title.lower()]
        partial_matches = [title for title in suggestions if query.lower() in title.lower() and query.lower() != title.lower()]
        # Prioritize exact matches, then partial matches
        all_matches = exact_matches + partial_matches
        
        # Get detailed information for each suggestion
        detailed_suggestions = []
        for title in all_matches[:limit]:
            # Get complete book information
            book_info = get_book_info(title)
            # Format for autocomplete response
            formatted_book = {
                'title': book_info['title'],
                'author': book_info['author'],
                'image': book_info['image'],
                'rating': book_info['rating']  # This will now be a number
            }
            detailed_suggestions.append(formatted_book)
        
        return detailed_suggestions
    return []

@app.route('/')
def index():
    # Get search query if any
    search_query = flask.request.args.get('search', '').strip().lower()
    
    # Filter popular books based on search query
    if search_query and isinstance(popular_df, pd.DataFrame) and not popular_df.empty:
        filtered_df = popular_df[
            popular_df['Book-Title'].str.lower().str.contains(search_query) |
            popular_df['Book-Author'].str.lower().str.contains(search_query)
        ]
    else:
        filtered_df = popular_df
    
    if isinstance(filtered_df, pd.DataFrame) and not filtered_df.empty:
        return flask.render_template('index.html',
                               book_name=list(filtered_df['Book-Title'].values),
                               book_author=list(filtered_df['Book-Author'].values),
                               image=list(filtered_df['Image-URL-M'].values),
                               votes=list(filtered_df['num_ratings'].values),
                               rating=list(filtered_df['avg_rating'].values),
                               search_query=search_query
                               )
    else:
        return flask.render_template('index.html', 
                               book_name=[],
                               book_author=[],
                               image=[],
                               votes=[],
                               rating=[],
                               search_query=search_query
                               )

@app.route('/autocomplete')
def autocomplete():
    """API endpoint for book title autocomplete"""
    query = flask.request.args.get('q', '')
    if len(query) < 2:
        return flask.jsonify([])
    
    suggestions = get_book_suggestions(query, limit=10)
    return flask.jsonify(suggestions)

@app.route('/recommend')
def recommend_ui():
    return flask.render_template('recommend.html')

@app.route('/recommend_books', methods=['post'])
def recommend():
    user_input = flask.request.form.get('user_input')
    
    # Check if data is loaded properly
    if isinstance(pt, pd.DataFrame) and len(similarity_scores) > 0:
        try:
            # Check if user_input exists in pt.index
            if user_input not in pt.index:
                # Try to find similar book titles
                similar_titles = [title for title in pt.index if user_input.lower() in title.lower()]
                if similar_titles:
                    # Sort by relevance (exact match first, then partial matches)
                    exact_matches = [title for title in similar_titles if user_input.lower() == title.lower()]
                    partial_matches = [title for title in similar_titles if user_input.lower() in title.lower() and user_input.lower() != title.lower()]
                    
                    # Prioritize exact matches, then partial matches
                    all_matches = exact_matches + partial_matches
                    
                    if all_matches:
                        # Clean the main suggestion to remove trailing punctuation
                        main_suggestion = all_matches[0].rstrip('.,?;:')
                        error_msg = f"Book not found. Did you mean: {main_suggestion}?"
                        # Add more suggestions if available
                        if len(all_matches) > 1:
                            # Clean additional suggestions
                            additional_suggestions = [title.rstrip('.,?;:') for title in all_matches[1:min(3, len(all_matches))]]
                            error_msg += f" Or try: {', '.join(additional_suggestions)}"
                    else:
                        error_msg = "Book not found. Please try another book."
                else:
                    error_msg = "Book not found. Please try another book."
                return flask.render_template('recommend.html', error=error_msg)
            
            # Get the index of the book in pt.index
            index = np.where(pt.index == user_input)[0][0]
            
            # Check if index is within bounds of similarity_scores
            if index >= len(similarity_scores):
                return flask.render_template('recommend.html', error=f"Error generating recommendations: Data inconsistency detected. Please try another book.")
            
            # Get similar items (ensure we don't go out of bounds)
            if index < len(similarity_scores):
                similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]
            else:
                return flask.render_template('recommend.html', error=f"Error generating recommendations: Index out of bounds. Please try another book.")
            
            data = []
            limited_info_count = 0
            for i in similar_items:
                # Check if the index is valid
                if i[0] >= len(pt.index):
                    continue
                    
                book_title = pt.index[i[0]]
                # Get complete book information
                book_info = get_book_info(book_title)
                
                # Count how many books have limited information
                if book_info['author'] == 'Unknown' and book_info['rating'] == 0:
                    limited_info_count += 1
                
                # Create item list in the expected format
                item = [
                    book_info['title'],
                    book_info['author'],
                    book_info['image'],
                    book_info['rating']
                ]
                data.append(item)
            
            # Add information about data limitations
            data_quality_info = {
                'total_recommendations': len(data),
                'limited_info_count': limited_info_count,
                'limited_info_percentage': round(limited_info_count / len(data) * 100, 1) if data else 0
            }
            
            print("Recommendation data:", data)
            return flask.render_template('recommend.html', data=data, book_title=user_input, data_quality=data_quality_info)

        except Exception as e:
            return flask.render_template('recommend.html', error=f"Error generating recommendations: {str(e)}")
    else:
        return flask.render_template('recommend.html', error="Recommendation system not properly loaded. Please check the data files.")

def get_book_info(book_title):
    """Get complete book information from available data sources"""
    # Initialize with default values
    book_info = {
        'title': book_title,
        'author': 'Unknown',
        'image': 'https://via.placeholder.com/150',
        'rating': 0
    }
    
    # Try to get information from our comprehensive lookup first
    if book_title in book_lookup:
        lookup_info = book_lookup[book_title]
        book_info['author'] = lookup_info.get('author', 'Unknown')
        book_info['image'] = lookup_info.get('image', 'https://via.placeholder.com/150')
        rating = lookup_info.get('rating', 0)
        if rating > 0:
            book_info['rating'] = rating
        # If rating is 0, we keep it as 0 and handle display in template
    
    # Get user ratings if available (these should override any existing ratings)
    avg_rating = get_average_rating(book_title)
    if avg_rating > 0:
        book_info['rating'] = round(avg_rating, 2)
    
    return book_info

@app.route('/book/<book_title>')
def book_details(book_title):
    """Display book details page"""
    # Decode URL encoded book title
    import urllib.parse
    book_title = urllib.parse.unquote(book_title)
    
    # Get book details
    book_info = {}
    if isinstance(books, pd.DataFrame) and not books.empty:
        temp_df = books[books['Book-Title'] == book_title]
        if not temp_df.empty:
            book_info = temp_df.iloc[0].to_dict()
    
    # Get user ratings
    ratings = get_book_ratings(book_title)
    avg_rating = get_average_rating(book_title)
    
    return flask.render_template('book_details.html', 
                           book=book_info, 
                           ratings=ratings, 
                           avg_rating=round(avg_rating, 2) if avg_rating > 0 else 0,
                           book_title=book_title)

@app.route('/rate_book', methods=['POST'])
def rate_book():
    """Handle book rating submission"""
    book_title = flask.request.form.get('book_title')
    rating = flask.request.form.get('rating')
    review = flask.request.form.get('review', '')
    
    if book_title and rating:
        try:
            rating = int(rating)
            if 1 <= rating <= 5:
                save_user_rating(book_title, rating, review)
                flask.flash('Thank you for your rating!', 'success')
            else:
                flask.flash('Rating must be between 1 and 5', 'error')
        except ValueError:
            flask.flash('Invalid rating value', 'error')
    else:
        flask.flash('Book title and rating are required', 'error')
    
    return flask.redirect(flask.url_for('book_details', book_title=book_title))

# Vercel serverless function handler
@app.route('/<path:path>', methods=['GET', 'POST'])
def catch_all(path):
    # This will handle any other routes and redirect to the main app
    return flask.redirect('/')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5018)))