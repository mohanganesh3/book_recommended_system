# Book Recommendation System 📚✨

Welcome to the Book Recommendation System repository! This project uses a collaborative filtering approach to recommend books based on user interactions and preferences, such as ratings or favorites.

# 🚀 Key Features

	•	Collaborative Filtering: Recommends books based on interactions from users with similar reading habits.
	•	Popular Book Recommendations: Suggests popular books based on preprocessed data.
	•	Interactive Web Interface: A user-friendly interface for getting book recommendations.

# 📂 Repository Structure
📦book_recommended_system
 ┣ 📂templates
 ┃ ┣ 📜index.html          # Home page showing popular book recommendations
 ┃ ┗ 📜recommend.html      # Page displaying personalized book recommendations
 ┣ 📜README.md             # Project documentation
 ┣ 📜app.py                # Flask web application for the recommendation system
 ┣ 📜book_recommended_system.ipynb  # Jupyter notebook for developing and testing the model
 ┣ 📜books.pkl             # Preprocessed book data
 ┣ 📜popular.pkl           # Preprocessed data for popular book recommendations
 ┣ 📜pt.pkl                # Pivot table of user-book interactions
 ┣ 📜similarity_scores.pkl  # Precomputed similarity scores matrix
 ┗ 📜requirements.txt      # Python dependencies and libraries
 
# 🧠 How It Works (Collaborative Filtering)

This system uses collaborative filtering, a technique that makes recommendations based on users’ behaviors and interactions. The primary approach involves:

	1.	User-Item Matrix:
	•	The system uses a pivot table (pt.pkl) containing user-book interactions. Each row represents a user, and each column represents a book, with values indicating whether a user has interacted with a book (e.g., rated it or marked it as a favorite).
	2.	Similarity Computation:
	•	Based on this matrix, the system computes similarity scores between users using techniques like cosine similarity or Pearson correlation. These scores are stored in the similarity_scores.pkl file.
	3.	Recommendation Generation:
	•	Once the similarity between users is known, the system identifies books liked by users with similar preferences and recommends those books to you.
	4.	Popular Books:
	•	In addition to collaborative filtering, the system also provides a list of popular books using preprocessed data from popular.pkl.

# 🔧 Usage

	1.	Web Interface:
	•	Run the Flask app (app.py) and visit http://127.0.0.1:5000/ in your browser.
	•	The homepage (index.html) shows popular books.
	•	For personalized recommendations, go to /recommend and enter your user ID.
	2.	Jupyter Notebook:
	•	Open book_recommended_system.ipynb to explore the recommendation logic and test the model interactively.
