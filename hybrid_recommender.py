import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gc

class HybridContentRecommender:
    """
    Kết hợp 2 phương pháp:
    1. Genre-based (như code cũ của bạn)
    2. Tag-based TF-IDF
    
    OPTIMIZED: Không tính toàn bộ similarity matrix trước
    """
    
    def __init__(self, movies_path, ratings_path, tags_path):
        self.movies = pd.read_csv(movies_path)
        self.ratings = pd.read_csv(ratings_path)
        self.tags = pd.read_csv(tags_path)
        
        # Matrices - CHỈ LƯU features, KHÔNG lưu similarity matrix
        self.genre_matrix = None
        self.tfidf_matrix = None
        self.genre_cols = None
        
    def prepare_genre_features(self):
        """
        PHẦN 1: Genre-based (code cũ của bạn)
        """
        print("Building Genre-based features...")
        
        # Split genres
        self.movies['genres_list'] = self.movies['genres'].str.split('|')
        
        # One-hot encoding
        for idx, row in self.movies.iterrows():
            if isinstance(row['genres_list'], list):
                for genre in row['genres_list']:
                    if genre != "(no genres listed)":
                        self.movies.at[idx, genre] = 1.0
        
        # Fill NaN
        self.genre_cols = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                           'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir',
                           'Horror', 'IMAX', 'Musical', 'Mystery', 'Romance',
                           'Sci-Fi', 'Thriller', 'War', 'Western']
        
        for col in self.genre_cols:
            if col not in self.movies.columns:
                self.movies[col] = 0.0
            else:
                self.movies[col] = self.movies[col].fillna(0.0)
        
        # Genre matrix
        self.genre_matrix = self.movies[self.genre_cols].values
        
        print(f"Genre matrix shape: {self.genre_matrix.shape}")
        print(f"Memory: {self.genre_matrix.nbytes / 1024 / 1024:.2f} MB")
        
        # KHÔNG tính similarity matrix toàn bộ
        print("Note: Similarity will be computed on-demand to save memory")
        
        return self.genre_cols
    
    def prepare_tag_features(self):
        """
        PHẦN 2: Tag-based TF-IDF
        """
        print("\nBuilding Tag-based features...")
        
        # Clean tags
        print("Cleaning tags...")
        self.tags['tag_clean'] = (
            self.tags['tag']
            .str.lower()
            .str.replace(r'[^a-z0-9\s]', '', regex=True)
            .str.strip()
        )
        
        # Filter tags
        print("Filtering noisy tags...")
        tag_counts = self.tags['tag_clean'].value_counts()
        valid_tags = tag_counts[tag_counts >= 3].index
        self.tags = self.tags[self.tags['tag_clean'].isin(valid_tags)]
        
        # Aggregate tags by movie
        print("Aggregating tags by movie...")
        movie_tags = self.tags.groupby('movieId')['tag_clean'].apply(
            lambda x: ' '.join(x)
        ).reset_index()
        movie_tags.columns = ['movieId', 'tags']
        
        # Merge
        self.movies = self.movies.merge(movie_tags, on='movieId', how='left')
        self.movies['tags'] = self.movies['tags'].fillna('')
        
        # Prepare genres for TF-IDF
        self.movies['genres_clean'] = (
            self.movies['genres']
            .str.replace('|', ' ')
            .str.lower()
            .str.replace('-', '')
        )
        
        # Combine: tags (weight=3) + genres (weight=2)
        print("Creating combined features...")
        def combine_features(row):
            features = []
            if row['tags']:
                features.extend([row['tags']] * 3)
            if row['genres_clean']:
                features.extend([row['genres_clean']] * 2)
            return ' '.join(features) if features else ''
        
        self.movies['combined_features'] = self.movies.apply(combine_features, axis=1)
        
        # TF-IDF
        print("Computing TF-IDF...")
        tfidf = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.7,
            stop_words='english'
        )
        
        self.tfidf_matrix = tfidf.fit_transform(self.movies['combined_features'])
        
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"Memory: {self.tfidf_matrix.data.nbytes / 1024 / 1024:.2f} MB (sparse)")
        
        # Tag coverage
        has_tags = (self.movies['tags'] != '').sum()
        coverage = has_tags / len(self.movies) * 100
        print(f"Tag coverage: {coverage:.1f}% ({has_tags}/{len(self.movies)})")
        
        # Clean up
        gc.collect()
        
        return coverage
    
    def get_adaptive_weights(self, movie_idx):
        """
        Tính trọng số động dựa trên tag availability
        """
        movie = self.movies.iloc[movie_idx]
        tag_count = len(movie['tags'].split()) if movie['tags'] else 0
        
        if tag_count >= 10:
            tag_weight = 0.7
            genre_weight = 0.3
        elif tag_count >= 3:
            tag_weight = 0.3 + (tag_count - 3) * (0.4 / 7)
            genre_weight = 1 - tag_weight
        else:
            tag_weight = 0.2
            genre_weight = 0.8
        
        return genre_weight, tag_weight
    
    def compute_similarity_for_movie(self, movie_idx):
        """
        TÍNH SIMILARITY CHO 1 PHIM (on-demand)
        Không tính toàn bộ matrix
        """
        # Genre similarity cho 1 phim
        genre_vec = self.genre_matrix[movie_idx:movie_idx+1]
        genre_scores = cosine_similarity(genre_vec, self.genre_matrix)[0]
        
        # Tag similarity cho 1 phim
        tag_vec = self.tfidf_matrix[movie_idx:movie_idx+1]
        tag_scores = cosine_similarity(tag_vec, self.tfidf_matrix)[0]
        
        # Adaptive weights
        genre_weight, tag_weight = self.get_adaptive_weights(movie_idx)
        
        # Hybrid
        hybrid_scores = genre_weight * genre_scores + tag_weight * tag_scores
        
        return hybrid_scores, genre_weight, tag_weight
    
    def recommend_similar_movies(self, title, top_n=10):
        """
        Gợi ý phim tương tự (content-based thuần)
        """
        try:
            # Tìm movie (xử lý cả trường hợp có/không có năm)
            title_clean = title.strip()
            matches = self.movies[
                self.movies['title'].str.contains(title_clean, case=False, na=False, regex=False)
            ]
            
            if len(matches) == 0:
                return f"Movie '{title}' not found."
            
            idx = matches.index[0]
            movie_title = self.movies.iloc[idx]['title']
            
            # Tính similarity (chỉ cho phim này)
            print(f"Computing similarities for '{movie_title}'...")
            hybrid_scores, g_weight, t_weight = self.compute_similarity_for_movie(idx)
            
            # Sort
            sim_indices = hybrid_scores.argsort()[::-1][1:top_n+1]
            
            # Results
            results = self.movies.iloc[sim_indices][
                ['movieId', 'title', 'genres']
            ].copy()
            results['similarity'] = hybrid_scores[sim_indices]
            results['rank'] = range(1, len(results) + 1)
            
            print(f"\n{'='*70}")
            print(f"RECOMMENDATIONS FOR: {movie_title}")
            print(f"{'='*70}")
            print(f"Weights: Genre={g_weight:.2f}, Tag={t_weight:.2f}")
            print(f"(Adaptive based on tag availability)\n")
            
            return results
            
        except IndexError:
            return f"Movie '{title}' not found."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def build_user_profile_genre(self, user_id):
        """
        User profile từ genres
        """
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return None
        
        # Merge với movies
        user_profile = user_ratings.merge(
            self.movies[['movieId'] + self.genre_cols], 
            on='movieId', 
            how='left'
        )
        
        # Weighted genres
        for genre in self.genre_cols:
            if genre in user_profile.columns:
                user_profile[genre] = user_profile[genre] * user_profile['rating']
        
        # Average
        genre_sums = user_profile[self.genre_cols].sum()
        total = genre_sums.sum()
        
        if total > 0:
            genre_profile = genre_sums / total
        else:
            genre_profile = pd.Series(0, index=self.genre_cols)
        
        return genre_profile.values
    
    def build_user_profile_tag(self, user_id):
        """
        User profile từ tags (TF-IDF)
        """
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        
        if len(user_ratings) == 0:
            return None
        
        # Lấy phim rating cao
        high_rated = user_ratings[user_ratings['rating'] >= 4.0]
        
        if len(high_rated) == 0:
            high_rated = user_ratings.nlargest(min(10, len(user_ratings)), 'rating')
        
        # Tính weighted average TF-IDF vector
        user_vector = np.zeros(self.tfidf_matrix.shape[1])
        total_weight = 0
        
        for _, row in high_rated.iterrows():
            movie_id = row['movieId']
            rating = row['rating']
            
            # Find index
            movie_idx = self.movies[self.movies['movieId'] == movie_id].index
            
            if len(movie_idx) > 0:
                idx = movie_idx[0]
                user_vector += self.tfidf_matrix[idx].toarray()[0] * rating
                total_weight += rating
        
        if total_weight > 0:
            user_vector = user_vector / total_weight
        
        return user_vector
    
    def recommend_for_user(self, user_id, top_n=10, exclude_watched=True):
        """
        HYBRID RECOMMENDATION cho user
        """
        print(f"\nGenerating recommendations for User {user_id}...")
        
        # Build profiles
        genre_profile = self.build_user_profile_genre(user_id)
        tag_profile = self.build_user_profile_tag(user_id)
        
        if genre_profile is None:
            return "User not found or has no ratings."
        
        # Genre-based scores
        print("Computing genre-based scores...")
        genre_scores = self.genre_matrix.dot(genre_profile)
        
        # Tag-based scores
        print("Computing tag-based scores...")
        if tag_profile is not None:
            tag_scores = cosine_similarity([tag_profile], self.tfidf_matrix)[0]
        else:
            tag_scores = np.zeros(len(self.movies))
        
        # Adaptive weights
        user_ratings = self.ratings[self.ratings['userId'] == user_id]
        watched_movies = self.movies[self.movies['movieId'].isin(user_ratings['movieId'])]
        avg_tag_count = watched_movies['tags'].apply(
            lambda x: len(x.split()) if x else 0
        ).mean()
        
        if avg_tag_count >= 5:
            genre_weight, tag_weight = 0.3, 0.7
        elif avg_tag_count >= 2:
            genre_weight, tag_weight = 0.5, 0.5
        else:
            genre_weight, tag_weight = 0.7, 0.3
        
        # Hybrid scores
        hybrid_scores = genre_weight * genre_scores + tag_weight * tag_scores
        
        # Exclude watched
        if exclude_watched:
            watched_indices = self.movies[
                self.movies['movieId'].isin(user_ratings['movieId'])
            ].index
            hybrid_scores[watched_indices] = -1
        
        # Top N
        top_indices = hybrid_scores.argsort()[::-1][:top_n]
        
        # Results
        results = self.movies.iloc[top_indices][
            ['movieId', 'title', 'genres']
        ].copy()
        results['score'] = hybrid_scores[top_indices]
        results['rank'] = range(1, len(results) + 1)
        
        print(f"\n{'='*70}")
        print(f"RECOMMENDATIONS FOR USER {user_id}")
        print(f"{'='*70}")
        print(f"Weights: Genre={genre_weight:.2f}, Tag={tag_weight:.2f}")
        print(f"(Based on avg {avg_tag_count:.1f} tags/movie in user history)\n")
        
        return results


# ===== USAGE =====
if __name__ == "__main__":
    import time
    
    print("="*70)
    print("HYBRID CONTENT-BASED RECOMMENDATION SYSTEM")
    print("Optimized for large datasets (87K+ movies)")
    print("="*70)
    
    # Initialize
    start = time.time()
    recommender = HybridContentRecommender(
        movies_path="movies.csv",
        ratings_path="ratings.csv",
        tags_path="tags.csv"
    )
    
    # Build features
    print("\nStep 1: Building Genre features...")
    recommender.prepare_genre_features()
    
    print("\nStep 2: Building Tag features...")
    recommender.prepare_tag_features()
    
    build_time = time.time() - start
    print(f"\nTotal build time: {build_time:.2f} seconds")
    
    # Test 1: Movie similarity
    print("\n" + "="*70)
    print("TEST 1: SIMILAR MOVIES")
    print("="*70)
    
    similar = recommender.recommend_similar_movies("Toy Story", top_n=10)
    print(similar.to_string(index=False))
    
    # Test 2: User recommendations
    print("\n" + "="*70)
    print("TEST 2: USER RECOMMENDATIONS")
    print("="*70)
    
    user_recs = recommender.recommend_for_user(user_id=1, top_n=10)
    print(user_recs.to_string(index=False))
    
    print("\n" + "="*70)
    print("System ready! Memory-efficient design.")
    print("="*70)
    