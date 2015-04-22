import json
import numpy as np

class Indexer:
    """
    """
    def __init__(self):
        """
        """
        self.reviews = list()

    def read_file(self, filename):
        """
        """
        f = open(filename)
        data = f.read()
        self.reviews = json.loads(data)

    def get_mappings(self):
        """
        """
        user_dict = dict()
        movie_dict = dict()
        for review in self.reviews:
            user = review['user']
            if user not in user_dict:
                nu = len(user_dict.keys())
                user_dict[user] = nu
            movie = review['movie']
            if movie not in movie_dict:
                nm = len(movie_dict.keys())
                movie_dict[movie] = nm
        nu = len(user_dict.keys())
        user_list = [''] * nu
        for user in user_dict:
            idx = user_dict[user]
            user_list[idx] = user
        nm = len(movie_dict.keys())
        movie_list = [''] * nm
        for movie in movie_dict:
            idx = movie_dict[movie]
            movie_list[idx] = movie 
        rating_matrix = np.zeros((nu, nm))
        for review in self.reviews:
            user = review['user']
            movie = review['movie']
            u_idx = user_dict[user]
            m_idx = movie_dict[movie]
            rating_matrix[u_idx][m_idx] = review['rating']
        return (user_list, movie_list, rating_matrix)
