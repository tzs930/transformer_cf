import numpy as np
import pandas as pd
import copy 
from tqdm import tqdm

def pad_list(input_list, padded_len=128):
    padding = []
    input_list = list(input_list)
    if len(input_list) < padded_len:
        padding = [0] * (padded_len - len(input_list))
    output_list = input_list + padding
    output_list = output_list[:padded_len]
    return output_list

def main():
    tags = pd.read_csv('ml-20m/genome-tags.csv')
    scores = pd.read_csv('ml-20m/genome-scores.csv')
    movies = pd.read_csv('ml-20m/movies.csv')

    # f = open('Hybrid/feature_extraction/Genome/movie_genomes.npy')
    # A = np.load('ml-20m/movie_genomes.npy', allow_pickle=True).item()
    # Asortedidx = (-A).argsort(axis=-1)
    rating_thres = 3.5
    user_ratings = pd.read_csv('ml-20m/ratings.csv')
    user_ratings_above_thres = user_ratings[user_ratings['rating'] > rating_thres][:1000000]

    user_id_list = list(set(user_ratings_above_thres.userId))
    # max_user_id = len(user_id_list)
    num_total_user = len(user_id_list)
    num_items_list = []
    user_movie_list = []

    for uid in tqdm(user_id_list, desc='Preprocess'):
        # if uid not in user_movie_dict.keys() :
        rating_list = user_ratings_above_thres[ user_ratings_above_thres['userId'] == uid ].movieId.tolist()
        if len(rating_list) > 0:
            num_items_list.append(len(rating_list))
        # user_movie_dict[uid] = rating_list
            user_movie_list.append(rating_list)

    # user_movie_dict = np.load('transformer_cf/rating_dict.npy', allow_pickle=True).item()
    max_num_movies_per_user = np.max(num_items_list)
    print(user_movie_list[0])
    movie_ids = list(set(user_ratings_above_thres.movieId.tolist()))
    movie_ids.insert(0, 0)      # Insert padding item
    num_total_movies = len(movie_ids)
    print("- num_max_movies_per_user : ", max_num_movies_per_user)
    print("- num_total_unique_movies : ", num_total_movies)

    movie_ids_to_idx = {}
    for i, mid in enumerate(movie_ids):
        movie_ids_to_idx[mid] = i

    padded_user_movie_list = []
    sparse_matrix = []

    # for i in tqdm(range(len(user_movie_list)), desc='Construct'):
    #     movie_list = []
    #     for idx in range(len(user_movie_list[i])):
    #         movie_list.append(movie_ids_to_idx[user_movie_list[i][idx]])
        
    #     padded_user_movie_list.append(pad_list(movie_list, max_num_movies_per_user))
        # sparse_vec = np.zeros(num_total_movies)
        # sparse_vec[np.array(movie_list)] = 1
        # sparse_matrix.append(sparse_vec)
    
    padded_user_movie_list = np.array(padded_user_movie_list)
    sparse_matrix = np.array(sparse_matrix)
    save_dict = {'padded_list': padded_user_movie_list, 'movie_ids_to_idx': movie_ids_to_idx, #'sparse_matrix': sparse_matrix, 
                 'num_movies': num_total_movies, 'num_users' : num_total_user }
    
    print(padded_user_movie_list[0])
    print(sparse_matrix[0])

    # np.save('padded_list.npy', padded_user_movie_list)
    # np.save('sparse_matrix.npy', sparse_matrix)
    np.save('save_dict.npy', save_dict)
    # np.save('rating_dict.npy', user_movie_dict)

    valid_num_user = 10000
    test_num_user = 10000

    np.random.seed(98765)
    idx_perm = np.random.permutation(num_total_user)

    train_idx = idx_perm[:(valid_num_user + test_num_user)]
    valid_idx = idx_perm[valid_num_user:test_num_user]
    test_idx = idx_perm[test_num_user:]

    valid_item_mask_prob = 0.2
    valid_item_num = num_total_movies * valid_item_mask_prob    

    test_item_mask_prob = 0.2
    test_item_num = num_total_movies * test_item_mask_prob

    # train_num_user = num_total_user - (valid_num_user + test_num_user)

    total_idx = np.arange(num_total_user)
    split_idx = np.random.choice(total_idx, valid_num_user + test_num_user)
    np.random.shuffle(split_idx)

    return user_movie_list

def load_npy():
    user_movie_dict = np.load('rating_dict.npy', allow_pickle=True).item()
    return user_movie_dict

if __name__ == "__main__":
    main()
