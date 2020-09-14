import numpy as np
import numpy.linalg as LA
from tqdm import tqdm 

from scipy import stats
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist


class Cluster(object):
    cluster_name = 0
    def __init__(self, recent_vectors_size=50, sampled_vectors_size=10):
        self.recent_vectors_size = recent_vectors_size    
        self.sampled_vectors_size = sampled_vectors_size    
        
        self.recent_vectors = deque(maxlen=self.recent_vectors_size)
        self.sampled_vectors = deque(maxlen=self.sampled_vectors_size)
        self.age = 0
        Cluster.cluster_name += 1
        self.cluster_name = Cluster.cluster_name

    def add_vector(self, vector:np.array):
        if len(self.sampled_vectors) < self.sampled_vectors_size:
            self.sampled_vectors.append(vector)
        else:
            self.recent_vectors.append(vector)
            if np.random.random() < 1. / self.age:
                position = np.random.randint(0, self.sampled_vectors_size)
                self.sampled_vectors[position] = vector
        self.age += 1

    def get_embeddings(self) -> np.array:
        if len(self.recent_vectors) < self.recent_vectors_size:
            return np.array(self.sampled_vectors)
        else:
            return np.vstack((self.recent_vectors, self.sampled_vectors))
    
    def get_name(self) -> str:
        return f"{self.cluster_name}"

    def __len__(self) -> int:
        return len(self.recent_vectors) + len(self.sampled_vectors)


class OnlineClustering(object):
    def __init__(self, threshold:float=0.7, clusters_size=50, sample_size=10, limit_num_clusters:int=None):
        self.threshold = threshold
        self.samples_per_cluster = clusters_size
        self.sample_size = sample_size
        self.limit_num_clusters = limit_num_clusters # limits the search to the first `limit_num_clusters` clusters
        self.clusters = list()

    def _make_new_cluster(self, new_vector:np.array):
        new_cluster = Cluster(recent_vectors_size=self.samples_per_cluster, sampled_vectors_size=self.sample_size)
        new_cluster.add_vector(new_vector)
        self.clusters.append(new_cluster)
        return new_cluster.get_name()

    def update_predict(self, new_vector:np.array):
        if len(self.clusters) == 0:
            cur_pred = self._make_new_cluster(new_vector)     
        else:
            # Make prediction
            max_similarity = -1
            cur_pred = 0
            cur_pred_idx = 0
            search_clusters = self.clusters if self.limit_num_clusters is None else self.clusters[:self.limit_num_clusters]
            for idx, cluster in enumerate(search_clusters):
                similarity = 1-np.mean(cdist(cluster.get_embeddings(), new_vector[None, :], metric='cosine'))
                if similarity > max_similarity:
                    max_similarity = similarity
                    cur_pred = cluster.get_name()
                    cur_pred_idx = idx
                print(similarity)
            # new cluster
            if max_similarity < self.threshold:
                cur_pred = self._make_new_cluster(new_vector)
            else:
                # update cluster with new vector
                self.clusters[cur_pred_idx].add_vector(new_vector)
        return cur_pred

    def __str__(self):
        out = ""
        for idx, cluster in enumerate(self.clusters):
            out += f"Cluster {idx} has {len(cluster)} vectors\n"
        return out[:-1]

    def __len__(self):
        return len(self.clusters)