from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def fasta_to_kmers(sequence, k=3):
    return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

def train_word2vec(sequences, k=3, embedding_size=100, window=5, min_count=1):
    kmers = [fasta_to_kmers(seq, k) for seq in sequences]
    model = Word2Vec(sentences=kmers, vector_size=embedding_size, window=window, min_count=min_count, workers=4)
    model.train(kmers, total_examples=len(kmers), epochs=10)
    return model

def sequence_to_vector(word2vec_model, sequence, k=3):
    kmers = fasta_to_kmers(sequence, k)
    vectors = [word2vec_model.wv[kmer] for kmer in kmers if kmer in word2vec_model.wv]
    return np.mean(vectors, axis=0)

# Sample list of FASTA sequences
sequences = ["ATCGTAGC", "ATCGGCTA", "TAGCTAGC", "GCTAGCTA"]

# Train the Word2Vec model
word2vec_model = train_word2vec(sequences, k=3, embedding_size=100, window=5, min_count=1)

# Get the vector representations for all sequences
vector_representations = [sequence_to_vector(word2vec_model, seq, k=3) for seq in sequences]

# Reduce the dimensionality to 3D using PCA
pca = PCA(n_components=3)
vectors_3d = pca.fit_transform(vector_representations)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for vector in vectors_3d:
    ax.scatter(vector[0], vector[1], vector[2], c='r', marker='o')
ax.set_xlabel('Dimension 1')
ax.set_ylabel('Dimension 2')
ax.set_zlabel('Dimension 3')
plt.show()
