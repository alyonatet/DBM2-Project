# Importing Libraries
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Read the CSV file
data = pd.read_csv('MusicDataSet.csv', header='infer')

# Drop unnecessary columns
columns_to_drop = ['instance_id', 'artist_name', 'track_name', 'obtained_date', 'key', 'duration_ms','music_genre']
data = data.drop(columns=columns_to_drop, axis=1)

# Convert 'mode' column to numeric
data['mode'] = data['mode'].map({'Major': 1, 'Minor': 0}).astype(float)

# Replace '?' with NaN
data.replace('?', np.nan, inplace=True)

# Convert columns to numeric and impute missing values with mean
data = data.apply(pd.to_numeric, errors='coerce')
data.fillna(data.mean(), inplace=True)

# Select specific columns for clustering
selected_columns = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness',
                     'mode', 'speechiness', 'tempo', 'valence', 'popularity',]
data_selected = data[selected_columns]

# Fit KMeans on the selected data
k_optimal = 11
k_means = KMeans(n_clusters=k_optimal, max_iter=50, random_state=1)
labels_optimal = k_means.fit_predict(data_selected)

# Add cluster labels to the DataFrame
data['cluster_label'] = labels_optimal

# Z-score normalization
numeric_columns = data_selected.select_dtypes(include=[np.number]).columns
z = (data_selected[numeric_columns] - data_selected[numeric_columns].mean()) / data_selected[numeric_columns].std()
z.columns = [f'z_{col}' for col in numeric_columns]

# Handling constant columns and outliers
z = z.replace([np.inf, -np.inf], np.nan)
z = z.dropna(axis=1, how='all')
z = z.dropna()

# Identify and remove outliers based on Z-scores
z_score_columns = [f'z_{col}' for col in numeric_columns]
for col in z_score_columns:
    z = z[z[col].between(-3, 3)]

# Uncomment if intend to use the normalized data (z)
'''
# Determine the optimal number of clusters using the elbow method
inertia = []
for k in range(1, 11):  # Trying different values for the number of clusters
    kmeans = KMeans(n_clusters=k, max_iter=50, random_state=1)
    kmeans.fit(z)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters (Z-score Normalized Data)')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (Inertia)')
plt.show()


# Determine the optimal number of clusters using Silhouette Score
silhouette_scores = []
for k in range(2, 16):  # Adjust the range as needed
    kmeans = KMeans(n_clusters=k, max_iter=50, random_state=1)
    kmeans.fit(data_selected)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(data_selected, labels)
    silhouette_scores.append(silhouette_avg)

# Plot the Silhouette Score curve
plt.plot(range(2, 16), silhouette_scores, marker='o')
plt.title('Silhouette Score for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()

# PCA and Visualization using original data or z-score normalized data
reduced_data = PCA(n_components=2).fit_transform(data_selected)  # Use original data or z

# Scatter plot
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels_optimal, cmap='viridis', edgecolors='k', alpha=0.7)
plt.title('K-Means Clustering Results')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Evaluate the quality of the clusters using silhouette score
silhouette_avg_optimal = silhouette_score(data_selected, labels_optimal)
print(f'Silhouette Score for Optimal Clusters: {silhouette_avg_optimal}')
'''
# Interpretation of Clusters
cluster_centers = k_means.cluster_centers_
feature_names = selected_columns  

# Create a DataFrame with cluster centers and feature names
cluster_centers_df = pd.DataFrame(cluster_centers, columns=feature_names)

# Print cluster centers for interpretation
print("Cluster Centers (Feature Averages for Each Cluster):")
print(cluster_centers_df)

# Assigning Genres based on the characteristics observed
genre_mapping = {
   0: "Rock",
   1: "Hiphop",
   2: "Jazz",
   3: "Electronic",
   4: "Anime",
   5: "Alternative",
   6: "Country",
   7: "Rap",
   8: "Blues",
   9: "Classical"
}

# Predicted labels for each instance
predicted_labels = k_means.labels_

# Create a new column 'predicted_genre' based on cluster assignments
data['music_genre'] = predicted_labels

# Map cluster labels to genres
data['music_genre'] = data['music_genre'].map(genre_mapping).fillna('Unknown')

# Print the DataFrame with cluster labels
print("Data with Cluster Labels:")
print(data[['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness',
            'mode', 'speechiness', 'tempo', 'valence', 'popularity', 'music_genre', 'cluster_label']])


reduced_data = PCA(n_components=2).fit_transform(data_selected)  # Use original data or z-score normalized data

# Scatter plot for each genre
plt.figure(figsize=(12, 8))

# Dictionary to map genres to colors

genre_colors = {
    'Rock': 'blue',
    'Hiphop': 'orange',
    'Jazz': 'green',
    'Electronic': 'red',
    'Anime': 'purple',
    'Alternative': 'brown',
    'Country': 'pink',
    'Rap': 'gray',
    'Blues': 'cyan',
    'Classical': 'lime',
    'Unknown': 'black'  
}
''''
for genre, color in genre_colors.items():
    genre_data = reduced_data[data['music_genre'] == genre]
    plt.scatter(genre_data[:, 0], genre_data[:, 1], label=genre, c=color, s=100)

plt.title('K-Means Clustering Results by Music Genre')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
'''


for genre, color in genre_colors.items():
    genre_data = reduced_data[data['music_genre'] == genre]
    plt.scatter(genre_data[:, 0], genre_data[:, 1], label=genre, c=color, s=80, alpha=0.7, edgecolors='k')

plt.title('K-Means Clustering Results with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()