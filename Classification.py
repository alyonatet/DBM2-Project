import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

import seaborn as sns
import matplotlib.pyplot as plt


###CLEANING AND PREPROCESSING
data = pd.read_csv('music_dataset.csv', header='infer')

#Drop unnecessary columns
columns_drop = ['instance_id', 'track_name', 'key', 'obtained_date', 'artist_name']  # popularity #duration_ms
rows_drop = ['Rap', 'Alternative']

# Drop rows based on conditions
data = data[~data['music_genre'].isin(rows_drop)]
data = data.drop(columns=columns_drop)
print(data.head(10))

#Replace '?' with NaN
data = data.applymap(lambda cell: np.NaN if str(cell) == "?" else cell)

#Replace missing values with NaN
data = data.fillna(np.NaN)

#Replace mode with binary numbers
data['mode'] = data['mode'].replace({'Minor': 0.0, 'Major': 1.0})

#Count missing values for every column in our dataset
NaN_values = data.isna().sum()
print("Missing values per column:")
print(NaN_values)

NaN_rows = data[data.isna().all(axis=1)]    #Check to see if they are all on the same row
print(NaN_rows)             

data = data.dropna(how='all')           #Drop rows with all NaN values

data = data.drop(columns='tempo')       #Lots of missing values, drop this column

#BOXPLOTS TO SEE OUTLIERS ETC..
# numerical_columns = data.select_dtypes(include=['float64']).columns

# plt.figure(figsize=(15, 10))
# num_numerical_columns = len(numerical_columns)
# num_rows = (num_numerical_columns + 1) // 2  #Number of rows needed

# for i, column in enumerate(numerical_columns, start=1):
#     plt.subplot(num_rows, 2, i)
#     sns.boxplot(x=data[column].values)
#     plt.title(f'Boxplot of {column}')

# plt.tight_layout()
# plt.show()


#OUTLIERS
numerical_columns = data.select_dtypes(include=['float64']).columns

#Calculate Z-scores for each numerical column
z_scores = np.abs(stats.zscore(data[numerical_columns]))

threshold = 3

#Find and remove rows with outliers
data_no_outliers = data[(z_scores < threshold).all(axis=1)]

# Print the shape before and after removing outliers
print("\n")
print("Size of dataset before removing outliers:", data.shape)
print("Size of dataset after removing outliers:", data_no_outliers.shape)


#NORMALIZATION
#Extract numerical columns for normalization
numerical_columns = data.select_dtypes(include=['float64']).columns

scaler = MinMaxScaler()

#Fit and transform the numerical columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

data = data.reset_index(drop=True)
print(data.head(10))        #Print first 10 rows


#Encode the artist name to be able to use it in the classifier
# label_encoder = LabelEncoder()
# data['artist_name_encoded'] = label_encoder.fit_transform(data['artist_name'])


###CLASSIFICATION
#Separate features and target variable (class)
X = data.drop(columns=['music_genre'])  #Drop 'music_genre' and the original 'artist_name' (now encoded)
Y = data['music_genre']     #class

#Check for imbalanced classes
class_distribution = data['music_genre'].value_counts()
print("Class Distribution:\n", class_distribution)

#Calculate class weights
class_weights = dict(1 / class_distribution)

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#RANDOM FOREST CLASSIFIER
clf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2, class_weight=class_weights)

#Train the classifier
clf.fit(X_train, Y_train)

#Make predictions on the test set
Y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(Y_test, Y_pred)
conf_matrix = confusion_matrix(Y_test, Y_pred)

# Cross-validation
scores = cross_val_score(clf, X, Y, cv=5)
print(f"Cross-validated accuracy: {np.mean(scores):.2f}")

# Print statements
print(f"Accuracy: {accuracy:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")

#FEATURE IMPORTANCE
feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': clf.feature_importances_})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
print(feature_importances)


###VISUALIZATION

#CORRELATION MATRIX
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.subplots_adjust(bottom=0.2)
plt.title("Correlation Heatmap", fontsize=14)
plt.xticks(fontsize=8) 
plt.yticks(fontsize=8)  
plt.show()

#TARGET CLASSES, DISTRIBUTION
plt.figure(figsize=(8, 6))
data['music_genre'].value_counts().plot(kind='bar', color="#98BF64")
plt.title("Distribution of Target Classes", fontsize=14)
plt.xticks(fontsize=5) 
plt.yticks(fontsize=5)  
plt.xlabel("Music Genre", fontsize=9)
plt.ylabel("Count", fontsize=9)
plt.show()

# CONFUSION MATRIX
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap="Greens", fmt="d", cbar=False)
plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted Label", fontsize=9) 
plt.ylabel("True Label", fontsize=9)
plt.show()

# FEATURE IMPORTANCE
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importances, palette="crest")
plt.title("Feature Importance", fontsize=14)
plt.xticks(fontsize=5) 
plt.yticks(fontsize=5)  
plt.xlabel("Importance", fontsize=9)  
plt.ylabel("Feature", fontsize=9) 
plt.show()