import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#function creates a heatmap:
def create_heatmap(matrix):
    plt.figure(figsize=(48, 40))
    sns.heatmap(matrix, annot=True, cmap="coolwarm")
    plt.show()

#load data:
file_path = "Mental Health/mental-heath-in-tech-2016_20161114.csv"
raw_data = pd.read_csv(file_path)

#exploration:
pd.set_option('display.max_columns', None)
print(raw_data.head())
print(raw_data.info())
print(raw_data.describe())

#feature cleaning:
data = raw_data

#gender:
#find out every value in the feature:
gender = data['What is your gender?'].unique()

#replace the values with male/female/other:
data['What is your gender?'].replace(to_replace = ['Male', 'male', 'Male ', 'M', 'm',
       'man', 'Cis male', 'Male.', 'male 9:1 female, roughly', 'Male (cis)', 'Man', 'Sex is male',
       'cis male', 'Malr', 'Dude', "I'm a man why didn't you make this a drop down question. You should of asked sex? And I would of answered yes please. Seriously how much text can this take? ",
       'mail', 'M|', 'Male/genderqueer', 'male ',
       'Cis Male', 'Male (trans, FtM)',
       'cisdude', 'cis man', 'MALE'], value = "Male", inplace = True)
data['What is your gender?'].replace(to_replace = ['Female', 'female', 'I identify as female.', 'female ',
       'Female assigned at birth ', 'F', 'Woman', 'fm', 'f', 'Cis female ', 'Transitioned, M2F',
       'Genderfluid (born female)', 'Female or Multi-Gender Femme', 'Female ', 'woman', 'female/woman',
       'Cisgender Female', 'fem', 'Female (props for making this a freeform field, though)',
       ' Female', 'Cis-woman', 'female-bodied; no feelings about gender',
       'AFAB'], value = "Female", inplace = True)
data['What is your gender?'].replace(to_replace = ['Bigender', 'non-binary', 'Other/Transfeminine',
       'Androgynous', 'Other', 'nb masculine',
       'none of your business', 'genderqueer', 'Human', 'Genderfluid',
       'Enby', 'genderqueer woman', 'mtf', 'Queer', 'Agender', 'Fluid',
       'Nonbinary', 'human', 'Unicorn', 'Genderqueer',
       'Genderflux demi-girl', 'Transgender woman'], value = "Other", inplace = True)

#age:
#calculate the 99% quantile to get outliers and replace them with the average value

age_q1 = data['What is your age?'].quantile(0.01)
age_q2 = data['What is your age?'].quantile(0.99)
data_age = data['What is your age?']

outliers = data_age[(data_age < age_q1) | (data_age > age_q2)]

data['What is your age?'].replace(to_replace = [outliers], value = data['What is your age?'].mean(), inplace = True)

#remove data with 60% null-values:
threshold = 0.5 * len(data)
data = data.dropna(axis=1, thresh=threshold)
print(data.info())

#split the fields with "|":
data_splitted = data['Which of the following best describes your work position?'].str.get_dummies('|')
data = pd.concat([data, data_splitted.add_prefix('Work position_')], axis=1)
data.drop('Which of the following best describes your work position?', axis=1, inplace=True)

#imputation:
imputer = SimpleImputer(strategy='most_frequent')
imputed_data = imputer.fit_transform(data)
data_imputed = pd.DataFrame(imputed_data, columns=data.columns)


#label encoding:
labeled_data = data_imputed
for column in labeled_data:
    if labeled_data[column].dtype == 'object':
        labeled_data[column] = labeled_data[column].astype('category').cat.codes

#useful Data:
#gender:
males = labeled_data[labeled_data['What is your gender?'] == 1]['What is your gender?'].count()
females = labeled_data[labeled_data['What is your gender?'] == 0]['What is your gender?'].count()
others = labeled_data[labeled_data['What is your gender?'] == 2]['What is your gender?'].count()

print("Gender: Males = " + str(males) + " Females = " + str(females) + " Others = " + str(others))

#Diagram Gender disribution
sizes = [males, females, others]
labels = ['Males', 'Females', 'Others']
colors = ['lightblue', 'lightgreen', 'pink']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=False, startangle=140)


plt.title('Gender Distribution')
plt.axis('equal')
plt.show()

#age
print('Mean Age: ' + str(data['What is your age?'].mean()))


# apply variance threshold
selector = VarianceThreshold(threshold=0.4)
selector.fit_transform(labeled_data)

# show the variances per feature
variance_data = pd.DataFrame({'features': labeled_data.columns, 'variances': selector.variances_})
x = variance_data[variance_data["variances"] < 0.40]
y = variance_data[variance_data["variances"] > 500]

for value in x["features"]:
    print(value + " deleted")
    labeled_data.drop(value, axis=1, inplace=True)

for value in y["features"]:
    print(value + " deleted")
    labeled_data.drop(value, axis=1, inplace=True)

#calculate covariance-matrix to remove features with high correlation
correlation_matrix = labeled_data.corr()
threshold = 0.6

high_correlation_cols = np.where(correlation_matrix > threshold)
cols_to_drop = set()

for col1 in range(len(correlation_matrix)):
    for col2 in range(col1 + 1, len(correlation_matrix)):
        if abs(correlation_matrix.iloc[col1, col2]) > threshold:
            cols_to_drop.add(labeled_data.columns[col2])


labeled_data = labeled_data.drop(columns=cols_to_drop)

print(labeled_data.info())

#standardscaler:
column_names = labeled_data.columns
scaler = StandardScaler()
scaled_features = scaler.fit_transform(labeled_data)
scaled_data = pd.DataFrame(scaled_features, columns = column_names)

#implement PCA:
#use ellbow-technique to find the number of relavent PCs:
pca = PCA()
pca.fit(scaled_data)
explained_variance = pca.explained_variance_ratio_

plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.xlabel('Numbers of Principal Components')
plt.ylabel('Variance')
plt.title('Elbow-Technique for PCA')
plt.show()

# compute the PCA:
n_components = 10
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(scaled_data)

correlation_matrix_pca = pd.DataFrame(pca.components_, columns=scaled_data.columns)

#calculate and show the variance of the PCs:
cumulative_variance = explained_variance.cumsum()

for i, ev in enumerate(explained_variance):
    print(f"PC {i+1}: Explained Variance = {ev:.2f}, Kumulative Explained Variance = {cumulative_variance[i]:.2f}")

#show the features to their PCs to find meanfull cluster:
feature_names = scaled_data.columns
weights = pca.components_
plt.figure(figsize=(50, 40))
sns.heatmap(weights, annot=True, fmt=".2f", cmap='coolwarm', xticklabels=feature_names, yticklabels=[f"PC{i+1}" for i in range(n_components)])
plt.title('Weighted Features')
plt.show()







