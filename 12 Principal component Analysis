import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
cancer.keys()
df=pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df.head()
cancer['target_names']

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df)
scaled=scaler.transform(df)
df.head()

#PCA
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit(scaled)
x_pca=pca.transform(scaled)
x_pca.shape

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=cancer['target'],cmap='rainbow')
plt.xlabel('1st PC')
plt.ylabel('2ndPc')

pca.components_
df_comp=pd.DataFrame(pca.components_,columns=cancer['feature_names'])

plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma')

#now we can go ahead and do any reg or classification models on the x_pca data rather than the original data of 30 features.

