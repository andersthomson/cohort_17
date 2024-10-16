from utils import load_datasets
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# from sklearn.cluster import DBSCAN
from pandas.core.frame import DataFrame as DataFrame
import seaborn as sns


def main():
    # Load the three data sets
    dfs: list[DataFrame] = load_datasets()

    # Merge the datasets using 'id_audit' as the common key
    df_merged = pd.merge(dfs[0], dfs[1], on="id_audit", how="inner")  # First merge temp and power
    df_merged = pd.merge(df_merged, dfs[2], on="id_audit", how="inner")  # Then merge with radio

    # df = df_merged
    # Take a random sample of the data
    df = df_merged.sample(n=20000, random_state=42)

    # Encode categorical features (branch_header, field_x)
    label_encoder = LabelEncoder()
    df['branch_header_encoded'] = label_encoder.fit_transform(df['branch_header'])
    df['field_x_encoded'] = label_encoder.fit_transform(df['field_x'])

    # Select numeric columns for clustering (including encoded features)
    features = ['value_x', 'id_trx_status', 'id_audit', 'branch_header_encoded', 'field_x_encoded']

    # Ensure the data is numeric and has no NaN values
    df_clustering = df[features].dropna()

    # Standardize the data (important for KMeans)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_clustering)

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_features)

    # Perform PCA for visualization (reduce to 2D for KMeans)
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(scaled_features)

    # Create the pair plot separately
    sns.pairplot(df[features + ['cluster']], hue='cluster', palette='viridis')
    plt.suptitle("Pair Plot of Clustering Results", y=1.02)
    plt.show()

    # Create the PCA/KMeans plot
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=df['cluster'], cmap='plasma', s=50)
    plt.title("KMeans Clustering with PCA (2D Projection)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.colorbar(label='Cluster Label')
    plt.show()


if __name__ == '__main__':
    main()

