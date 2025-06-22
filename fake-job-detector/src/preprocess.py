import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

def load_data():
    df = pd.read_csv('data/fake_job_postings.csv')
    df = df[['title', 'description', 'fraudulent']]
    df = df.dropna()
    return df

def get_features_and_labels(df):
    # Split real and fake jobs
    real = df[df['fraudulent'] == 0]
    fake = df[df['fraudulent'] == 1]

    # Upsample fake jobs to match real jobs
    fake_upsampled = resample(fake,
                              replace=True,
                              n_samples=len(real),
                              random_state=42)

    # Combine and shuffle
    df_balanced = pd.concat([real, fake_upsampled])
    df_balanced = df_balanced.sample(frac=1).reset_index(drop=True)

    X = df_balanced['title'] + " " + df_balanced['description']
    y = df_balanced['fraudulent']

    return train_test_split(X, y, test_size=0.2, random_state=42)
