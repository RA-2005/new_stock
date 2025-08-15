import pandas as pd

# Upload CSV (Google Colab specific)
from google.colab import files
uploaded = files.upload()

# Load into DataFrame
df = pd.read_csv('your_file.csv')df