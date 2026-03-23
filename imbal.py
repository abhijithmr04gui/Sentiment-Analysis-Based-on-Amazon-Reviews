import pandas as pd

df = pd.read_csv('amazon_alexa.tsv',sep = "\t")
print(df['feedback'].value_counts(normalize=True))