
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

fichier = 'FR_youtube_trending_data.csv'

df = pd.read_csv(fichier)

print(df.head(5))


#### Créer le df : nb de vidéo uploadées par jour


df['publishedAt'] = df['publishedAt'].apply(lambda x: x[:-10])

df2 = df.groupby('publishedAt').count()
df2.plot(x = df2.index, y = 'title', kind = 'scatter')

plt.scatter(df2.index, df2['title'])


df3 = df2['title']
df3.plot()

plt.show()

print(df_nb_par_date.loc[df_nb_par_date['title']>300])

print(df2.shape)

"""
print(df.groupby('publishedAt').count(
"""

df2.to_csv("export_1.csv")

df3.to_csv("export_fr_par_jour.csv")



#### Créer le df : nb de vidéo uploadées par heure

df = pd.read_csv(fichier)


df['publishedAt'] = df['publishedAt'].apply(lambda x: x[:-1])

df_par_heure = df
df_par_heure = df_par_heure.groupby('publishedAt').count()
df_par_heure = df_par_heure['title']
df_par_heure.iloc[-100:].plot()
plt.show()
df_par_heure