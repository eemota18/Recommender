import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_name(id):
	try:
		name = artist_index[artist_index.id == id]["name"].values[0]
		return name
	except:
		return ""
def combine_feature(row):
	try:
		return  str(row['genre'])+ " "+ str(row['played'])+ " " + str(row['artistID'])
	except:
		print("ERROR",row)


# Read files and extract the corresponding columns
df = pd.read_table("artist_played.dat",sep='\t',delim_whitespace=True, names=('userID', 'artistID', 'played'))
df2 = pd.read_table('clean_tagged.txt',sep='\t',delim_whitespace=True, names=('userID', 'artistID', 'genre'))
df2.drop_duplicates(subset=['userID','artistID'], keep='first', inplace=True)


# merge the tables
new_df = pd.merge(df2,df,on=['userID','artistID'])
print(new_df.head())

# get the artist ids
artist_index = pd.read_csv('artists.dat',sep='\t')
artist_index = artist_index.drop(columns=['url','pictureURL'])
print(artist_index.head())

features = ['artistID','genre','played']

# fill missing data
for feat in features:
	new_df[feat] = new_df[feat].fillna('')
new_df['new_features']= new_df.apply(combine_feature,axis = 1)

# ccreate count matrix using the new featues
cv = CountVectorizer()
count_matrix = cv.fit_transform(new_df['new_features'])
cosine_sim = cosine_similarity(count_matrix)

# random user to test data
test_user = 13883
recommended = list(enumerate(cosine_sim[test_user]))

final_recco = sorted(recommended,key=lambda x:x[1],reverse=True)
i = 0

# print top 10 reccomendations
for artist in final_recco:
	name = get_name(artist[0])
	if name != "":
		print(name)
		i = i + 1
	if i > 10:
		break