import pandas as pd
from sklearn.model_selection import train_test_split

MoodyLyric_path = 'MoodyLyrics4Q_cleaned.csv'

df = pd.read_csv(MoodyLyric_path)

print(df.head())

train, test = train_test_split(df, test_size=0.3, random_state=100, shuffle=True, stratify=df['mood'])
test, val = train_test_split(test, test_size=0.5, random_state=100, shuffle=True, stratify=test['mood'])

print(train.shape)
print(train['mood'].value_counts())

print(test.shape)
print(test['mood'].value_counts())

print(val.shape)
print(val['mood'].value_counts())

train['split'] = 'train'
test['split'] = 'test'
val['split'] = 'val'

df = pd.concat([train, test, val])

#save to csv
df.to_csv('MoodyLyrics4Q_cleaned_split.csv', index=False)