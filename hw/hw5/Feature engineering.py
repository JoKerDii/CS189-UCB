''' top important feature selection
# https://machinelearningmastery.com/feature-selection-machine-learning-python/

from sklearn.feature_selection import SelectKBest, chi2

#Feature selection using SelectKBest
test = SelectKBest(score_func=chi2, k=8)
fit = test.fit(t_train_enc, t_label_enc)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(t_train_enc)
# summarize selected features
print(features[0:5,:])
print (t_train_enc.head())

'''
'''extract words

# .strip(): prints the string by removing leading and trailing whitespaces

def get_title(dataset, feature_name):
    return dataset[feature_name].map(lambda name:name.split(',')[1].split('.')[0].strip())

'''

'''cut the range of age

age_bins = [0,15,35,45,60,200]
age_labels = ['15-','15-35','35-45','40-60','60+']
dataset['AgeRange'] = pd.cut(dataset['Age'], bins=age_bins, labels=age_labels, include_lowest=True)

'''

'''create a new col

dataset['Family'] = ''
dataset.loc[dataset['FamilySize'] == 0, 'Family'] = 'alone'
dataset.loc[(dataset['FamilySize'] > 0) & (dataset['FamilySize'] <= 3), 'Family'] = 'small'

'''

'''get_dummies(..., drop_first = True)

t_test = titanic_test.copy()
dummy_col = ['pclass','sex','embarked']
for col in dummy_col:
    dummy = pd.get_dummies(t_test[col],drop_first=True)
    t_test = pd.concat([t_test,dummy], axis = 1)    
t_test.head()


'''

'''check the corr

colormap = plt.cm.viridis
plt.figure(figsize=(12,12))
plt.title('Correlation between Features', y=1.05, size = 15)
sns.heatmap(X_train_analysis.corr(),
            linewidths=0.1, 
            vmax=1.0, 
            square=True, 
            cmap=colormap, 
            linecolor='white', 
            annot=True)
'''