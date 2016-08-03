# -*- coding: utf-8 -*-
"""
Created on Tue May 03 12:14:07 2016
"""

import pandas as pd
import sklearn 
import re
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
import datetime

in_file_train = "/desktop/DataSets/RescueAnimal/train.csv"
in_file_test = "/desktop/DataSets/RescueAnimal/test.csv"

prediction = '/desktop/DataSets/RescueAnimal/prediction.csv'

data_train = pd.read_csv(in_file_train, header=0, sep=',')
data_test = pd.read_csv(in_file_test)

def convert_age(string):

    time_dict = {'year':365, 'day':1, 'month':30, 'week':7}
    
    if string == 'NA':
        return 0
    num,time = string.split()
    num = int(num)
    time = time.replace('s','')
    
    return num*time_dict[time]

def lifestage(num):
    if num == '0':
        return 'Unknown'

    yrs = num/365
    
    if yrs < 1:
        return 'baby'
    elif 1<= yrs < 3:
        return 'adult'
    elif 3 <= yrs < 8:
        return 'adult'
    else:
        return 'old'

def gender(string, i):
    gend_info = string.split()
    if len(gend_info) == 1:
        return 'Unknown'
    else:
        return gend_info[i]
        
def hairlength(string):
    if re.search('Longhair', string):
        return "Long"
    elif re.search('Shorthair', string):
        return "Short"
    else:
        return "Unknown"

def is_mix(string):
    if re.search('Longhair|Shorthair|Mix|/', string):
        return 1
    return 0

def dummy_data(DF, field):
    dummies = pd.get_dummies(DF[field], prefix=field)
    DF = DF.join(dummies)
    DF.drop(field, inplace=True, axis = 1)
    return DF

def simplify(string):
    if string == 'NA':
        return 'Unknown'
    return string.split('/')[0].replace(' Mix','').lower()
    
def eight_to_six(time):
    if time == 'NA':
        return -1
    date = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    if 8 <= date.hour <= 18:
        return 1
    return 0

def weekday(time):
    if time == 'NA':
        return -1
    date = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    if date.weekday() > 4:
        return 0
    return 1

def season(time):
    if time == 'NA':
        return 'Unknown'
    
    date= datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
    if 1<= date.month < 4:
        return 'Winter'
    elif 4 <= date.month < 7:
        return 'Spring'
    elif 7<= date.month <10:
        return 'Summer'
    return 'Fall'

def cute_name(name):
    if len(name)<2:
        return 0
    if name[-1] == 'y' or name[-2:] == 'ie':
        return 1
    return 0

def make_colors_breeds(lst, df, simple_field):
    for thing in lst:
        df[thing] = df.apply(lambda x: 1 if re.search(thing, x[simple_field]) else 0, axis = 1)

def make_counts_col(fld, df):
    fld_ct_lbl = fld+'_count'
    fld_counts = pd.DataFrame(df[fld].value_counts()).reset_index()
    fld_counts.columns = [fld, fld_ct_lbl]
    return df.merge(fld_counts)

color_list = ['black','white','tabby','brown', 'orange', 'blue',
              'tan', 'red','tortie', 'calico', 'chocolate',
              'torbie', 'sable', 'yellow', 'gray', 'buff',
              'cream', 'fawn']   
breed_list = ['german shepherd','labrador retriever',
              'boxer','pit bull', 'cattle dog', 'dachshund',
              'chihuahua','poodle','terrier', 'siamese',
              'collie', 'beagle', 'schnauzer', 'husky',
              'rottweiler', 'shepherd', 'shu tzu', 'bulldog']

breed_list2 = ['german shepherd','labrador retriever',
              'pit bull', 'chihuahua']

def create_fields(dataset):
    
    dataset.fillna('NA', inplace=True)
    dataset['age'] = dataset.apply(lambda x: convert_age(x['AgeuponOutcome']), axis = 1)
    dataset['lifestage'] = dataset.apply(lambda x: lifestage(x['age']), axis = 1)    
    dataset['gender'] = dataset.apply(lambda x: gender(x['SexuponOutcome'], 1), axis =1)
    dataset['fixed'] = dataset.apply(lambda x: gender(x['SexuponOutcome'], 0), axis =1)
    dataset['hairlength'] = dataset.apply(lambda x: hairlength(x['Breed']), axis =1)
    dataset['mix'] = dataset.apply(lambda x: is_mix(x['Breed']), axis = 1)
    
    dataset['black'] = dataset.apply(lambda x: int(x['Color'] == 'Black'), axis =1)
    dataset['white'] = dataset.apply(lambda x: int(x['Color'] == 'White'), axis =1)
    
    dataset['simple_breed'] = dataset.apply(lambda x: simplify(x['Breed']), axis = 1)    
    dataset['simple_color'] = dataset.apply(lambda x: simplify(x['Color']), axis = 1)
    
    #make_colors_breeds(color_list, dataset, 'simple_color')
    make_colors_breeds(breed_list2, dataset, 'simple_breed')    
    
    dataset['daytime'] = dataset.apply(lambda x: eight_to_six(x['DateTime']), axis = 1)
    dataset['season'] = dataset.apply(lambda x: season(x['DateTime']), axis = 1)   
    dataset['weekday'] = dataset.apply(lambda x: weekday(x['DateTime']), axis = 1)    
    
    dataset['has_name'] = dataset.apply(lambda x: int(x['Name'] != 'NA'), axis = 1)
    dataset['two_word_name'] = dataset.apply(lambda x: int(len(x['Name'].split(' '))>1), axis =1)
    dataset['cute_name'] = dataset.apply(lambda x: cute_name(x['Name']), axis =1 ) 

    dataset = make_counts_col('Name', dataset)
    dataset = make_counts_col('simple_breed', dataset)
    dataset = make_counts_col('simple_color', dataset)
    
    print dataset.info()    
    dataset['unique_name'] = dataset.apply(lambda x: int(x['Name_count'] < 2), axis = 1)

    dataset['class_breed'] = dataset.apply(lambda x: x['simple_breed'] if x['simple_breed_count'] >=200 else 'Unknown', axis = 1)
    dataset['class_color'] = dataset.apply(lambda x: x['simple_color'] if x['simple_color_count'] >=300 else 'Unknown', axis = 1)
    
    return dataset

def fix_predict(string):
    """
    Adoption,Died,Euthanasia,Return_to_owner,Transfer
    """
    if string == 'Adoption':
        return '1,0,0,0,0'
    elif string == 'Died':
        return '0,1,0,0,0'
    elif string == 'Euthanasia':
        return '0,0,1,0,0'
    elif string == 'Return_to_owner':
        return '0,0,0,1,0'
    elif string == 'Transfer':
        return '0,0,0,0,1'

def results_file(classifier):
    b = classifier.predict_proba(test_mat)
    
    results = pd.read_csv("/desktop/DataSets/RescueAnimal/sample_submission.csv")
    
    results['ID'] = test_labs
    results['Adoption'] = b[:,0]
    results['Died'] = b[:,1]
    results['Euthanasia'] = b[:,2] 
    results['Return_to_owner'] = b[:,3] 
    results['Transfer'] = b[:,4]
    
    output_loc = "/desktop/DataSets/RescueAnimal/prediction.csv"    
    
    results.to_csv(output_loc, index=False)
    
    print "Results exported to: %s" %output_loc
    

def create_stacking_df(lst, test=False):
    if test:
        mat = test_mat
        lab = test_labs
    else:
        mat = train_mat
        lab = train_labs
    cf = lst[0]
    cf_init = cf.predict_proba(mat)

    results = pd.DataFrame()

    results['ID'] = lab
    results['Adoption'] = cf_init[:,0]
    results['Died'] = cf_init[:,1]
    results['Euthanasia'] = cf_init[:,2] 
    results['Return_to_owner'] = cf_init[:,3] 
    results['Transfer'] = cf_init[:,4]
    
    i = 1
    for cfl in lst[1:]:
        cf_init = cfl.predict_proba(mat)    
        
        #results2 = pd.read_csv("/desktop/DataSets/RescueAnimal/sample_submission.csv")
        results2 = pd.DataFrame()
        
        results2['ID'] = lab
        results2['Adoption'] = cf_init[:,0]
        results2['Died'] = cf_init[:,1]
        results2['Euthanasia'] = cf_init[:,2] 
        results2['Return_to_owner'] = cf_init[:,3] 
        results2['Transfer'] = cf_init[:,4]
        
        results = results.merge(results2, left_on='ID', right_on='ID', suffixes=['',i])   
        i +=1
        
    return results
    
    

dataset = create_fields(data_train)
dataset_test = create_fields(data_test)

train_cols = ['AnimalType', 'lifestage', 'age', 'gender', 'fixed', 
          'hairlength', 'mix', 'has_name', 'season', 
          'daytime', 'two_word_name', 'cute_name',
          'class_breed', 'class_color']

#train_cols += color_list
#train_cols += breed_list

to_dummy = ['AnimalType', 'gender', 'fixed', 'hairlength', 
         'season','lifestage', 'class_breed', 'class_color']

limited_cols = ['AnimalType', 'age', 'fixed', 'has_name',
                'gender', 'weekday', 'daytime', 'hairlength','class_breed', 'class_color']

#limited_cols += color_list
#limited_cols += breed_list2

to_dummy_limit = ['AnimalType', 'fixed', 'gender', 'hairlength', 'class_breed', 'class_color']

#train = dataset[train_cols]
train = dataset[limited_cols]
train['train'] = 0
train_labs = dataset['AnimalID'].tolist()
#test = dataset_test[train_cols]
test = dataset_test[limited_cols]
test['train'] = 1
test_labs = dataset_test['ID'].tolist()
target = dataset['OutcomeType']

both = train.append(test, ignore_index=True)

for var in to_dummy_limit:
    both = dummy_data(both, var)

train = both[both['train'] == 0]
test = both[both['train'] == 1]

train.drop('train', inplace=True, axis =1)
test.drop('train', inplace=True, axis = 1)

train_mat = train.as_matrix()
test_mat = test.as_matrix()
target_mat = target.as_matrix()

##classify data

print "\n\n*********** Beginning Classification *************\n\n"


param_grid = {'n_estimators': [50, 100, 200],
              'max_depth': [2,4,6],
              'min_samples_split': [1,2, 10],
              'max_features': ['auto', None, .5]}
print "\n Random Forest \n"
rf_pre = RandomForestClassifier(n_jobs = -1, random_state=12)
clf = GridSearchCV(rf_pre, param_grid)
clf.fit(train_mat, target_mat)

print clf.best_params_
rf = clf.best_estimator_

#rf = RandomForestClassifier(n_jobs = 2, n_estimators = 500)

scores = cross_val_score(rf, train_mat, target_mat, cv=5)
print scores

rf.fit(train_mat, target_mat)
a = zip(train.columns.values,rf.feature_importances_)
a.sort(key=lambda x: -x[1])

print "top 25 features: \n"
for i in a[:30]:
    print i

"""
rf_pred = rf.predict_proba(test_mat)


print "\n Ada Boost \n"

param_grid = {'n_estimators': [10,50,100],
              'learning_rate': [.05,.1,.25,.5]}

AB_pre = AdaBoostClassifier()
clf = GridSearchCV(AB_pre, param_grid)
clf.fit(train_mat, target_mat)

print clf.best_params_
AB = clf.best_estimator_

scores = cross_val_score(AB, train_mat, target_mat, cv=5)
print scores

AB_pred = AB.predict_proba(test_mat)


param_grid = {'n_estimators': [50],
              'learning_rate': [.05],
               'max_features': [None, 2, 4],
               'max_depth': [2,6],
               'min_samples_split': [1,2] }

GBC_pre = GradientBoostingClassifier(random_state = 12)
clf = GridSearchCV(GBC_pre, param_grid)
clf.fit(train_mat, target_mat)

print clf.best_params_
GBC = clf.best_estimator_

#GBC = GradientBoostingClassifier(random_state = 12).fit(train_mat, target_mat)
scores = cross_val_score(GBC, train_mat, target_mat, cv=5)
print scores

a = zip(train.columns.values,GBC.feature_importances_)
a.sort(key=lambda x: -x[1])

print "top 25 features: \n"
for i in a[:25]:
    print i

GB_pred = AB.predict_proba(test_mat)

print "\n Logisitic Regression \n"

param_grid = {'penalty':['l1','l2']}

log_rg_pre = LogisticRegression(random_state = 12)
clf = GridSearchCV(log_rg_pre, param_grid)
clf.fit(train_mat, target_mat)

print clf.best_params_
log_rg = clf.best_estimator_

scores = cross_val_score(log_rg, train_mat, target_mat, cv = 5)
print scores

print "\n Stacking \n"

stacked_train = create_stacking_df([rf, AB, GBC, log_rg])
stacked_train.drop('ID', inplace=True, axis=1)

lr = LogisticRegression(random_state=12)

scores = cross_val_score(lr, stacked_train, target_mat, cv=5)
print scores

lr.fit(stacked_train, target_mat)

stacked_test = create_stacking_df([rf, AB, GBC, log_rg], test=True)
stacked_test.drop('ID', inplace=True, axis=1)

b = lr.predict_proba(stacked_test)

results = pd.read_csv("/desktop/DataSets/RescueAnimal/sample_submission.csv")

results['ID'] = test_labs
results['Adoption'] = b[:,0]
results['Died'] = b[:,1]
results['Euthanasia'] = b[:,2] 
results['Return_to_owner'] = b[:,3] 
results['Transfer'] = b[:,4]

output_loc = "/desktop/DataSets/RescueAnimal/prediction.csv"    

results.to_csv(output_loc, index=False)

## generates output file
#
#c = []
#for prediction in b:
#    c.append(fix_predict(prediction))
#
#out_file = zip(test_labs, c)
#
#o = open('C:/users/brjohn/desktop/datasets/rescueanimal/prediction.csv', 'w')
#o.write('ID,Adoption,Died,Euthanasia,Return_to_owner,Transfer\n')
#for line in out_file:
#    o.write('%s,%s\n' %line)
#o.close()

"""
