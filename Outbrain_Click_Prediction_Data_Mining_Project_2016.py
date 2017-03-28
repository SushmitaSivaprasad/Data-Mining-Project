import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter
from sklearn.datasets import load_boston
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import csv
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation

clicks_train = pd.read_csv("/N/u/mmakam/Karst/Desktop/Data Mining OutBrain Project/clicks_train.csv")
events = pd.read_csv("/N/u/mmakam/Karst/Desktop/Data Mining OutBrain Project/events.csv")
promoted_content = pd.read_csv("/N/u/mmakam/Karst/Desktop/Data Mining OutBrain Project/promoted_content.csv")

documents_categories = pd.read_csv("/N/u/mmakam/Karst/Desktop/Data Mining OutBrain Project/documents_categories.csv")
documents_entities = pd.read_csv("/N/u/mmakam/Karst/Desktop/Data Mining OutBrain Project/documents_entities.csv")
#documents_meta = pd.read_csv("/N/u/mmakam/Karst/Desktop/Data Mining OutBrain Project/documents_meta.csv")
documents_topics = pd.read_csv("/N/u/mmakam/Karst/Desktop/Data Mining OutBrain Project/documents_topics.csv")
#page_views = pd.read_csv("/N/dc2/scratch/mmakam/page_views.csv")
#page_views_sample = pd.read_csv("/N/u/mmakam/Karst/Desktop/Data Mining OutBrain Project/page_views_sample.csv")

#clicks_test = pd.read_csv("/N/u/mmakam/Karst/Desktop/Data Mining OutBrain Project/clicks_test.csv")

#print "clicks_train", np.shape(clicks_train)
#print "documents_categories", np.shape(documents_categories)
#print "documents_entities", np.shape(documents_entities)
#print "documents_meta", np.shape(documents_meta)
#print "documents_topics", np.shape(documents_topics)
#print "events", np.shape(events)
#print "page_views", np.shape(page_views)
#print "page_views_sample", np.shape(page_views_sample)
#print "clicks_test", np.shape(clicks_test)

#print clicks_train
#print clicks_test
#page_views_counts = page_views_sample.groupby(['document_id'], sort = True)['uuid'].count()

#print page_views_counts.sort(inplace = 'True')

print len(clicks_train['ad_id'].unique())
print len(clicks_train['ad_id'])
duplicate_bool = page_views_sample.duplicated()
type(duplicate_bool)
If (duplicate_bool == True).any() == False:
  print "No duplicates in data"

duplicate_bool = len(page_views_sample.duplicated()== 'True')

    

#clicks_train_counts['rank'] = clicks_train_counts.groupby('display_id','ad_id')['clicked'].rank('dense', ascending=True)

#print clicks_train_rank


print page_views_sample.columns
print documents_entities.columns

print len(promoted_content)
print len(promoted_content['ad_id'].unique())


print len(clicks_train)
print len(clicks_train['ad_id'].unique())
print len(clicks_train['display_id'].unique())

#joining clicks_train and events table

#clicks_events = clicks_train.join(events, on=['display_id'], how = 'left', lsuffix='_left', rsuffix='_right')
print clicks_events.columns

print clicks_train.shape
print events.shape
print clicks_events.shape

clicks_events_inner = clicks_train.join(events, on=['display_id'], how = 'inner', lsuffix='_CT', rsuffix='_EV')
print clicks_events_inner.shape

#pd.merge(left, right, how='inner', on=None, left_on=None, right_on=None,
#         left_index=False, right_index=False, sort=True,
#         suffixes=('_x', '_y'), copy=True, indicator=False)
clicks_events_merge = pd.merge(clicks_train,events, how = 'inner', on = 'display_id',left_index=False, right_index=False, sort=True )

clicks_events_promoted = pd.merge(clicks_events_merge,promoted_content, how = 'inner', on = ['ad_id'],left_index=False, right_index=False, sort=True )
#clicks_events_inner.join(promoted_content, on=['ad_id'], how = 'inner', lsuffix='_CE', rsuffix='_PC')
print clicks_events_promoted.shape

promoted_content_ad_id = promoted_content['ad_id'].unique()
clicks_train_ad_id = clicks_train['ad_id'].unique()
type(clicks_train_ad_id)
k = 0
for i in clicks_train_ad_id:
    print k
    k = k + 1
    if i not in  promoted_content_ad_id:
        print i        
        break;


if clicks_train_ad_id not in promoted_content_ad_id:
	print "Not in"


if promoted_content_ad_id not in clicks_train_ad_id:
	print "Not in"


print clicks_events_promoted.columns

print clicks_events_inner.columns

events.groupby(['display_id']).agg(['dis'])

event_counts = events.groupby(['display_id'], sort = True)['uuid'].count()

for i in event_counts:
    if i>1:
        print i



clicks_train_count = clicks_train.groupby(['display_id'], sort = True)['clicked'].agg('sum')

for i in clicks_train_count:
    if i>1:
        print i


event_counts1 = events.groupby(['uuid'], sort = True)['display_id'].count()

for i in event_counts1:
    if i>1:
        print i

event_counts1 = events.groupby(['uuid'], sort = True)['display_id'].count()


len(promoted_content['ad_id'].unique())
promoted_content.shape


clicks_add_cnt = clicks_train.groupby(['ad_id'], sort = True)['display_id'].nunique().reset_index()
clicks_add_cnt1 = clicks_train.groupby(['ad_id'], sort = True)['ad_id'].count()

#creating host and promoted copies of categories
documents_categories_host = documents_categories.rename(columns = {'document_id':'host_document_id', 'category_id': 'host_category_id', 'confidence_level': 'host_category_cl'})
documents_categories_promoted = documents_categories.rename(columns = {'document_id':'promoted_document_id', 'category_id': 'promoted_category_id', 'confidence_level': 'promoted_category_cl'})

pd.documents_categories.drop()

clicks_events_promoted_r = clicks_events_promoted.loc[:,['display_id','ad_id','host_document_id','promoted_document_id']].copy()
clicks_events_promoted.drop()

host_and_promoted = pd.merge(clicks_events_promoted_r, documents_categories_host, how = 'inner', on = 'host_document_id' )

pd.documents_categories_host.drop()

host_and_promoted1 = pd.merge(host_and_promoted, documents_categories_promoted, how = 'inner', on = 'promoted_document_id' )

#creating host and promoted copies of entities
documents_entities_host = documents_entities.rename(columns = {'document_id':'host_document_id', 'entity_id': 'host_entitity_id', 'confidence_level': 'host_entity_cl'})
documents_entities_promoted = documents_entities.rename(columns = {'document_id':'promoted_document_id', 'entity_id': 'promoted_entity_id', 'confidence_level': 'promoted_entity_cl'})

pd.documents_entities.drop()
