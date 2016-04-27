
import IPython.core.display as di
# This line will hide code by default when the notebook is exported as HTML
di.display_html('<script>jQuery(function() {if (jQuery("body.notebook_app").length == 0) { jQuery(".input_area").toggle(); jQuery(".prompt").toggle();}});</script>', raw=True)
from IPython.core.display import HTML
from __future__ import division

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
from utilplot import barplot, pointplot

import statsmodels.api as sm
from sklearn.cross_validation import KFold

import warnings
warnings.filterwarnings("ignore")




#read in table 
data = pd.read_csv('DS Product Take Home Data.csv', sep = ',')

#subset data to extract patients information
patients = data[['member_age','member_sex','health_risk_assesment','event_id','outcome']]
groupby_col = ['member_age','health_risk_assesment']
patients_grouped = patients.groupby(groupby_col).count().reset_index()


#stacked bar plot
patients_grouped.pivot('health_risk_assesment', 'member_age')['event_id'].plot(kind='bar', 
                                                                stacked=True, 
                                                                alpha=0.7, 
                                                                figsize = (35,10),
                                                                fontsize = 40,
                                                                colormap = 'RdYlBu',
                                                                rot = 0
                                                                )
plt.xlabel('Health Risk Assessment', size=40)
plt.ylabel('# of patients', size = 40)
plt.title('Stacked Bar Plot of Patients with Different Age Group within Each Health Risk', size = 40)
plt.legend(bbox_to_anchor=(1, 1), loc=2, prop={'size':30})
ax = plt.gca()
ax.set_axis_bgcolor("white")

#calculate failure rate
def failure_rate(table, groupby_col, failure, total):
    """ Aggregate data and calculate the failures % withthin the aggregated group 
        Args: table
              groupby_col: list, column names 
              failure: number of failured patients in that group
              total: total number of patients in that group 
        Returns: an aggregated table with failure %
    """
    table_new = table.groupby(groupby_col).count().reset_index()
    table_new['failure_percentage'] = table_new[failure]/ table_new[total] * 100
    
    return table_new



#aggregate failure % within each age group 
color = sns.light_palette("seagreen")[-2]
patients_age = failure_rate(patients, 'member_age', 'outcome', 'event_id')
#plot
barplot(patients_age, 'member_age', 'failure_percentage',None,  
        'Failure % for Each Age Group', 'Member Age', 'Failure% = failures/patients', 
        (35, 10), color)


#data process: group by sex
patients.loc[:,'member_sex'] = patients.loc[:,'member_sex'].map({0: 'female', 1: 'male'})
patients.loc[:,'outcome'] = patients.loc[:,'outcome'].fillna('non-failure')#count
# of failures for each gender
patients_sex = patients.groupby(['member_sex','outcome']).count().reset_index()
#plot
barplot(patients_sex, 'member_sex', 'event_id', 'outcome', 
        '# of Failure vs Non-failuare Patients for Each Gender', 'Member Gender', 'Failure% = failures/patients', 
        (35, 10), color)



#plot
patients_risk = failure_rate(data, 'health_risk_assesment', 'outcome', 'event_id')
barplot(patients_risk, 'health_risk_assesment', 'failture_percentage', None,                                #columns
        'Failure % for Each Health Risk Level',  'Health Risk Assessment', 'Failure% = failures/patients',   #title label
        (35, 10), color)                                                                                    #figsize, color


#calculate total failure % for each doctor
data_doctors = data.groupby(['servicing_provider_name']).count().reset_index()
data_doctors['failure_percentage'] = data_doctors['outcome']/data_doctors['event_id']
col_keep = ['servicing_provider_name', 'failure_percentage', 'event_id']
data_doctors[col_keep].sort_values('failure_percentage')[:20]


#number of patients in each health risk group 
col_keep1 = ['servicing_provider_name','health_risk_assesment','event_id']
doctor_risk_group = data[col_keep1].groupby(['servicing_provider_name','health_risk_assesment']).count().reset_index()
#total number of 
col_keep2 = ['servicing_provider_name','event_id']
doctor_total_patients = data[col_keep2].groupby(['servicing_provider_name']).count().reset_index()
#calculate patients in each group %
doctor_join = doctor_risk_group.merge(doctor_total_patients, on = 'servicing_provider_name', how = 'left')
doctor_join['Patient_Percentage'] = doctor_join['event_id_x'] / doctor_join['event_id_y']
#final table output 
col_keep3 = ['servicing_provider_name','health_risk_assesment','Patient_Percentage']
doctor_join[col_keep3][doctor_join['health_risk_assesment'].isin([8,9,10])].sort_values('Patient_Percentage', ascending = False)[:20]


#########modeling#######################

#calculate correlations
corr = data[['member_age','member_sex','health_risk_assesment']].corr()
corr


#One-Hot-Coding: servicing_provider_name is categorical variable.

def replace(x):
    if x == 'failure': 
        return 1
    else: 
        return 0

#data process 
data_for_model = data.copy(deep = True)
data_for_model['outcome'] = data_for_model['outcome'].map(lambda x: replace(x))  #0 and 1
data_for_model['Intercept'] = 1
#one-hot-coding: create dummy variable for doctors 
dummy_ranks = pd.get_dummies(data_for_model['servicing_provider_name'])#doctor's name to dummy variable 

#data set
cols_to_keep = ['outcome', 'Intercept', 'health_risk_assesment']
data_vectorized = data_for_model[cols_to_keep].join(dummy_ranks.ix[:, 1:])
train_cols = data_vectorized.columns[1:]
#seperate feature and target 
X_data = data_vectorized[train_cols]
y_target = data_vectorized['outcome']


def zero_one(x):
    if x < 0.5: 
        return 0
    else: 
        return 1

#cross-validation
def cross_validation(X_data, target, iteration):
    """Split data into train and test sets. 
       Fit a logistic regression and predict y for test set
       Calculate prediction accuracies 
       Args: features: table with all x variables 
             target: corresponding y (0,1)
             proportion: float, [0,1] train set : test set size ratio
             iteration: int. numbers of cross validations 
      Returns: a list of prediction accuracy ratios 
    """
    ratio = []
    kf = KFold(n=len(X_data), n_folds= iteration, shuffle=False, random_state=None)
    
    for train_index, test_index in kf:
        #split data
        X_train, X_test = X_data.ix[train_index, :], X_data.ix[test_index, :]
        y_train, y_test = target[train_index], target[test_index]
        #fit model 
        logit = sm.Logit(y_train, X_train)
        result = logit.fit()
        #prediction 
        y_pred = result.predict(X_test)
        y_pred = [zero_one(y) for y in y_pred]
        diff = y_test - y_pred
        ratio.append([1 -np.count_nonzero(diff)/np.float(len(diff))])
        
    return ratio



ratio = cross_validation(X_data, y_target, 10)


# ###68% accuracy on average
np.mean(ratio)


#fit model using all data 
logit = sm.Logit(y_target, X_data)
result = logit.fit()
result.summary()

#calculate exp(Coef)
results = pd.DataFrame(np.exp(result.params), columns = ['Exp(Coef)'])
results.sort_values('Exp(Coef)', ascending = False)[: -1]



#intitalize empty data 
zero_data = np.zeros(shape=(10,len(X_data.columns)))
data_for_pred = pd.DataFrame(zero_data, columns=X_data.columns)
data_for_pred['health_risk_assesment'] = np.arange(1,len(data['health_risk_assesment'].unique())+1,1)
#select 3 effective and 3 ineffective doctors to see the probability of a patient at risk X revisit if a patient is treated by doctor X
doctor_list = ['Spaceman', 'Cox', 'Boom', 'Octopus','Dre', 'Lower']


doctor_pred = pd.DataFrame()
#assign value 1 to each doctor
col_keep = ['health_risk_assesment', 'servicing_provider_name', 'y_pred']
for doctor in doctor_list: 
    data_new = data_for_pred.copy(deep = True)
    data_new[doctor] = 1
    data_new['y_pred'] = result.predict(data_new)
    data_new['servicing_provider_name'] = doctor 
    doctor_pred = doctor_pred.append(data_new[col_keep])


#plot the probability
color = sns.diverging_palette(240, 10, n=6)
pointplot(doctor_pred, 'health_risk_assesment', 'y_pred', 'servicing_provider_name', 
          'Predicted Probability of Patients Re-visit for Different Doctors', 'Health Risk Assessment', 'Probability', 
           color, (35, 10))

