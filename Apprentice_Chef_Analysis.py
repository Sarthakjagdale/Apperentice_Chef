
"""
Created on Thu Feb 05 15:14:36 2020
@author: Sarthak.Jagdale



Info : 
    ->  No bugs
    ->  This is the final model which has the best Score for statistical significance.
    ->  The model is Decision tree that helps in understanding the puchase behaviour.    
"""

################################################################################
# Import Packages
################################################################################

import random as rand # random number generation
import pandas as pd # data science essentials
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # enhanced data visualization
import statsmodels.formula.api as smf # regression modeling
from sklearn.model_selection import train_test_split # train/test split
from sklearn.linear_model import LinearRegression
import gender_guesser.detector as gender # using gender guesser
import sklearn.linear_model # linear models
from sklearn.model_selection import train_test_split # train-test split
from sklearn.linear_model import LogisticRegression  # logistic regression
from sklearn.metrics import confusion_matrix         # confusion matrix
from sklearn.metrics import roc_auc_score            # auc score
from sklearn.preprocessing import StandardScaler     # standard scaler
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier 
# setting random seed
rand.seed(a = 222)


# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


print('''
                                                                    
|  | _| _ _  _  _  |_ _    _  _ _ _|. _|_.   _   _  _  _ |  |_. _ _ 
|/\|(-|(_(_)|||(-  |_(_)  |_)| (-(_||(_|_|\/(-  (_|| )(_||\/|_|(__) 
                          |                               /         
''')


################################################################################
# Load Data
################################################################################

original_df = 'Apprentice_Chef_Dataset.xlsx'
#reading the file
chef = pd.read_excel(original_df)

#Get the top 10 rows for each column
chef.head(n=3)



# In[ ]:


################################################################################
# Feature Engineering, Variable Selection and (optional) Dataset Standardization
################################################################################

#########################
# text_split_feature
#########################
def text_split_feature(col, df, sep=' ', new_col_name='number_of_names'):
    """
Splits values in a string Series (as part of a DataFrame) and sums the number
of resulting items. Automatically appends summed column to original DataFrame.

PARAMETERS
----------
col          : column to split
df           : DataFrame where column is located
sep          : string sequence to split by, default ' '
new_col_name : name of new column after summing split, default
               'number_of_names'
"""
    
    df[new_col_name] = 0
    
    
    for index, val in df.iterrows():
        df.loc[index, new_col_name] = len(df.loc[index, col].split(sep = ' '))
        


logit_full = smf.logit(formula = """CROSS_SELL_SUCCESS ~ chef\
    ['FOLLOWED_RECOMMENDATIONS_PCT']""",
                        data = chef)

# telling Python to run the data through the blueprint
results = logit_full.fit()


# printing the results
print(results.summary())

##############################################################################

# instantiating a logistic regression model object
logistic_full = smf.logit(formula = """ chef['CROSS_SELL_SUCCESS'] ~ 
                                        chef['TOTAL_MEALS_ORDERED'] +
                                        chef['UNIQUE_MEALS_PURCH'] +
                                        chef['CONTACTS_W_CUSTOMER_SERVICE'] +
                                        chef['PRODUCT_CATEGORIES_VIEWED'] +
                                        chef['AVG_TIME_PER_SITE_VISIT'] +
                                        chef['MOBILE_NUMBER'] +
                                        chef['CANCELLATIONS_BEFORE_NOON'] +
                                        chef['CANCELLATIONS_AFTER_NOON'] +
                                        chef['TASTES_AND_PREFERENCES'] +
                                        chef['MOBILE_LOGINS'] +
                                        chef['PC_LOGINS'] +
                                        chef['WEEKLY_PLAN'] +
                                        chef['EARLY_DELIVERIES'] +
                                        chef['LATE_DELIVERIES'] +
                                        chef['PACKAGE_LOCKER'] +
                                        chef['REFRIGERATED_LOCKER'] +
                                        chef['FOLLOWED_RECOMMENDATIONS_PCT'] +
                                        chef['AVG_PREP_VID_TIME'] +
                                        chef['LARGEST_ORDER_SIZE'] +
                                        chef['MASTER_CLASSES_ATTENDED'] +
                                        chef['MEDIAN_MEAL_RATING'] +
                                        chef['AVG_CLICKS_PER_VISIT'] +
                                        chef['TOTAL_PHOTOS_VIEWED'] """,
                                                data = chef)


# fitting the model object
results_full = logistic_full.fit()


# checking the results SUMMARY
results_full.summary()
#############################################################################

chef['FAMILY_NAME'] = chef['FAMILY_NAME'].fillna('Unknown')
chef['FAMILY_NAME'].isnull().sum()

# saving results
chef.to_excel('chef_feature_rich.xlsx', index = False)

# loading saved file
chef = pd.read_excel('chef_feature_rich.xlsx')

#############################################################################

#Some Marketing Data analysis for better advertisements. 
# STEP 1: splitting All emails
# Customer List
customer_list = []

# looping over each email address
for index, col in chef.iterrows():
    
    # splitting email domain at '@'
    split_email = chef.loc[index, 'EMAIL'].split(sep = '@')
    
    # appending customer_list with the results
    customer_list.append(split_email)
    

# converting customer_list into a DataFrame 
email_df = pd.DataFrame(customer_list)

#chef data plotting
#chef.plot(kind='scatter', x='')

# displaying the results
email_df


# STEP 2: concatenating with original DataFrame

# multiple concatenations processing
chef = pd.read_excel('Apprentice_Chef_Dataset.xlsx')


# renaming column to concatenate
email_df.columns= ['name' , 'ALL_EMAIL_DOMAIN']


# concatenating ALL_EMAIL_DOMAIN with chef DataFrame
chef = pd.concat([chef, email_df.loc[:, 'ALL_EMAIL_DOMAIN']], axis = 1)


# printing value counts of ALL_EMAIL_DOMAIN
chef.loc[: ,'ALL_EMAIL_DOMAIN'].value_counts()

# email domain types
professional_email_domain = [ '@mmm.com','@amex.com','@apple.com','@boeing.com',
    '@caterpillar.com','@chevron.com','@cisco.com',
    '@cocacola.com','@disney.com','@dupont.com','@exxon.com',
    '@ge.org','@goldmansacs.com','@homedepot.com','@ibm.com',
    '@intel.com','@jnj.com','@jpmorgan.com','@mcdonalds.com',
    '@merck.com','@microsoft.com','@nike.com','@pfizer.com',
    '@pg.com','@travelers.com','@unitedtech.com','@unitedhealth.com',
    '@verizon.com','@visa.com','@walmart.com']
personal_email_domains = ['@gmail.com','@yahoo.com','@protonmail.com']
junk_email_domains  = ['@me.com','@aol.com','@hotmail.com','@live.com',
'@msn.com','@passport.com']


# placeholder list
customer_list = []


# looping to group observations by domain type
for domain in chef['ALL_EMAIL_DOMAIN']:
        if '@' + domain in professional_email_domain:
            customer_list.append('Proffessional Email')
            
        elif '@' + domain in personal_email_domains:
            customer_list.append('Personal Email')
            
        elif '@' + domain in junk_email_domains:
            customer_list.append('Junk Email')
            
        else:
            print('Unknown')


# concatenating with original DataFrame
chef['DOMAIN_GROUP'] = pd.Series(customer_list)
print('---------------------------------------------------------------------------')

# checking results
print(chef['DOMAIN_GROUP'].value_counts()) 

print('---------------------------------------------------------------------------')
############################################################################

# defining a function for categorical boxplots
def categorical_boxplots(response, cat_var, data):
    """
    This function can be used for categorical variables

    PARAMETERS
    ----------
    response : str, response variable
    cat_var  : str, categorical variable
    data     : DataFrame of the response and categorical variables
    """

    chef.boxplot(column          = response,
                    by           = cat_var,
                    vert         = False,
                    patch_artist = False,
                    meanline     = True,
                    showmeans    = True)
    
    plt.suptitle("")
    plt.show()


# calling the function for each categorical variable
categorical_boxplots(response = 'CROSS_SELL_SUCCESS',
                     cat_var  = 'DOMAIN_GROUP',
                     data     = chef)

# one hot encoding categorical variables
one_hot_Domain_group       = pd.get_dummies(chef['DOMAIN_GROUP'])
one_hot_All_Domain_group   = pd.get_dummies(chef['ALL_EMAIL_DOMAIN'])
one_hot_NAME = pd.get_dummies(chef['NAME'])
one_hot_FIRST_NAME  = pd.get_dummies(chef['FIRST_NAME'])
one_hot_FAMILY_NAME  = pd.get_dummies(chef['FAMILY_NAME'])
one_hot_EMAIL = pd.get_dummies(chef['EMAIL'])


# joining codings together
chef = chef.join([one_hot_Domain_group, one_hot_All_Domain_group])


#Save results
chef.to_excel('chef_feature_engineered.xlsx',
                 index = False)
chef = pd.read_excel('chef_feature_engineered.xlsx')


#############################################################################

print("""


Now the percentage of Revenue from each group of individual. 

""")



##counting total Revenue for individual Domain Groups
proffessional_revenue = chef.query("DOMAIN_GROUP == 'Proffessional Email'")['REVENUE'].sum()

personal_revenue = chef.query("DOMAIN_GROUP == 'Personal Email'")['REVENUE'].sum()

junk_revenue = chef.query("DOMAIN_GROUP == 'Junk Email'")['REVENUE'].sum()

Total = proffessional_revenue + personal_revenue + junk_revenue

print('---------------------------------------------------------------------------')

print(f' Total Revenue: {Total}')

print(f' Revenue from Personal Email: {round(personal_revenue/Total*100,2)}%')

print(f' Revenue from Professional Email: {round(proffessional_revenue/Total*100, 2)}%')

print(f' Revenue from Junk Email: {round(junk_revenue/Total*100,2)}%')

print('---------------------------------------------------------------------------')
################################################################################

########################
# DEFINING OUTLIER THRESHOLD
REVENUE_HI = 2000
REVENUE_LO = 500
TOTAL_MEALS_ORDERED_HI = 150
TOTAL_MEALS_ORDERED_LO = 20
PRODUCT_CATEGORIES_VIEWED_HI = 10
PRODUCT_CATEGORIES_VIEWED_LO = 2
AVG_PREP_VID_TIME_HI = 270
AVG_PREP_VID_TIME_LO = 80
LARGEST_ORDER_SIZE_HI = 10
LARGEST_ORDER_SIZE_LO = 2
UNIQUE_MEALS_PURCH_HI = 8
CANCELLATIONS_AFTER_NOON_HI = 2
AVG_TIME_PER_SITE_VISIT_HI = 180
CANCELLATIONS_BEFORE_NOON_HI = 4
MOBILE_LOGINS_HI = 6
PC_LOGINS_HI = 2
WEEKLY_PLAN_HI = 13
################################################################

#  calculating the outlier for Revenue
chef['REVENUE_OUT'] = 0
condition_hi = chef.loc[0:,'REVENUE_OUT'][chef['REVENUE'] > REVENUE_HI]
condition_hi = chef.loc[0:,'REVENUE_OUT'][chef['REVENUE'] < REVENUE_LO]

chef['REVENUE_OUT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

chef['TOTAL_MEALS_ORDERED_OUT'] = 0
condition_hi = chef.loc[0:,'TOTAL_MEALS_ORDERED_OUT'][chef['TOTAL_MEALS_ORDERED'] > TOTAL_MEALS_ORDERED_HI]

chef['TOTAL_MEALS_ORDERED_OUT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#  calculating the outlier for
chef['UNIQUE_MEALS_PURCH_OUT'] = 0
condition_hi = chef.loc[0:,'UNIQUE_MEALS_PURCH_OUT'][chef['UNIQUE_MEALS_PURCH']                                                      > UNIQUE_MEALS_PURCH_HI]

chef['UNIQUE_MEALS_PURCH_OUT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#  calculating the outlier for
chef['CANCELLATIONS_AFTER_NOON_OUT'] = 0
condition_hi = chef.loc[0:,'CANCELLATIONS_AFTER_NOON_OUT'][chef['CANCELLATIONS_AFTER_NOON']                                                            > CANCELLATIONS_AFTER_NOON_HI]

chef['CANCELLATIONS_AFTER_NOON_OUT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#  calculating the outlier for
chef['AVG_TIME_PER_SITE_VISIT_OUT'] = 0
condition_hi = chef.loc[0:,'AVG_TIME_PER_SITE_VISIT_OUT'][chef['AVG_TIME_PER_SITE_VISIT']                                                           > AVG_TIME_PER_SITE_VISIT_HI]

chef['AVG_TIME_PER_SITE_VISIT_OUT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#  calculating the outlier for
chef['CANCELLATIONS_BEFORE_NOON_OUT'] = 0
condition_hi = chef.loc[0:,'CANCELLATIONS_BEFORE_NOON_OUT'][chef['CANCELLATIONS_BEFORE_NOON']                                                             > CANCELLATIONS_BEFORE_NOON_HI]

chef['CANCELLATIONS_BEFORE_NOON_OUT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#  calculating the outlier for
chef['MOBILE_LOGINS_OUT'] = 0
condition_hi = chef.loc[0:,'MOBILE_LOGINS_OUT'][chef['MOBILE_LOGINS'] > MOBILE_LOGINS_HI]

chef['MOBILE_LOGINS_OUT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#  calculating the outlier for
chef['PC_LOGINS_OUT'] = 0
condition_hi = chef.loc[0:,'PC_LOGINS_OUT'][chef['PC_LOGINS'] > PC_LOGINS_HI]

chef['PC_LOGINS_OUT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)

#  calculating the outlier for
chef['WEEKLY_PLAN_OUT'] = 0
condition_hi = chef.loc[0:,'WEEKLY_PLAN_OUT'][chef['WEEKLY_PLAN'] > WEEKLY_PLAN_HI]

chef['WEEKLY_PLAN_OUT'].replace(to_replace = condition_hi,
                                value      = 1,
                                inplace    = True)



# calculating the outlier for
chef['PRODUCT_CATEGORIES_VIEWED_OUT'] = 0
condition_hi = chef.loc[0:,'PRODUCT_CATEGORIES_VIEWED_OUT'][chef['PRODUCT_CATEGORIES_VIEWED']                                                             > PRODUCT_CATEGORIES_VIEWED_HI]
condition_lo = chef.loc[0:,'PRODUCT_CATEGORIES_VIEWED_OUT'][chef['PRODUCT_CATEGORIES_VIEWED']                                                             < PRODUCT_CATEGORIES_VIEWED_LO]

chef['PRODUCT_CATEGORIES_VIEWED_OUT'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)


# calculating the outlier for
chef['AVG_PREP_VID_TIME_OUT'] = 0
condition_hi = chef.loc[0:,'AVG_PREP_VID_TIME_OUT'][chef['AVG_PREP_VID_TIME']                                                     > AVG_PREP_VID_TIME_HI]
condition_lo = chef.loc[0:,'AVG_PREP_VID_TIME_OUT'][chef['AVG_PREP_VID_TIME']                                                     < AVG_PREP_VID_TIME_LO]

chef['AVG_PREP_VID_TIME_OUT'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)

# calculating the outlier for
chef['LARGEST_ORDER_SIZE_OUT'] = 0
condition_hi = chef.loc[0:,'LARGEST_ORDER_SIZE_OUT'][chef['LARGEST_ORDER_SIZE']                                                      > PRODUCT_CATEGORIES_VIEWED_HI]
condition_lo = chef.loc[0:,'LARGEST_ORDER_SIZE_OUT'][chef['LARGEST_ORDER_SIZE']                                                      < PRODUCT_CATEGORIES_VIEWED_LO]

chef['LARGEST_ORDER_SIZE_OUT'].replace(to_replace = condition_hi,
                                    value      = 1,
                                    inplace    = True)


#################################################################################


# calling text_split_feature function
text_split_feature(col = 'NAME',
                   df  = chef)


# checking results
chef['number_of_names']


##############################################################################
chef.to_excel('chef_feature_engineered.xlsx',
                 index = False)
chef = pd.read_excel('chef_feature_engineered.xlsx')

###########################################################################



# Correlation betweeen vairables

df_corr = chef.corr().round(2)

df_corr['CROSS_SELL_SUCCESS'].sort_values(ascending = False)


# explanatory sets from last session

# creating a dictionary to store candidate models

candidate_dict = {

 # full model
 'logit_full'   : ['REVENUE', 'TOTAL_MEALS_ORDERED', 'UNIQUE_MEALS_PURCH', 
                   'CONTACTS_W_CUSTOMER_SERVICE', 'PRODUCT_CATEGORIES_VIEWED', 
                   'AVG_TIME_PER_SITE_VISIT', 'MOBILE_NUMBER', 
                   'CANCELLATIONS_BEFORE_NOON', 'CANCELLATIONS_AFTER_NOON',
                   'TASTES_AND_PREFERENCES', 'MOBILE_LOGINS', 'PC_LOGINS',
                   'WEEKLY_PLAN', 'EARLY_DELIVERIES', 'LATE_DELIVERIES',``
                   'PACKAGE_LOCKER', 'REFRIGERATED_LOCKER',
                   'FOLLOWED_RECOMMENDATIONS_PCT', 'AVG_PREP_VID_TIME',
                   'LARGEST_ORDER_SIZE', 'MASTER_CLASSES_ATTENDED', 
                   'MEDIAN_MEAL_RATING', 'AVG_CLICKS_PER_VISIT', 'TOTAL_PHOTOS_VIEWED',
                   'Junk Email', 'Personal Email', 'Proffessional Email'],
 
 # significant variables only
 'logit_sig'    : ['CANCELLATIONS_BEFORE_NOON', 'MOBILE_NUMBER', 'number_of_names',
                   'Proffessional Email', 'FOLLOWED_RECOMMENDATIONS_PCT']

}


################################################################################
# Train/Test Split
################################################################################
# train/test split with the logit_sig variables
chef_data   =  chef.loc[ : , candidate_dict['logit_sig']]
chef_target =  chef.loc[ : , 'CROSS_SELL_SUCCESS']


# train/test split
X_train, X_test, y_train, y_test = train_test_split(
            chef_data,
            chef_target,
            random_state = 802,
            test_size    = 0.25,
            stratify     = chef_target)

################################################################################
# Final Model (instantiate, fit, and predict)
################################################################################

# INSTANTIATING a classification tree object
full_tree = DecisionTreeClassifier(max_depth = 6)


# FITTING the training data
full_tree_fit = full_tree.fit(X_train, y_train)


# PREDICTING on new data
full_tree_pred = full_tree_fit.predict(X_test)

print('---------------------------------------------------------------------------')
print('---------------------------------------------------------------------------')
# SCORING the model
print(' Tree Training ACCURACY:', full_tree_fit.score(X_train, y_train).round(4))
print(' Tree Testing  ACCURACY:', full_tree_fit.score(X_test, y_test).round(4))
print(' Tree AUC Score        :', roc_auc_score(y_true  = y_test,
                                          y_score = full_tree_pred).round(4))
print('---------------------------------------------------------------------------')
print('---------------------------------------------------------------------------')

# train accuracy
full_tree_train_acc = full_tree_fit.score(X_train, y_train).round(4)


# test accuracy
full_tree_test_acc  = full_tree_fit.score(X_test, y_test).round(4)


# auc value
full_tree_auc       = roc_auc_score(y_true  = y_test,
                                    y_score = full_tree_pred).round(4)


################################################################################
# Final Model Score (score)
################################################################################

# use this space to score your final model on the testing set
# MAKE SURE TO SAVE YOUR TEST SCORE AS test_score
# Example: test_score = final_model.score(X_test, y_test)

test_score = full_tree_test_acc.round(4)

print("The Final Test Score:",test_score)


print("""


 ____                      _                              
 | |_  _  _ |      _      (_ _  _     _     _  |_. _  _|  
 | | )(_|| )|(  \/(_)|_|  | (_)|   \/(_)|_||   |_||||(-.  
                /                  /                      

""")


# In[ ]:




