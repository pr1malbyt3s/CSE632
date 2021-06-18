# Import necessary libraries:
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder
from kmodes.kprototypes import KPrototypes
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Kaggle specific imports:
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Import the raw uncleaned data set:
nc_data_raw = nc_data_raw = pd.read_csv('/kaggle/input/stanford-open-policing-project-north-carolina/NC.csv', index_col='id')

# Drop state, county_fips, stop_time, county_name, location_raw, fine_grained_location, and officer_id columns:
nc_data_drop = nc_data_raw.drop(['state', 'county_fips', 'stop_time', 'county_name', 'driver_age_raw', 'location_raw', 'fine_grained_location', 'search_type_raw', 'driver_race', 'officer_id'], axis=1)

# Remove the rows with no district, a total of 336993:
nc_data_d = nc_data_drop.loc[nc_data_drop['district'].notnull()]

# Remove the 7 rows with missing driver data:
nc_data_t = nc_data_d.loc[nc_data_d['driver_gender'].notnull()]

# Convert stop_date to date format and create new features:
nc_data_t['stop_date'] = pd.to_datetime(nc_data_t['stop_date'])
nc_data_t['stop_year'] = nc_data_t['stop_date'].dt.year
nc_data_t['stop_month'] = nc_data_t['stop_date'].dt.month
nc_data_t['stop_day'] = nc_data_t['stop_date'].dt.day
nc_data_t['stop_week'] = nc_data_t['stop_date'].dt.isocalendar()['week']
nc_data_t['stop_weekday'] = nc_data_t['stop_date'].dt.isocalendar()['day']
nc_data_t.drop('stop_date', axis=1, inplace=True)

# Replace null search_type values with 'None'. All null search_type values have False search_conducted values:
nc_data_t['search_type'].fillna('None', inplace=True)

# Replace null search_basis values with 'None'. All null search_basis values have False search_conducted values:
nc_data_t['search_basis'].fillna('None', inplace=True)

# Replace unknown drug_related_stop values with False (all others are true, so we assume uknowns are false):
nc_data_t['drugs_related_stop'].fillna(False, inplace=True)

# Only keep samples with known driver ages:
nc_data = nc_data_t.loc[nc_data_t['driver_age'].notnull()]

# Only retain 2015 samples:
nc_data_2015 = nc_data.loc[nc_data['stop_year'] == 2015]
nc_data_2015.drop('stop_year', axis=1, inplace=True)

# Import the population data set
pop_data = pd.read_csv('../input/nc-county-populations-2015/county-population-estimates-standard-revised.csv', sep=';')

# Drop the Year, geom, geo_point_2d, and ctyfips features:
pop_data.drop(['Year', 'geom', 'geo_point_2d', 'ctyfips'], axis=1, inplace=True)

# Remove the 'County' from each county attribute to match future mapping for districts:
pop_data['County'] = pop_data['County'].str.replace(' County', '')
# Change the index to county names:
pop_data.set_index('County', inplace=True)

# Create a data frame of each stop count from each district:
stop_count = (nc_data_2015['district'].value_counts()).to_frame()
# Rename the column to Stop_Count:
stop_count.rename(columns={'district':'Stop_Count'}, inplace=True)

# Create a mapping of districts to counties which will be used to aggregate populations:
dc_map = {'A1':['Currituck'],
          'A2':['Hertford', 'Gates', 'Bertie'],
          'A3':['Pasquotank', 'Chowan', 'Perquimans', 'Camden'],
          'A4':['Beaufort', 'Washington', 'Tyrrell', 'Hyde'],
          'A5':['Pitt', 'Martin'],
          'A6':['Craven', 'Pamlico'],
          'A7':['Lenoir', 'Jones'],
          'A8':['Carteret'],
          'A9':['Pitt'],
          'B1':['Cumberland'],
          'B2':['Sampson'],
          'B3':['Onslow'],
          'B4':['Duplin', 'Pender'],
          'B5':['Columbus', 'Bladen'],
          'B6':['New Hanover', 'Brunswick'],
          'B7':['Robeson'],
          'B8':['Harnett'],
          'C1':['Nash', 'Edgecombe'],
          'C2':['Wayne'],
          'C3':['Wake'],
          'C4':['Vance', 'Warren', 'Franklin'],
          'C5':['Wilson', 'Greene'],
          'C6':['Johnston'],
          'C7':['Durham', 'Granville'],
          'C8':['Halifax', 'Northampton'],
          'D1':['Chatham', 'Lee'],
          'D2':['Guilford'],
          'D3':['Rockingham'],
          'D4':['Person', 'Caswell'],
          'D5':['Alamance'],
          'D6':['Randolph'],
          'D7':['Orange'],
          'E1':['Davidson'],
          'E2':['Stanly', 'Montgomery'],
          'E3':['Rowan'],
          'E4':['Forsyth'],
          'E5':['Stokes', 'Surry'],
          'E6':['Cabarrus'],
          'E7':['Davie', 'Yadkin'],
          'F1':['Burke'],
          'F2':['Wilkes', 'Ashe', 'Alleghany'],
          'F3':['Caldwell', 'Watauga'],
          'F4':['Iredell', 'Alexander'],
          'F5':['Catawba', 'Lincoln'],
          'G1':['Yancey', 'Mitchell', 'Avery', 'Madison'],
          'G2':['McDowell', 'Rutherford'],
          'G3':['Henderson', 'Transylvania', 'Polk'],
          'G4':['Buncombe'],
          'G5':['Haywood', 'Jackson'],
          'G6':['Swain', 'Macon', 'Clay', 'Cherokee', 'Graham'],
          'H1':['Gaston'],
          'H2':['Richmond', 'Scotland'],
          'H3':['Union', 'Anson'],
          'H4':['Cleveland'],
          'H5':['Mecklenburg'],
          'H6':['Moore', 'Hoke']
         }
         
# Iterate through the district mapping:
for key in dc_map:
    pop_val = 0
    # Iterate through each county in the district:
    for county in dc_map[key]:
        # Add the population to the total:
        pop_val += pop_data.at[county, 'Population']
    # Assign the total population for each district to the data frame:
    stop_count.at[key, 'Population'] = pop_val
    
# Drop the A9 district:
stop_count.drop('A9', inplace=True)

## Clustering Analysis
# Get all the categorical features:
s = (nc_data_2015.dtypes == 'object')
object_cols = list(s[s].index)
# Append the boolean features:
object_cols.extend(['search_conducted', 'contraband_found', 'is_arrested', 'drugs_related_stop'])
print(object_cols)

# Create a data frame copy to label encode all categorical features:
nc_encoded = nc_data_2015.copy()
label_encoder = LabelEncoder()
for col in object_cols:
    nc_encoded[col] = label_encoder.fit_transform(nc_encoded[col])
    
# Create a subset using only age, gender, and race and use it to perform KPrototypes clustering:
X = nc_encoded.iloc[:, [1,2,3]]
kp = KPrototypes(n_clusters=3, init="Cao", n_init=1, verbose=1)
cluster_labels = kp.fit_predict(X, categorical=[0, 2])
X['Cluster'] = cluster_labels

# Create a count plot showing clusters related to feature values:
plt.figure(figsize=(16,10))
sns.countplot(x='Cluster', hue='driver_race_raw', data=X)
plt.legend(title='Driver Race', labels=['Asian', 'Black Hispanic', 'Black', 'Other', 'Unknown Hispanic', 'Unknown Non-Hispanic', 'White Hispanic', 'White'])
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.title("Driver Race Count Per Cluster")
plt.show()

# Create a count plot showing clusters related to feature values:
plt.figure(figsize=(16,10))
sns.countplot(x='Cluster', hue='driver_gender', data=X, palette='mako')
plt.legend(title='Driver Gender', labels=['Female', 'Male'])
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.title("Driver Gender Count Per Cluster")
plt.show()

# Create a cat plot showing age distribution for clusters by gender:
g = sns.catplot(x="driver_gender", y="driver_age", hue="Cluster", height=10, kind="boxen", palette="mako", data=X)
(g.set_xticklabels(['Female', 'Male'])
.set_axis_labels("Gender", "Age"))
g.set(title="Gender Age Distribution Per Cluster")

# Create a cat plot showing age distribution for clusters by race:
h = sns.catplot(x="driver_race_raw", y="driver_age", hue="Cluster", height=10, kind="boxen", palette="rocket", data=X)
(h.set_xticklabels(['Asian', 'Black Hispanic', 'Black', 'Other', 'Unknown (H)', 'Unknown (NH)', 'White Hispanic', 'White'])
.set_axis_labels("Race", "Age"))
h.set(title="Race Age Distribution Per Cluster")

## Pattern Analysis
# Create an attribute correlation:
nc_corr = nc_encoded.corr()
# Create a feature heat map using the correlation:
plt.figure(figsize=(16,10))
plt.title("Feature Correlation Heat Map")
mask = np.triu(np.ones(nc_corr.shape)).astype(np.bool)
sns.heatmap(nc_corr, mask=mask, cmap="GnBu", annot=True, vmin=-1, vmax=1, linewidths=0.2, fmt=".2f")

# Create a count plot showing stop outcomes by driver race:
plt.figure(figsize=(16,10))
j = sns.countplot(x='stop_outcome', hue='driver_race_raw', data=nc_encoded)
(j.set_xticklabels(['Arrest', 'Citation', 'No Action', 'Verbal Warning', 'Written Warning']))
plt.legend(title='Driver Race', labels=['Asian', 'Black Hispanic', 'Black', 'Other', 'Unknown Hispanic', 'Unknown Nonhispanic', 'White Hispanic', 'White'])
plt.xlabel("Stop Outcome")
plt.ylabel("Count")
j.set(title="Stop Outcomes By Driver Race")
plt.show()

# Create a count plot showing stop outcomes by driver gender:
plt.figure(figsize=(16,10))
j = sns.countplot(x='stop_outcome', hue='driver_gender', data=nc_encoded)
(j.set_xticklabels(['Arrest', 'Citation', 'No Action', 'Verbal Warning', 'Written Warning']))
plt.legend(title='Driver Gender', labels=['Female', 'Male'])
plt.xlabel("Stop Outcome")
plt.ylabel("Count")
j.set(title="Stop Outcomes By Driver Gender")
plt.show()

# Create a cat plot showing age distribution for stop outcomes by gender:
b = sns.catplot(x='stop_outcome', y='driver_age', hue='driver_gender', kind='boxen', height=10, data=nc_encoded)
b.set_xticklabels(['Arrest', 'Citation', 'No Action', 'Verbal Warning', 'Written Warning'])
b.set(title='Stop Outcome Age Distribution by Gender')
b.set_axis_labels('Stop Outcome', 'Driver Age')
b._legend.set_title('Driver Gender')
gender_labels=['Female', 'Male']
for t, l in zip(b._legend.texts, gender_labels): t.set_text(l)

# Create a cat plot showing age distribution for stop outcomes by race:
c = sns.catplot(x='stop_outcome', y='driver_age', hue='driver_race_raw', kind='boxen', legend_out=True, height=10,data=nc_encoded)
c.set_xticklabels(['Arrest', 'Citation', 'No Action', 'Verbal Warning', 'Written Warning'])
c.set(title='Stop Outcome Age Distribution by Race')
c.set_axis_labels('Stop Outcome', 'Driver Age')
c._legend.set_title('Driver Race')
new_labels=['Asian', 'Black Hispanic', 'Black', 'Other', 'Unknown Hispanic', 'Unknown Nonhispanic', 'White Hispanic', 'White']
for t, l in zip(c._legend.texts, new_labels): t.set_text(l)
c._legend.set_bbox_to_anchor((1.05, 0.5))

## Outlier Analysis
# Computer and add the 'Per_Capita' feature:
for i, row in stop_count.iterrows():
    stop_count.at[i, "Per_Capita"] = (stop_count.at[i, "Stop_Count"]/stop_count.at[i, "Population"])
    
# Create a scatter plot showing stop count versus per capita:
plt.figure(figsize=(16,10))
sns.scatterplot(x="Per_Capita", y="Stop_Count", hue="Population", palette="copper", size="Population", sizes=(100, 200), data=stop_count)
plt.xlabel("Stops Per Capita")
plt.ylabel("Stop Count")
plt.title("Stop Count Verus Per Capita")

# Create a box plot for the per capita values:
plt.figure(figsize=(16,8))
sns.boxplot(x=stop_count['Per_Capita'], color="red")
sns.stripplot(x=stop_count['Per_Capita'], color="black")
plt.title("Per Capita Stop Count Outlier Detection")
plt.xlabel("Stop Count Per Capita")

# Detecting Outliers Using Z-Score:
Z_outliers = stop_count[(np.abs(stats.zscore(stop_count['Per_Capita'])) > 3)]
Z_outliers.head()

# Detecting Outliers using IQR:
Q1 = stop_count['Per_Capita'].quantile(0.25)
Q3 = stop_count['Per_Capita'].quantile(0.75)
IQR = Q3 - Q1
IQR_outliers = stop_count[(stop_count['Per_Capita'] < (Q1 - 1.5 * IQR)) | (stop_count['Per_Capita'] > (Q3 + 1.5 * IQR))]
IQR_outliers.head()
