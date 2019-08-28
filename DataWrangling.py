import pandas as pd

floodingReport = pd.Series([5,6,7,8,12])
floodingReport

# set contry names as index
floodingReport  = pd.Series([5,6,7,8,12], index = ['India Country', 'France Country', 'Japan Country', 'USA Country', 'China Country'])
floodingReport

# view the no of flooding report in thr India Country
floodingReport['India Country']

# view no of countries with more than 6 flooding reports
floodingReport[floodingReport > 6]

# create a pandas series from a dictionary
floodingReport_dict = {'India Country': 12, 'France Country': 30, 'Japan Country': 354, 'USA Country': 42, 'China Country': 13}
floodingReport_dict

# convert the dictionary into pd series and view that
floodingReport = pd.Series(floodingReport_dict)
floodingReport

# change the index of a series to shorter names
floodingReport.Index = ['India', 'France', 'Japan', 'USA', 'China']
floodingReport.Index

# Create a dataframe from a dictionary of equal length lists or numpy array
data = {'country':['India', 'France', 'Japan', 'USA', 'China'], 'year':[2010,2011,2012,2013,2014], 'reports': [4,24,31,2,3]}
df = pd.DataFrame(data)
df

# Set the order of the columns using the column attributes
dfColumnOrdered = pd.DataFrame(data, columns=['country', 'year', 'reports'])
dfColumnOrdered

# Add column
dfColumnOrdered['newCoverage'] = pd.Series([42.5, 65.98, 12.2, 30.2, 92.1])
dfColumnOrdered
