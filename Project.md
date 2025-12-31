```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
df = pd.read_csv("Customer-churn.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>...</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>1</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>34</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.5</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>45</td>
      <td>No</td>
      <td>No phone service</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>2</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
df.shape
```




    (7043, 21)




```python
df.columns
```




    Index(['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
           'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
           'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
           'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
           'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn'],
          dtype='object')



Removing unrelated Features


```python
df.drop(['SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines', 'OnlineBackup', 'DeviceProtection','StreamingTV', 'StreamingMovies'],axis = 1,inplace = True)
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>tenure</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>TechSupport</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>Female</td>
      <td>1</td>
      <td>DSL</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>No</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>Male</td>
      <td>34</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.5</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>Male</td>
      <td>2</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>Male</td>
      <td>45</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>Female</td>
      <td>2</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>



Changing the Total charges string format to numeric


```python
df_churnYes['TotalCharges'] = pd.to_numeric(
    df_churnYes['TotalCharges'],
    errors='coerce'
)
```

    C:\Users\Abdul Hakim\AppData\Local\Temp\ipykernel_22336\2926151609.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_churnYes['TotalCharges'] = pd.to_numeric(
    


```python
df['gender'].value_counts()
```




    gender
    Male      3555
    Female    3488
    Name: count, dtype: int64




```python
def tenure_period(days):
    if days < 6:
        return "New"
    if days >= 6 and days <=24:
        return "Mid"
    if days > 24:
        return "Long"

df['tenure_periodstatus'] = df['tenure'].apply(tenure_period)
```


```python
df.groupby(['Churn','tenure_periodstatus'])['Contract'].value_counts()
```




    Churn  tenure_periodstatus  Contract      
    No     Long                 Two year          1489
                                One year          1015
                                Month-to-month     791
           Mid                  Month-to-month     851
                                One year           264
                                Two year           137
           New                  Month-to-month     578
                                One year            28
                                Two year            21
    Yes    Long                 Month-to-month     353
                                One year           137
                                Two year            48
           Mid                  Month-to-month     562
                                One year            25
           New                  Month-to-month     740
                                One year             4
    Name: count, dtype: int64




```python
df['Contract'].value_counts()
```




    Contract
    Month-to-month    3875
    Two year          1695
    One year          1473
    Name: count, dtype: int64




```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>tenure</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>TechSupport</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
      <th>tenure_periodstatus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>Female</td>
      <td>1</td>
      <td>DSL</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>No</td>
      <td>New</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>Male</td>
      <td>34</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.5</td>
      <td>No</td>
      <td>Long</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>Male</td>
      <td>2</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>Yes</td>
      <td>New</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>Male</td>
      <td>45</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>No</td>
      <td>Long</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>Female</td>
      <td>2</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>Yes</td>
      <td>New</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.groupby('Churn')['tenure_periodstatus'].value_counts()
```




    Churn  tenure_periodstatus
    No     Long                   3295
           Mid                    1252
           New                     627
    Yes    New                     744
           Mid                     587
           Long                    538
    Name: count, dtype: int64




```python
sns.countplot(data = df,hue='Churn',x ='tenure_periodstatus',palette = 'bright')
```




    <Axes: xlabel='tenure_periodstatus', ylabel='count'>




    
![png](output_16_1.png)
    



```python
df.groupby('Contract')['Churn'].value_counts()
```




    Contract        Churn
    Month-to-month  No       2220
                    Yes      1655
    One year        No       1307
                    Yes       166
    Two year        No       1647
                    Yes        48
    Name: count, dtype: int64




```python
sns.countplot(data=df, x='Contract', hue='Churn',palette = 'dark')
plt.title("Total Charges Distribution by Churn")
plt.show()
```


    
![png](output_18_0.png)
    



```python
df.groupby('Churn')['MonthlyCharges'].mean()
```




    Churn
    No     61.265124
    Yes    74.441332
    Name: MonthlyCharges, dtype: float64




```python
sns.histplot(data=df, x='MonthlyCharges', hue='Churn',multiple='stack',palette = 'bright')
plt.title("Total Charges Distribution by Churn")
plt.show()
```


    
![png](output_20_0.png)
    



```python
df.groupby('PaymentMethod')['Churn'].value_counts()
```




    PaymentMethod              Churn
    Bank transfer (automatic)  No       1286
                               Yes       258
    Credit card (automatic)    No       1290
                               Yes       232
    Electronic check           No       1294
                               Yes      1071
    Mailed check               No       1304
                               Yes       308
    Name: count, dtype: int64




```python
figure = plt.figure(figsize = (10,8))
sns.histplot(data=df, x='PaymentMethod', hue='Churn',multiple ='stack',shrink = 0.9)
plt.title("Total Charges Distribution by Churn")
plt.show()
```


    
![png](output_22_0.png)
    



```python
df.groupby('Churn')['InternetService'].value_counts()
```




    Churn  InternetService
    No     DSL                1962
           Fiber optic        1799
           No                 1413
    Yes    Fiber optic        1297
           DSL                 459
           No                  113
    Name: count, dtype: int64




```python
sns.histplot(data=df, x='InternetService', hue='Churn',multiple ='stack',shrink = 0.9,palette= 'rainbow')
plt.title("Total Charges Distribution by Churn")
plt.show()
```


    
![png](output_24_0.png)
    



```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>tenure</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>TechSupport</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
      <th>tenure_periodstatus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>Female</td>
      <td>1</td>
      <td>DSL</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>No</td>
      <td>New</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>Male</td>
      <td>34</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>One year</td>
      <td>No</td>
      <td>Mailed check</td>
      <td>56.95</td>
      <td>1889.5</td>
      <td>No</td>
      <td>Long</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>Male</td>
      <td>2</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Mailed check</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>Yes</td>
      <td>New</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>Male</td>
      <td>45</td>
      <td>DSL</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>No</td>
      <td>Long</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>Female</td>
      <td>2</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>Yes</td>
      <td>New</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_churnYes = df[df['Churn']=='Yes']
```


```python
df['Churn'].value_counts()
```




    Churn
    No     5174
    Yes    1869
    Name: count, dtype: int64




```python
df_churnYes['tenure_periodstatus'].value_counts()
```




    tenure_periodstatus
    New     744
    Mid     587
    Long    538
    Name: count, dtype: int64



# New tenure analysis


```python
df_churnYes[df_churnYes['tenure_periodstatus'] == 'New']['PaymentMethod'].value_counts()
```




    PaymentMethod
    Electronic check             437
    Mailed check                 202
    Bank transfer (automatic)     56
    Credit card (automatic)       49
    Name: count, dtype: int64




```python
df_churnYes[df_churnYes['tenure_periodstatus'] == 'New']['Contract'].value_counts()
```




    Contract
    Month-to-month    740
    One year            4
    Name: count, dtype: int64




```python
df_churnYes[df_churnYes['tenure_periodstatus'] == 'New']['InternetService'].value_counts()
```




    InternetService
    Fiber optic    431
    DSL            236
    No              77
    Name: count, dtype: int64




```python
df_churnYes[df_churnYes['tenure_periodstatus'] == 'New']['PaperlessBilling'].value_counts()
```




    PaperlessBilling
    Yes    516
    No     228
    Name: count, dtype: int64



Majority of customer having { Payment M => Electronic check: 437 ,contract => Month-to-month : 740 ,
InternetService => Fiber optic : 431 ,PaperlessBilling => 516}


```python
df_churnYes[(df_churnYes['tenure_periodstatus'] == 'New') &(df_churnYes['PaymentMethod'] == 'Electronic check') & (df_churnYes['Contract'] == 'Month-to-month') & (df_churnYes['InternetService'] == 'Fiber optic') & (df_churnYes['PaperlessBilling'] == 'Yes')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>tenure</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>TechSupport</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
      <th>tenure_periodstatus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>Female</td>
      <td>2</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>Yes</td>
      <td>New</td>
    </tr>
    <tr>
      <th>36</th>
      <td>6047-YHPVI</td>
      <td>Male</td>
      <td>5</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>69.70</td>
      <td>316.9</td>
      <td>Yes</td>
      <td>New</td>
    </tr>
    <tr>
      <th>47</th>
      <td>7760-OYPDY</td>
      <td>Female</td>
      <td>2</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>80.65</td>
      <td>144.15</td>
      <td>Yes</td>
      <td>New</td>
    </tr>
    <tr>
      <th>80</th>
      <td>5919-TMRGD</td>
      <td>Female</td>
      <td>1</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>79.35</td>
      <td>79.35</td>
      <td>Yes</td>
      <td>New</td>
    </tr>
    <tr>
      <th>122</th>
      <td>0404-SWRVG</td>
      <td>Male</td>
      <td>3</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>74.40</td>
      <td>229.55</td>
      <td>Yes</td>
      <td>New</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6933</th>
      <td>6502-MJQAE</td>
      <td>Male</td>
      <td>1</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>69.60</td>
      <td>69.6</td>
      <td>Yes</td>
      <td>New</td>
    </tr>
    <tr>
      <th>6936</th>
      <td>7693-LCKZL</td>
      <td>Male</td>
      <td>5</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>80.15</td>
      <td>385</td>
      <td>Yes</td>
      <td>New</td>
    </tr>
    <tr>
      <th>6970</th>
      <td>8083-YTZES</td>
      <td>Male</td>
      <td>4</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>74.35</td>
      <td>265.35</td>
      <td>Yes</td>
      <td>New</td>
    </tr>
    <tr>
      <th>7010</th>
      <td>0723-DRCLG</td>
      <td>Female</td>
      <td>1</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>74.45</td>
      <td>74.45</td>
      <td>Yes</td>
      <td>New</td>
    </tr>
    <tr>
      <th>7032</th>
      <td>6894-LFHLY</td>
      <td>Male</td>
      <td>1</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>75.75</td>
      <td>75.75</td>
      <td>Yes</td>
      <td>New</td>
    </tr>
  </tbody>
</table>
<p>256 rows × 13 columns</p>
</div>



Major problems of  256 New customers were churned because of Fiber optic (Internet Service) ,Electronic check (Payment Method) , Month-to-Month (Contract) and paperless billing 

# Mid Customer analysis


```python
df_churnYes[df_churnYes['tenure_periodstatus'] == 'Mid']['PaymentMethod'].value_counts()
```




    PaymentMethod
    Electronic check             353
    Bank transfer (automatic)     86
    Credit card (automatic)       74
    Mailed check                  74
    Name: count, dtype: int64




```python
df_churnYes[df_churnYes['tenure_periodstatus'] == 'Mid']['Contract'].value_counts()
```




    Contract
    Month-to-month    562
    One year           25
    Name: count, dtype: int64




```python
df_churnYes[df_churnYes['tenure_periodstatus'] == 'Mid']['InternetService'].value_counts()
```




    InternetService
    Fiber optic    433
    DSL            130
    No              24
    Name: count, dtype: int64




```python
df_churnYes[df_churnYes['tenure_periodstatus'] == 'Mid']['PaperlessBilling'].value_counts()
```




    PaperlessBilling
    Yes    460
    No     127
    Name: count, dtype: int64




```python
df_churnYes[(df_churnYes['tenure_periodstatus'] == 'Mid') &(df_churnYes['PaymentMethod'] == 'Electronic check') & (df_churnYes['Contract'] == 'Month-to-month') & (df_churnYes['InternetService'] == 'Fiber optic') & (df_churnYes['PaperlessBilling'] == 'Yes')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>tenure</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>TechSupport</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
      <th>tenure_periodstatus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>9305-CDSKC</td>
      <td>Female</td>
      <td>8</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>99.65</td>
      <td>820.5</td>
      <td>Yes</td>
      <td>Mid</td>
    </tr>
    <tr>
      <th>55</th>
      <td>1658-BYGOY</td>
      <td>Male</td>
      <td>18</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>95.45</td>
      <td>1752.55</td>
      <td>Yes</td>
      <td>Mid</td>
    </tr>
    <tr>
      <th>95</th>
      <td>8637-XJIVR</td>
      <td>Female</td>
      <td>12</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>78.95</td>
      <td>927.35</td>
      <td>Yes</td>
      <td>Mid</td>
    </tr>
    <tr>
      <th>117</th>
      <td>5299-RULOA</td>
      <td>Female</td>
      <td>10</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>100.25</td>
      <td>1064.65</td>
      <td>Yes</td>
      <td>Mid</td>
    </tr>
    <tr>
      <th>184</th>
      <td>1918-ZBFQJ</td>
      <td>Female</td>
      <td>13</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>79.25</td>
      <td>1111.65</td>
      <td>Yes</td>
      <td>Mid</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6861</th>
      <td>6692-UDPJC</td>
      <td>Female</td>
      <td>14</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>91.65</td>
      <td>1301</td>
      <td>Yes</td>
      <td>Mid</td>
    </tr>
    <tr>
      <th>6868</th>
      <td>1195-OIYEJ</td>
      <td>Male</td>
      <td>13</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>91.10</td>
      <td>1135.7</td>
      <td>Yes</td>
      <td>Mid</td>
    </tr>
    <tr>
      <th>6878</th>
      <td>2990-HWIML</td>
      <td>Female</td>
      <td>6</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>89.50</td>
      <td>573.3</td>
      <td>Yes</td>
      <td>Mid</td>
    </tr>
    <tr>
      <th>6953</th>
      <td>1564-NTYXF</td>
      <td>Female</td>
      <td>13</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>82.00</td>
      <td>1127.2</td>
      <td>Yes</td>
      <td>Mid</td>
    </tr>
    <tr>
      <th>7009</th>
      <td>7703-ZEKEF</td>
      <td>Male</td>
      <td>23</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>81.00</td>
      <td>1917.1</td>
      <td>Yes</td>
      <td>Mid</td>
    </tr>
  </tbody>
</table>
<p>245 rows × 13 columns</p>
</div>



Major problems of  245 Mid customers were churned because of Fiber optic (Internet Service) ,Electronic check (Payment Method) , Month-to-Month (Contract) and paperless billing 

# Long Customer analysis


```python
df_churnYes[df_churnYes['tenure_periodstatus'] == 'Long']['PaymentMethod'].value_counts()
```




    PaymentMethod
    Electronic check             281
    Bank transfer (automatic)    116
    Credit card (automatic)      109
    Mailed check                  32
    Name: count, dtype: int64




```python
df_churnYes[df_churnYes['tenure_periodstatus'] == 'Long']['Contract'].value_counts()
```




    Contract
    Month-to-month    353
    One year          137
    Two year           48
    Name: count, dtype: int64




```python
df_churnYes[df_churnYes['tenure_periodstatus'] == 'Long']['InternetService'].value_counts()
```




    InternetService
    Fiber optic    433
    DSL             93
    No              12
    Name: count, dtype: int64




```python
df_churnYes[df_churnYes['tenure_periodstatus'] == 'Long']['PaperlessBilling'].value_counts()
```




    PaperlessBilling
    Yes    424
    No     114
    Name: count, dtype: int64




```python
df_churnYes[(df_churnYes['tenure_periodstatus'] == 'Long') &(df_churnYes['PaymentMethod'] == 'Electronic check') & (df_churnYes['Contract'] == 'Month-to-month') & (df_churnYes['InternetService'] == 'Fiber optic') & (df_churnYes['PaperlessBilling'] == 'Yes')]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>tenure</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>TechSupport</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
      <th>tenure_periodstatus</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>7892-POOKP</td>
      <td>Female</td>
      <td>28</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>104.80</td>
      <td>3046.05</td>
      <td>Yes</td>
      <td>Long</td>
    </tr>
    <tr>
      <th>26</th>
      <td>6467-CHFZW</td>
      <td>Male</td>
      <td>47</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>99.35</td>
      <td>4749.15</td>
      <td>Yes</td>
      <td>Long</td>
    </tr>
    <tr>
      <th>38</th>
      <td>5380-WJKOV</td>
      <td>Male</td>
      <td>34</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>106.35</td>
      <td>3549.25</td>
      <td>Yes</td>
      <td>Long</td>
    </tr>
    <tr>
      <th>99</th>
      <td>4598-XLKNJ</td>
      <td>Female</td>
      <td>25</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>98.50</td>
      <td>2514.5</td>
      <td>Yes</td>
      <td>Long</td>
    </tr>
    <tr>
      <th>110</th>
      <td>0486-HECZI</td>
      <td>Male</td>
      <td>55</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>96.75</td>
      <td>5238.9</td>
      <td>Yes</td>
      <td>Long</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6914</th>
      <td>7142-HVGBG</td>
      <td>Male</td>
      <td>43</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>103.00</td>
      <td>4414.3</td>
      <td>Yes</td>
      <td>Long</td>
    </tr>
    <tr>
      <th>6926</th>
      <td>1450-SKCVI</td>
      <td>Female</td>
      <td>56</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>73.85</td>
      <td>4092.85</td>
      <td>Yes</td>
      <td>Long</td>
    </tr>
    <tr>
      <th>6962</th>
      <td>0886-QGENL</td>
      <td>Female</td>
      <td>27</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>101.25</td>
      <td>2754.45</td>
      <td>Yes</td>
      <td>Long</td>
    </tr>
    <tr>
      <th>6967</th>
      <td>8739-WWKDU</td>
      <td>Male</td>
      <td>25</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>89.50</td>
      <td>2196.15</td>
      <td>Yes</td>
      <td>Long</td>
    </tr>
    <tr>
      <th>6993</th>
      <td>6583-QGCSI</td>
      <td>Female</td>
      <td>50</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>88.05</td>
      <td>4367.35</td>
      <td>Yes</td>
      <td>Long</td>
    </tr>
  </tbody>
</table>
<p>166 rows × 13 columns</p>
</div>



# Monthly Charges Analysis 

Contract : Month to Month


```python
sns.histplot(data= df_churnYes[df_churnYes['Contract']=='Month-to-month'] ,x ='MonthlyCharges',palette='rainbow',bins=20,hue='PaymentMethod',multiple='stack')
plt.axvline(x=68,color ='red',linewidth = 3,label= 'x = 68 to 103')
plt.axvline(x=103,color ='red',linewidth = 3)
```




    <matplotlib.lines.Line2D at 0x215a5760f50>




    
![png](output_52_1.png)
    

Most people were churned having monthly charge in range ~68 to ~103 charges (Electronic Check)
# Total Charges Analysis

Contract : One Year and Two year


```python
sns.histplot(data= df_churnYes[(df_churnYes['Contract'] == 'One year') | (df_churnYes['Contract'] == 'Two year')], x ='TotalCharges',bins =20,palette ='Set2',hue='PaymentMethod',shrink=0.9,multiple='stack')
```




    <Axes: xlabel='TotalCharges', ylabel='Count'>




    
![png](output_56_1.png)
    



```python
sns.scatterplot(data= df_churnYes , x= 'TotalCharges',y = 'tenure_periodstatus')
```




    <Axes: xlabel='TotalCharges', ylabel='tenure_periodstatus'>




    
![png](output_57_1.png)
    



```python
MidTenure_Twoyear_TotalChargeavg = df_churnYes[(df_churnYes['Contract'] =='Two year') & (df_churnYes['tenure_periodstatus'] =='Mid')]['TotalCharges'].mean()
```


```python
NewTenure_Oneyear_TotalChargeavg = df_churnYes[(df_churnYes['Contract'] =='One year') & (df_churnYes['tenure_periodstatus'] =='New')]['TotalCharges'].mean()
```


```python
MidTenure_Oneyear_TotalChargeavg =df_churnYes[(df_churnYes['Contract'] =='One year') & (df_churnYes['tenure_periodstatus'] =='Mid')]['TotalCharges'].mean()
```


```python
LongTenure_Oneyear_TotalChargeavg = df_churnYes[(df_churnYes['Contract'] =='One year') & (df_churnYes['tenure_periodstatus'] =='Long')]['TotalCharges'].mean()
```


```python
NewTenure_Twoyear_TotalChargeavg = df_churnYes[(df_churnYes['Contract'] =='Two year') & (df_churnYes['tenure_periodstatus'] =='New')]['TotalCharges'].mean()
```


```python
LongTenure_Twoyear_TotalChargeavg = df_churnYes[(df_churnYes['Contract'] =='Two year') & (df_churnYes['tenure_periodstatus'] =='Long')]['TotalCharges'].mean()
```


```python
Average_TotalCharges_Churned = pd.DataFrame(
    {'One year' : [NewTenure_Oneyear_TotalChargeavg,MidTenure_Oneyear_TotalChargeavg,
                                                                  LongTenure_Oneyear_TotalChargeavg],
                                                     'Two year' : [NewTenure_Twoyear_TotalChargeavg,MidTenure_Twoyear_TotalChargeavg,
                                                                   LongTenure_Twoyear_TotalChargeavg]  
                                          },index = ['New','Mid','Long']
            )


```


```python
Average_TotalCharges_Churned.fillna(0)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>One year</th>
      <th>Two year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>New</th>
      <td>213.875000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Mid</th>
      <td>1011.648000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>Long</th>
      <td>4736.091241</td>
      <td>5432.363542</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python


```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```


```python

```
