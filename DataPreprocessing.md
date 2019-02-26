

```python
import pandas as pd
import numpy as np
import re, string
import nltk
import random
import sklearn
from nltk.collocations import *
from nltk import FreqDist, word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
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

## Downloaded Data and used code to convert to csv


```python
######## write a function to take all the json files i want to run this and do them for all. 
```

headers=True
####put file name here in json and file name to be written. This one is for reviews.json

filein = 'review.json'
fileout = 'reviews.csv'
with open(filein, encoding= 'cp866') as jsonf, open (fileout, 'w') as csvf:
    for line in jsonf:
        data = json.loads(line)
         
        if headers:
            keys =[]
            for k, v in data.items():
                keys.append(k)
            writer= csv.DictWriter(csvf, fieldnames=keys)
            writer.writeheader()
            headers=False
        writer. writerow(data)


```python
#### START #######
```


```python

```

# Bringing in Review DataSet and starting to clean 


```python
reviews= pd.read_csv('reviews.csv')
```


```python
reviews.head()
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
      <th>review_id</th>
      <th>user_id</th>
      <th>business_id</th>
      <th>stars</th>
      <th>useful</th>
      <th>funny</th>
      <th>cool</th>
      <th>text</th>
      <th>date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Q1sbwvVQXV2734tPgoKj4Q</td>
      <td>hG7b0MtEbXx5QzbzE6C_VA</td>
      <td>ujmEBvifdJM6h6RLv4wQIg</td>
      <td>1.0</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>Total bill for this horrible service? Over $8G...</td>
      <td>2013-05-07 04:34:36</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GJXCdrto3ASJOqKeVWPi6Q</td>
      <td>yXQM5uF2jS6es16SJzNHfg</td>
      <td>NZnhc2sEQy3RmzKTZnqtwQ</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>I *adore* Travis at the Hard Rock's new Kelly ...</td>
      <td>2017-01-14 21:30:33</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2TzJjDVDEuAW6MR5Vuc1ug</td>
      <td>n6-Gk65cPZL6Uz8qRm3NYw</td>
      <td>WTqjgwHlXbSFevF32_DJVw</td>
      <td>5.0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>I have to say that this office really has it t...</td>
      <td>2016-11-09 20:09:03</td>
    </tr>
    <tr>
      <th>3</th>
      <td>yi0R0Ugj_xUx_Nek0-_Qig</td>
      <td>dacAIZ6fTM6mqwW5uxkskg</td>
      <td>ikCg8xy5JIg_NGPx-MSIDA</td>
      <td>5.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>Went in for a lunch. Steak sandwich was delici...</td>
      <td>2018-01-09 20:56:38</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11a8sVPMUFtaC7_ABRkmtw</td>
      <td>ssoyf2_x0EQMed6fgHeMyQ</td>
      <td>b1b1eb3uo-w561D0ZfCEiQ</td>
      <td>1.0</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>Today was my second out of three sessions I ha...</td>
      <td>2018-01-30 23:07:38</td>
    </tr>
  </tbody>
</table>
</div>




```python
## checking to see how many reviews are classified as useful 

print (np.sum(reviews['useful'] > 0))
print (np.sum(reviews['useful'] > 1))
print (np.sum(reviews['useful'] > 2))
print (np.sum(reviews['useful'] > 3))
print (np.sum(reviews['useful'] > 4))
```

    3115446
    1684814
    1015774
    665966
    464194



```python
#checking len and shape 
reviews.shape
```




    (6685900, 9)




```python
### checking to see how many reviews fall into what catergory 

print (np.sum(reviews['stars'] == 1))
print (np.sum(reviews['stars'] == 2))
print (np.sum(reviews['stars'] == 3))
print (np.sum(reviews['stars'] == 4))
print (np.sum(reviews['stars'] == 5))
```

    1002159
    542394
    739280
    1468985
    2933082



```python
## Changed the reviews to drop columns that wont be used. 
reviews1= reviews.drop(['funny', 'cool','date'], axis=1)
```


```python
## checking updated df 
reviews1.head(1)
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
      <th>review_id</th>
      <th>user_id</th>
      <th>business_id</th>
      <th>stars</th>
      <th>useful</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Q1sbwvVQXV2734tPgoKj4Q</td>
      <td>hG7b0MtEbXx5QzbzE6C_VA</td>
      <td>ujmEBvifdJM6h6RLv4wQIg</td>
      <td>1.0</td>
      <td>6</td>
      <td>Total bill for this horrible service? Over $8G...</td>
    </tr>
  </tbody>
</table>
</div>




```python
reviews1.isna().sum()
```




    review_id      0
    user_id        0
    business_id    0
    stars          0
    useful         0
    text           2
    dtype: int64




```python
reviews1.dropna(inplace=True)
```


```python
reviews1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 6685898 entries, 0 to 6685899
    Data columns (total 6 columns):
    review_id      object
    user_id        object
    business_id    object
    stars          float64
    useful         int64
    text           object
    dtypes: float64(1), int64(1), object(4)
    memory usage: 357.1+ MB



```python
reviews1.shape

```




    (6685898, 6)




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

## Bringing in Business Dataset and starting to clean



```python
business = pd.read_csv('business.csv')
```


```python
business.head()
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
      <th>business_id</th>
      <th>name</th>
      <th>address</th>
      <th>city</th>
      <th>state</th>
      <th>postal_code</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>stars</th>
      <th>review_count</th>
      <th>is_open</th>
      <th>attributes</th>
      <th>categories</th>
      <th>hours</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1SWheh84yJXfytovILXOAQ</td>
      <td>Arizona Biltmore Golf Club</td>
      <td>2818 E Camino Acequia Drive</td>
      <td>Phoenix</td>
      <td>AZ</td>
      <td>85016</td>
      <td>33.522143</td>
      <td>-112.018481</td>
      <td>3.0</td>
      <td>5</td>
      <td>0</td>
      <td>{'GoodForKids': 'False'}</td>
      <td>Golf, Active Life</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>QXAEGFB4oINsVuTFxEYKFQ</td>
      <td>Emerald Chinese Restaurant</td>
      <td>30 Eglinton Avenue W</td>
      <td>Mississauga</td>
      <td>ON</td>
      <td>L5R 3E7</td>
      <td>43.605499</td>
      <td>-79.652289</td>
      <td>2.5</td>
      <td>128</td>
      <td>1</td>
      <td>{'RestaurantsReservations': 'True', 'GoodForMe...</td>
      <td>Specialty Food, Restaurants, Dim Sum, Imported...</td>
      <td>{'Monday': '9:0-0:0', 'Tuesday': '9:0-0:0', 'W...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>gnKjwL_1w79qoiV3IC_xQQ</td>
      <td>Musashi Japanese Restaurant</td>
      <td>10110 Johnston Rd, Ste 15</td>
      <td>Charlotte</td>
      <td>NC</td>
      <td>28210</td>
      <td>35.092564</td>
      <td>-80.859132</td>
      <td>4.0</td>
      <td>170</td>
      <td>1</td>
      <td>{'GoodForKids': 'True', 'NoiseLevel': "u'avera...</td>
      <td>Sushi Bars, Restaurants, Japanese</td>
      <td>{'Monday': '17:30-21:30', 'Wednesday': '17:30-...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>xvX2CttrVhyG2z1dFg_0xw</td>
      <td>Farmers Insurance - Paul Lorenz</td>
      <td>15655 W Roosevelt St, Ste 237</td>
      <td>Goodyear</td>
      <td>AZ</td>
      <td>85338</td>
      <td>33.455613</td>
      <td>-112.395596</td>
      <td>5.0</td>
      <td>3</td>
      <td>1</td>
      <td>NaN</td>
      <td>Insurance, Financial Services</td>
      <td>{'Monday': '8:0-17:0', 'Tuesday': '8:0-17:0', ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>HhyxOkGAM07SRYtlQ4wMFQ</td>
      <td>Queen City Plumbing</td>
      <td>4209 Stuart Andrew Blvd, Ste F</td>
      <td>Charlotte</td>
      <td>NC</td>
      <td>28217</td>
      <td>35.190012</td>
      <td>-80.887223</td>
      <td>4.0</td>
      <td>4</td>
      <td>1</td>
      <td>{'BusinessAcceptsBitcoin': 'False', 'ByAppoint...</td>
      <td>Plumbing, Shopping, Local Services, Home Servi...</td>
      <td>{'Monday': '7:0-23:0', 'Tuesday': '7:0-23:0', ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
business.shape
```




    (192609, 14)




```python
business1= business.drop(['address','postal_code','latitude','longitude','is_open','attributes','hours'],axis=1)
```


```python
business1.head()
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
      <th>index</th>
      <th>business_id</th>
      <th>name</th>
      <th>city</th>
      <th>state</th>
      <th>stars</th>
      <th>review_count</th>
      <th>categories</th>
      <th>is_restaurant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1SWheh84yJXfytovILXOAQ</td>
      <td>Arizona Biltmore Golf Club</td>
      <td>Phoenix</td>
      <td>AZ</td>
      <td>3.0</td>
      <td>5</td>
      <td>Golf, Active Life</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>QXAEGFB4oINsVuTFxEYKFQ</td>
      <td>Emerald Chinese Restaurant</td>
      <td>Mississauga</td>
      <td>ON</td>
      <td>2.5</td>
      <td>128</td>
      <td>Specialty Food, Restaurants, Dim Sum, Imported...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>gnKjwL_1w79qoiV3IC_xQQ</td>
      <td>Musashi Japanese Restaurant</td>
      <td>Charlotte</td>
      <td>NC</td>
      <td>4.0</td>
      <td>170</td>
      <td>Sushi Bars, Restaurants, Japanese</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>xvX2CttrVhyG2z1dFg_0xw</td>
      <td>Farmers Insurance - Paul Lorenz</td>
      <td>Goodyear</td>
      <td>AZ</td>
      <td>5.0</td>
      <td>3</td>
      <td>Insurance, Financial Services</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>HhyxOkGAM07SRYtlQ4wMFQ</td>
      <td>Queen City Plumbing</td>
      <td>Charlotte</td>
      <td>NC</td>
      <td>4.0</td>
      <td>4</td>
      <td>Plumbing, Shopping, Local Services, Home Servi...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
business1.isna().sum()
```




    index            0
    business_id      0
    name             0
    city             0
    state            0
    stars            0
    review_count     0
    categories       0
    is_restaurant    0
    dtype: int64




```python
business1.dropna(inplace=True)
```


```python
business1.isna().sum()
```




    business_id     0
    name            0
    city            0
    state           0
    stars           0
    review_count    0
    categories      0
    dtype: int64




```python
business1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 192126 entries, 0 to 192608
    Data columns (total 7 columns):
    business_id     192126 non-null object
    name            192126 non-null object
    city            192126 non-null object
    state           192126 non-null object
    stars           192126 non-null float64
    review_count    192126 non-null int64
    categories      192126 non-null object
    dtypes: float64(1), int64(1), object(5)
    memory usage: 11.7+ MB



```python
business1= business1.reset_index()
```


```python
business1.head(1)
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
      <th>index</th>
      <th>business_id</th>
      <th>name</th>
      <th>city</th>
      <th>state</th>
      <th>stars</th>
      <th>review_count</th>
      <th>categories</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1SWheh84yJXfytovILXOAQ</td>
      <td>Arizona Biltmore Golf Club</td>
      <td>Phoenix</td>
      <td>AZ</td>
      <td>3.0</td>
      <td>5</td>
      <td>Golf, Active Life</td>
    </tr>
  </tbody>
</table>
</div>




```python
business1[business1.categories == '[]']
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
      <th>index</th>
      <th>business_id</th>
      <th>name</th>
      <th>city</th>
      <th>state</th>
      <th>stars</th>
      <th>review_count</th>
      <th>categories</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
business_list= list(business1['categories'])
```


```python
type(business_list)

```




    list




```python
len(business_list)
```




    192126




```python
business2= business1
```


```python
business2.head()
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
      <th>index</th>
      <th>business_id</th>
      <th>name</th>
      <th>city</th>
      <th>state</th>
      <th>stars</th>
      <th>review_count</th>
      <th>categories</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1SWheh84yJXfytovILXOAQ</td>
      <td>Arizona Biltmore Golf Club</td>
      <td>Phoenix</td>
      <td>AZ</td>
      <td>3.0</td>
      <td>5</td>
      <td>Golf, Active Life</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>QXAEGFB4oINsVuTFxEYKFQ</td>
      <td>Emerald Chinese Restaurant</td>
      <td>Mississauga</td>
      <td>ON</td>
      <td>2.5</td>
      <td>128</td>
      <td>Specialty Food, Restaurants, Dim Sum, Imported...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>gnKjwL_1w79qoiV3IC_xQQ</td>
      <td>Musashi Japanese Restaurant</td>
      <td>Charlotte</td>
      <td>NC</td>
      <td>4.0</td>
      <td>170</td>
      <td>Sushi Bars, Restaurants, Japanese</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>xvX2CttrVhyG2z1dFg_0xw</td>
      <td>Farmers Insurance - Paul Lorenz</td>
      <td>Goodyear</td>
      <td>AZ</td>
      <td>5.0</td>
      <td>3</td>
      <td>Insurance, Financial Services</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>HhyxOkGAM07SRYtlQ4wMFQ</td>
      <td>Queen City Plumbing</td>
      <td>Charlotte</td>
      <td>NC</td>
      <td>4.0</td>
      <td>4</td>
      <td>Plumbing, Shopping, Local Services, Home Servi...</td>
    </tr>
  </tbody>
</table>
</div>




```python
business2['is_restaurant'] =0
```


```python
business2.head(2)
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
      <th>index</th>
      <th>business_id</th>
      <th>name</th>
      <th>city</th>
      <th>state</th>
      <th>stars</th>
      <th>review_count</th>
      <th>categories</th>
      <th>is_restaurant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1SWheh84yJXfytovILXOAQ</td>
      <td>Arizona Biltmore Golf Club</td>
      <td>Phoenix</td>
      <td>AZ</td>
      <td>3.0</td>
      <td>5</td>
      <td>Golf, Active Life</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>QXAEGFB4oINsVuTFxEYKFQ</td>
      <td>Emerald Chinese Restaurant</td>
      <td>Mississauga</td>
      <td>ON</td>
      <td>2.5</td>
      <td>128</td>
      <td>Specialty Food, Restaurants, Dim Sum, Imported...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
business2.loc[business2['categories'].str.contains('Restaurants'), 'is_restaurant'] = 1


# for loop took to long 
# for i in range(192126):
#     if "Restaurants" in business2['categories'][i]:
#         business2.loc[business2.is_restaurant][i] = 1
#     else:
#         business2.loc[business2.is_restaurant][i] = 0 
```


```python
business2.head(2)
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
      <th>index</th>
      <th>business_id</th>
      <th>name</th>
      <th>city</th>
      <th>state</th>
      <th>stars</th>
      <th>review_count</th>
      <th>categories</th>
      <th>is_restaurant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1SWheh84yJXfytovILXOAQ</td>
      <td>Arizona Biltmore Golf Club</td>
      <td>Phoenix</td>
      <td>AZ</td>
      <td>3.0</td>
      <td>5</td>
      <td>Golf, Active Life</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>QXAEGFB4oINsVuTFxEYKFQ</td>
      <td>Emerald Chinese Restaurant</td>
      <td>Mississauga</td>
      <td>ON</td>
      <td>2.5</td>
      <td>128</td>
      <td>Specialty Food, Restaurants, Dim Sum, Imported...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
business2[1000:1019]
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
      <th>index</th>
      <th>business_id</th>
      <th>name</th>
      <th>city</th>
      <th>state</th>
      <th>stars</th>
      <th>review_count</th>
      <th>categories</th>
      <th>is_restaurant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1000</th>
      <td>1002</td>
      <td>YH8Bn-7pLR-SFR8MCgQj1w</td>
      <td>Jules Cafe Patisserie</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>3.5</td>
      <td>44</td>
      <td>Bakeries, Food, Mediterranean, French, Restaur...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1001</th>
      <td>1003</td>
      <td>RLyqeVI4a-019BRK-9IhzQ</td>
      <td>Angelo's Cobbler Shoppe</td>
      <td>Cleveland</td>
      <td>OH</td>
      <td>5.0</td>
      <td>3</td>
      <td>Shoe Repair, Local Services</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1002</th>
      <td>1004</td>
      <td>OY1kLAhs9I6Ix4wUmSNAfQ</td>
      <td>Morgane Bistro &amp; Pub</td>
      <td>Saint-Jean-sur-Richelieu</td>
      <td>QC</td>
      <td>4.0</td>
      <td>4</td>
      <td>Pubs, Gastropubs, Restaurants, Nightlife, Bars</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1003</th>
      <td>1005</td>
      <td>4FE9FM8uvn9CV76Bxfxogw</td>
      <td>T&amp;T Roti Trinidad &amp; Toronto Roti</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>4.0</td>
      <td>9</td>
      <td>Caribbean, Restaurants</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1004</th>
      <td>1006</td>
      <td>J0P152h7wimvdJ-aV0QLug</td>
      <td>Stone Creek Coffee</td>
      <td>Madison</td>
      <td>WI</td>
      <td>4.5</td>
      <td>38</td>
      <td>Food, Coffee &amp; Tea</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1005</th>
      <td>1007</td>
      <td>nXL0quDqN4YYSOyYhcQByw</td>
      <td>Holt Window Cleaning</td>
      <td>Chandler</td>
      <td>AZ</td>
      <td>5.0</td>
      <td>4</td>
      <td>Window Washing, Home Services, Home &amp; Garden, ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1006</th>
      <td>1008</td>
      <td>pvXdMR9tcQlwXcXJLllmPg</td>
      <td>Kangaroo Express</td>
      <td>Rock Hill</td>
      <td>SC</td>
      <td>4.0</td>
      <td>3</td>
      <td>Automotive, Convenience Stores, Gas Stations, ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1007</th>
      <td>1009</td>
      <td>nYvBZYg9rfqWFTYuxSVMdw</td>
      <td>Cantina Laredo</td>
      <td>Westlake</td>
      <td>OH</td>
      <td>4.0</td>
      <td>62</td>
      <td>Mexican, Bars, Restaurants, Vegetarian, Nightlife</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1008</th>
      <td>1010</td>
      <td>dDr4b8AKwpOiP_-JsQ93vg</td>
      <td>The Chocolate Bar</td>
      <td>Calgary</td>
      <td>AB</td>
      <td>3.0</td>
      <td>7</td>
      <td>Food, Desserts</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1009</th>
      <td>1011</td>
      <td>OiZsIGscvwlL_0yhWmjJtw</td>
      <td>Pacific East</td>
      <td>Solon</td>
      <td>OH</td>
      <td>3.5</td>
      <td>11</td>
      <td>Food, Malaysian, Restaurants, Japanese</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1010</th>
      <td>1012</td>
      <td>zBRYyvJCfmqK5Zq-bkqhgw</td>
      <td>Advance Auto Parts</td>
      <td>Bedford</td>
      <td>OH</td>
      <td>3.0</td>
      <td>3</td>
      <td>Auto Parts &amp; Supplies, Automotive</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1011</th>
      <td>1013</td>
      <td>zsSGQz1EAxrnQ9xMCUCowA</td>
      <td>Lush Hair Style Bar</td>
      <td>Henderson</td>
      <td>NV</td>
      <td>5.0</td>
      <td>3</td>
      <td>Hair Salons, Blow Dry/Out Services, Beauty &amp; S...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1012</th>
      <td>1014</td>
      <td>uNxANKBV3_rnkPYfIy9WKA</td>
      <td>Hampton Inn &amp; Suites Phoenix Gilbert</td>
      <td>Gilbert</td>
      <td>AZ</td>
      <td>4.0</td>
      <td>52</td>
      <td>Hotels &amp; Travel, Hotels, Event Planning &amp; Serv...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1013</th>
      <td>1015</td>
      <td>-ubVkCv_U4815t3ibaIuhg</td>
      <td>Clothes Minded</td>
      <td>Phoenix</td>
      <td>AZ</td>
      <td>5.0</td>
      <td>13</td>
      <td>Jewelry, Shopping, Fashion, Women's Clothing, ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1014</th>
      <td>1016</td>
      <td>-NaIIkbUsm0-L0oL9r6VTg</td>
      <td>Ouest Antonio's</td>
      <td>Laval</td>
      <td>QC</td>
      <td>3.5</td>
      <td>12</td>
      <td>Pizza, Restaurants, Sandwiches</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1015</th>
      <td>1017</td>
      <td>DQfZM9b3n3DDMzEg4b4WhA</td>
      <td>Levin Furniture</td>
      <td>Wexford</td>
      <td>PA</td>
      <td>3.0</td>
      <td>25</td>
      <td>Home Services, Shopping, Mattresses, Interior ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1016</th>
      <td>1018</td>
      <td>CMhtlIP3-XbrWUzcA3JKpQ</td>
      <td>Consignments On Centre</td>
      <td>Pittsburgh</td>
      <td>PA</td>
      <td>2.0</td>
      <td>7</td>
      <td>Fashion, Accessories, Shopping, Women's Clothing</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1017</th>
      <td>1019</td>
      <td>sFGuQfzox4oOp0JhfgX1zg</td>
      <td>Quality Alterations</td>
      <td>Huntersville</td>
      <td>NC</td>
      <td>2.5</td>
      <td>16</td>
      <td>Sewing &amp; Alterations, Local Services</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1018</th>
      <td>1021</td>
      <td>p-8PgN7S4VUUXH6y5sDV1Q</td>
      <td>America's Best Contacts &amp; Eyeglasses</td>
      <td>Chandler</td>
      <td>AZ</td>
      <td>3.0</td>
      <td>38</td>
      <td>Doctors, Eyewear &amp; Opticians, Optometrists, Op...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
np.sum(business2['is_restaurant'] == 1)
```




    59371




```python
business2.loc[business2['categories'].str.contains('Food'), 'is_restaurant'] = 1
```


```python
np.sum(business2['is_restaurant'] == 1)
```




    74587




```python
business2= business2[business2.is_restaurant == 1]
```


```python
business2= business2.reset_index()
```


```python
business2.head(2)
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
      <th>level_0</th>
      <th>index</th>
      <th>business_id</th>
      <th>name</th>
      <th>city</th>
      <th>state</th>
      <th>stars</th>
      <th>review_count</th>
      <th>categories</th>
      <th>is_restaurant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>QXAEGFB4oINsVuTFxEYKFQ</td>
      <td>Emerald Chinese Restaurant</td>
      <td>Mississauga</td>
      <td>ON</td>
      <td>2.5</td>
      <td>128</td>
      <td>Specialty Food, Restaurants, Dim Sum, Imported...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>gnKjwL_1w79qoiV3IC_xQQ</td>
      <td>Musashi Japanese Restaurant</td>
      <td>Charlotte</td>
      <td>NC</td>
      <td>4.0</td>
      <td>170</td>
      <td>Sushi Bars, Restaurants, Japanese</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
business2= business2.drop(['index','categories','is_restaurant'], axis=1)
```


```python
business2.head(2)
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
      <th>level_0</th>
      <th>business_id</th>
      <th>name</th>
      <th>city</th>
      <th>state</th>
      <th>stars</th>
      <th>review_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>QXAEGFB4oINsVuTFxEYKFQ</td>
      <td>Emerald Chinese Restaurant</td>
      <td>Mississauga</td>
      <td>ON</td>
      <td>2.5</td>
      <td>128</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>gnKjwL_1w79qoiV3IC_xQQ</td>
      <td>Musashi Japanese Restaurant</td>
      <td>Charlotte</td>
      <td>NC</td>
      <td>4.0</td>
      <td>170</td>
    </tr>
  </tbody>
</table>
</div>




```python
business2= business2.drop(['level_0'], axis=1)
```


```python
business2.head(2)
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
      <th>business_id</th>
      <th>name</th>
      <th>city</th>
      <th>state</th>
      <th>stars</th>
      <th>review_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>QXAEGFB4oINsVuTFxEYKFQ</td>
      <td>Emerald Chinese Restaurant</td>
      <td>Mississauga</td>
      <td>ON</td>
      <td>2.5</td>
      <td>128</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gnKjwL_1w79qoiV3IC_xQQ</td>
      <td>Musashi Japanese Restaurant</td>
      <td>Charlotte</td>
      <td>NC</td>
      <td>4.0</td>
      <td>170</td>
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
####### trying to sepearte out the catgories columns #######




```


```python
business3=business2
```


```python
business3[35900:35905]
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
      <th>level_0</th>
      <th>index</th>
      <th>business_id</th>
      <th>name</th>
      <th>city</th>
      <th>state</th>
      <th>stars</th>
      <th>review_count</th>
      <th>categories</th>
      <th>is_restaurant</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35900</th>
      <td>92576</td>
      <td>92806</td>
      <td>jG46RAFm2kWshUIZ4HoJtg</td>
      <td>Tadka Sizzles</td>
      <td>Phoenix</td>
      <td>AZ</td>
      <td>3.0</td>
      <td>11</td>
      <td>Fast Food, Event Planning &amp; Services, Indian, ...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35901</th>
      <td>92579</td>
      <td>92809</td>
      <td>YszwELGfyPgPhjy2UJwAxQ</td>
      <td>Later Tater</td>
      <td>Gilbert</td>
      <td>AZ</td>
      <td>4.0</td>
      <td>8</td>
      <td>Food, Food Trucks</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35902</th>
      <td>92580</td>
      <td>92810</td>
      <td>SbxJg9yYyIclmPDSur1FCQ</td>
      <td>Golden Hawaiian BBQ</td>
      <td>Chandler</td>
      <td>AZ</td>
      <td>3.5</td>
      <td>134</td>
      <td>Hawaiian, Asian Fusion, Barbeque, Restaurants</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35903</th>
      <td>92582</td>
      <td>92812</td>
      <td>TFhZOVGcViyYZkxJ62X-dQ</td>
      <td>CoCo</td>
      <td>Markham</td>
      <td>ON</td>
      <td>3.5</td>
      <td>29</td>
      <td>Juice Bars &amp; Smoothies, Bubble Tea, Tea Rooms,...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>35904</th>
      <td>92583</td>
      <td>92813</td>
      <td>IhNASEZ3XnBHmuuVnWdIwA</td>
      <td>Brew Tea Bar</td>
      <td>Las Vegas</td>
      <td>NV</td>
      <td>5.0</td>
      <td>1506</td>
      <td>Tea Rooms, Desserts, Cafes, Restaurants, Food,...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#code to expand the categories and expand them out 
```


```python
def expand_categories(df, cat_var, key):
    all_cats = df[cat_var].str.cat(sep=', ')
    all_cats = all_cats.replace('[', '')
    all_cats = all_cats.replace(']', '')
    all_cats = all_cats.replace("\'","")
    all_cats = all_cats.replace('"','')
    all_cats_list = all_cats.split(', ')
    unique_cats = list(set(all_cats_list))
    unique_cats.remove('Restaurants')
    unique_cats.remove('Food')
    df_cats = pd.DataFrame(index=df[key], columns=unique_cats, data=False)
    df_out = df.merge(df_cats, how='left', left_on=key, right_index=True)
    for cat in unique_cats:
        df_out[cat] = df_out[cat_var].str.contains(cat)
    return df_out 
```


```python

```


```python

```


```python

```


```python
restaurants_df = expand_categories(business3, 'categories', 'business_id')
restaurants_df.drop('categories', axis=1, inplace=True)
```

    /home/mubarakb/miniconda3/lib/python3.7/site-packages/ipykernel_launcher.py:14: UserWarning: This pattern has match groups. To actually get the groups, use str.extract.
      



```python
restaurants_df[16000:16005]
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
      <th>level_0</th>
      <th>index</th>
      <th>business_id</th>
      <th>name</th>
      <th>city</th>
      <th>state</th>
      <th>stars</th>
      <th>review_count</th>
      <th>is_restaurant</th>
      <th>Mass Media</th>
      <th>Hawaiian</th>
      <th>Libraries</th>
      <th>Speech Therapists</th>
      <th>Southern</th>
      <th>Pet Stores</th>
      <th>Masonry/Concrete</th>
      <th>Eastern European</th>
      <th>Eyebrow Services</th>
      <th>Firewood</th>
      <th>Sushi Bars</th>
      <th>Wineries</th>
      <th>Egyptian</th>
      <th>Island Pub</th>
      <th>Wine Bars</th>
      <th>Family Practice</th>
      <th>Sicilian</th>
      <th>Officiants</th>
      <th>Eatertainment</th>
      <th>Hookah Bars</th>
      <th>Computers</th>
      <th>Fast Food</th>
      <th>Rotisserie Chicken</th>
      <th>Dive Bars</th>
      <th>Teeth Whitening</th>
      <th>Trusts</th>
      <th>Golf Cart Rentals</th>
      <th>Holiday Decorations</th>
      <th>Airsoft</th>
      <th>Transmission Repair</th>
      <th>Furniture Reupholstery</th>
      <th>Personal Shopping</th>
      <th>Hospitals</th>
      <th>Boat Dealers</th>
      <th>Ethnic Food</th>
      <th>Medical Spas</th>
      <th>Oil Change Stations</th>
      <th>Video Game Stores</th>
      <th>Backshop</th>
      <th>Electronics</th>
      <th>Bar Crawl</th>
      <th>Limos</th>
      <th>Senegalese</th>
      <th>Airport Shuttles</th>
      <th>Gastropubs</th>
      <th>Supernatural Readings</th>
      <th>Spiritual Shop</th>
      <th>Eyewear &amp; Opticians</th>
      <th>Medical Supplies</th>
      <th>Iberian</th>
      <th>Plus Size Fashion</th>
      <th>Macarons</th>
      <th>Real Estate Services</th>
      <th>Tax Services</th>
      <th>Knitting Supplies</th>
      <th>Pan Asian</th>
      <th>Foundation Repair</th>
      <th>Electricians</th>
      <th>Burmese</th>
      <th>Pop-Up Restaurants</th>
      <th>Art Classes</th>
      <th>Udon</th>
      <th>Skydiving</th>
      <th>Luggage</th>
      <th>Czech</th>
      <th>Massage Therapy</th>
      <th>Nicaraguan</th>
      <th>Thai</th>
      <th>Caterers</th>
      <th>Wine Tours</th>
      <th>Check Cashing/Pay-day Loans</th>
      <th>Hobby Shops</th>
      <th>Mongolian</th>
      <th>Vape Shops</th>
      <th>Fruits &amp; Veggies</th>
      <th>Market Stalls</th>
      <th>Beer Tours</th>
      <th>Golf</th>
      <th>RV Parks</th>
      <th>Stadiums &amp; Arenas</th>
      <th>Chiropractors</th>
      <th>Shopping</th>
      <th>Patisserie/Cake Shop</th>
      <th>Hydrotherapy</th>
      <th>Printing Services</th>
      <th>Emergency Medicine</th>
      <th>Trailer Repair</th>
      <th>Tobacco Shops</th>
      <th>Airport Terminals</th>
      <th>Turkish</th>
      <th>Tanning Beds</th>
      <th>Summer Camps</th>
      <th>Churches</th>
      <th>Septic Services</th>
      <th>Musicians</th>
      <th>Colleges &amp; Universities</th>
      <th>Restaurant Supplies</th>
      <th>Personal Chefs</th>
      <th>Pilates</th>
      <th>Pubs</th>
      <th>Russian</th>
      <th>Tickets</th>
      <th>Architectural Tours</th>
      <th>Signmaking</th>
      <th>Laotian</th>
      <th>Kitchen &amp; Bath</th>
      <th>Donairs</th>
      <th>Strip Clubs</th>
      <th>Peruvian</th>
      <th>Hiking</th>
      <th>General Dentistry</th>
      <th>Donuts</th>
      <th>University Housing</th>
      <th>Dance Schools</th>
      <th>Dance Wear</th>
      <th>Fireplace Services</th>
      <th>Pressure Washers</th>
      <th>Campgrounds</th>
      <th>Falafel</th>
      <th>Fences &amp; Gates</th>
      <th>Roadside Assistance</th>
      <th>Pakistani</th>
      <th>Home Window Tinting</th>
      <th>Persian/Iranian</th>
      <th>Body Shops</th>
      <th>Cardiologists</th>
      <th>Antiques</th>
      <th>Indian</th>
      <th>Adult</th>
      <th>Haunted Houses</th>
      <th>Glass &amp; Mirrors</th>
      <th>Oaxacan</th>
      <th>Watches</th>
      <th>Irish Pub</th>
      <th>Pharmacy</th>
      <th>Shoe Stores</th>
      <th>Rehabilitation Center</th>
      <th>Day Spas</th>
      <th>Empanadas</th>
      <th>Real Estate Agents</th>
      <th>Laundry Services</th>
      <th>Afghan</th>
      <th>Psychics</th>
      <th>Water Suppliers</th>
      <th>Hot Tub &amp; Pool</th>
      <th>Cards &amp; Stationery</th>
      <th>Ticket Sales</th>
      <th>Venezuelan</th>
      <th>Graphic Design</th>
      <th>Halal</th>
      <th>Screen Printing/T-Shirt Printing</th>
      <th>Shanghainese</th>
      <th>Scandinavian</th>
      <th>Hotels</th>
      <th>Comedy Clubs</th>
      <th>Tasting Classes</th>
      <th>Gas Stations</th>
      <th>Car Buyers</th>
      <th>Sleep Specialists</th>
      <th>Tours</th>
      <th>Landscaping</th>
      <th>Propane</th>
      <th>Smog Check Stations</th>
      <th>Holistic Animal Care</th>
      <th>Bistros</th>
      <th>Arcades</th>
      <th>Ophthalmologists</th>
      <th>Health Coach</th>
      <th>Bars</th>
      <th>Meat Shops</th>
      <th>Wholesalers</th>
      <th>British</th>
      <th>Australian</th>
      <th>Notaries</th>
      <th>Professional Services</th>
      <th>Town Car Service</th>
      <th>Doulas</th>
      <th>Anesthesiologists</th>
      <th>Party Equipment Rentals</th>
      <th>Shopping Centers</th>
      <th>Belgian</th>
      <th>Home Cleaning</th>
      <th>Tattoo</th>
      <th>Mauritius</th>
      <th>Brazilian</th>
      <th>Midwives</th>
      <th>Photographers</th>
      <th>Ice Cream &amp; Frozen Yogurt</th>
      <th>Batting Cages</th>
      <th>Vehicle Wraps</th>
      <th>Courthouses</th>
      <th>Transportation</th>
      <th>Vietnamese</th>
      <th>Alternative Medicine</th>
      <th>Barbers</th>
      <th>Bus Tours</th>
      <th>Ethiopian</th>
      <th>Syrian</th>
      <th>Hair Stylists</th>
      <th>Metal Fabricators</th>
      <th>Coffeeshops</th>
      <th>Breakfast &amp; Brunch</th>
      <th>Kombucha</th>
      <th>Puerto Rican</th>
      <th>Recording &amp; Rehearsal Studios</th>
      <th>Furniture Stores</th>
      <th>Shaved Snow</th>
      <th>General Festivals</th>
      <th>Auto Parts &amp; Supplies</th>
      <th>Financial Services</th>
      <th>Cupcakes</th>
      <th>Pets</th>
      <th>Filipino</th>
      <th>Wedding Planning</th>
      <th>Cosmetic Dentists</th>
      <th>Argentine</th>
      <th>Dry Cleaning</th>
      <th>Race Tracks</th>
      <th>Jewelry</th>
      <th>Legal Services</th>
      <th>Tanning</th>
      <th>Poutineries</th>
      <th>Appliances &amp; Repair</th>
      <th>Cardio Classes</th>
      <th>Costumes</th>
      <th>Contractors</th>
      <th>Animal Physical Therapy</th>
      <th>Wheel &amp; Rim Repair</th>
      <th>Mobile Phone Repair</th>
      <th>Salvadoran</th>
      <th>Bounce House Rentals</th>
      <th>Event Photography</th>
      <th>Olive Oil</th>
      <th>Cannabis Clinics</th>
      <th>Auto Glass Services</th>
      <th>Head Shops</th>
      <th>Honey</th>
      <th>International</th>
      <th>Butcher</th>
      <th>Candy Stores</th>
      <th>Skin Care</th>
      <th>Art Supplies</th>
      <th>Hair Extensions</th>
      <th>Bird Shops</th>
      <th>Mountain Biking</th>
      <th>Game Meat</th>
      <th>Tabletop Games</th>
      <th>Sports Clubs</th>
      <th>Customs Brokers</th>
      <th>Used Bookstore</th>
      <th>Christmas Markets</th>
      <th>Car Share Services</th>
      <th>Pain Management</th>
      <th>Beauty &amp; Spas</th>
      <th>Pet Services</th>
      <th>Cannabis Dispensaries</th>
      <th>Beer</th>
      <th>Vintage &amp; Consignment</th>
      <th>Delicatessen</th>
      <th>Paintball</th>
      <th>Austrian</th>
      <th>Hair Salons</th>
      <th>Bingo Halls</th>
      <th>Vocational &amp; Technical School</th>
      <th>Clowns</th>
      <th>Print Media</th>
      <th>Singaporean</th>
      <th>Conveyor Belt Sushi</th>
      <th>Used Car Dealers</th>
      <th>Weight Loss Centers</th>
      <th>Ice Delivery</th>
      <th>Noodles</th>
      <th>Pet Boarding</th>
      <th>Home &amp; Garden</th>
      <th>Photo Booth Rentals</th>
      <th>Personal Injury Law</th>
      <th>DJs</th>
      <th>Bikes</th>
      <th>Appliances</th>
      <th>Animal Shelters</th>
      <th>Visitor Centers</th>
      <th>Siding</th>
      <th>Home Decor</th>
      <th>Vegan</th>
      <th>Lounges</th>
      <th>Convenience Stores</th>
      <th>Internet Cafes</th>
      <th>Flea Markets</th>
      <th>Preschools</th>
      <th>Soba</th>
      <th>Opera &amp; Ballet</th>
      <th>Makeup Artists</th>
      <th>Educational Services</th>
      <th>Dermatologists</th>
      <th>Salad</th>
      <th>Gelato</th>
      <th>Massage</th>
      <th>Armenian</th>
      <th>Honduran</th>
      <th>Soul Food</th>
      <th>Beer Bar</th>
      <th>Windshield Installation &amp; Repair</th>
      <th>Soccer</th>
      <th>Boating</th>
      <th>Sports Medicine</th>
      <th>Heating &amp; Air Conditioning/HVAC</th>
      <th>Product Design</th>
      <th>Leisure Centers</th>
      <th>Pawn Shops</th>
      <th>Immigration Law</th>
      <th>Hot Pot</th>
      <th>Boat Charters</th>
      <th>Passport &amp; Visa Services</th>
      <th>Handyman</th>
      <th>Bookkeepers</th>
      <th>Cooking Classes</th>
      <th>Custom Cakes</th>
      <th>Books</th>
      <th>Mediterranean</th>
      <th>Windows Installation</th>
      <th>Arabian</th>
      <th>Vitamins &amp; Supplements</th>
      <th>Motorcycle Repair</th>
      <th>Rolfing</th>
      <th>Tires</th>
      <th>Kosher</th>
      <th>Kids Activities</th>
      <th>Paint &amp; Sip</th>
      <th>Guamanian</th>
      <th>Do-It-Yourself Food</th>
      <th>Hotels &amp; Travel</th>
      <th>Pet Adoption</th>
      <th>Spin Classes</th>
      <th>Sri Lankan</th>
      <th>Psychologists</th>
      <th>Whiskey Bars</th>
      <th>Dentists</th>
      <th>Fish &amp; Chips</th>
      <th>Taxis</th>
      <th>Mexican</th>
      <th>Ethical Grocery</th>
      <th>Service Stations</th>
      <th>Pediatric Dentists</th>
      <th>Fire Protection Services</th>
      <th>Walking Tours</th>
      <th>Florists</th>
      <th>Hakka</th>
      <th>Botanical Gardens</th>
      <th>Occupational Therapy</th>
      <th>Keys &amp; Locksmiths</th>
      <th>Tonkatsu</th>
      <th>Beer Hall</th>
      <th>Water Heater Installation/Repair</th>
      <th>Boat Repair</th>
      <th>Home Services</th>
      <th>Kebab</th>
      <th>Active Life</th>
      <th>Outdoor Furniture Stores</th>
      <th>Nurseries &amp; Gardening</th>
      <th>Specialty Schools</th>
      <th>Cabinetry</th>
      <th>Wine Tasting Classes</th>
      <th>Minho</th>
      <th>Special Education</th>
      <th>Carpet Installation</th>
      <th>Awnings</th>
      <th>Day Camps</th>
      <th>Beverage Store</th>
      <th>Wine Tasting Room</th>
      <th>Martial Arts</th>
      <th>Health &amp; Medical</th>
      <th>Supper Clubs</th>
      <th>Tapas/Small Plates</th>
      <th>Greek</th>
      <th>Archery</th>
      <th>Childbirth Education</th>
      <th>Unofficial Yelp Events</th>
      <th>Private Tutors</th>
      <th>Jazz &amp; Blues</th>
      <th>Livestock Feed &amp; Supply</th>
      <th>Pub Food</th>
      <th>Fishing</th>
      <th>Hotel bar</th>
      <th>Ethnic Grocery</th>
      <th>Pool &amp; Hot Tub Service</th>
      <th>Buffets</th>
      <th>Cantonese</th>
      <th>Gyms</th>
      <th>Naturopathic/Holistic</th>
      <th>Pet Photography</th>
      <th>Advertising</th>
      <th>Parks</th>
      <th>Employment Agencies</th>
      <th>Live/Raw Food</th>
      <th>Nightlife</th>
      <th>Pop-up Shops</th>
      <th>Street Art</th>
      <th>IT Services &amp; Computer Repair</th>
      <th>Insurance</th>
      <th>Arts &amp; Entertainment</th>
      <th>Flowers</th>
      <th>Colonics</th>
      <th>Yoga</th>
      <th>Dance Studios</th>
      <th>Veterinarians</th>
      <th>Flowers &amp; Gifts</th>
      <th>Pedicabs</th>
      <th>Piercing</th>
      <th>Travel Services</th>
      <th>Painters</th>
      <th>Golf Equipment Shops</th>
      <th>Desserts</th>
      <th>Diners</th>
      <th>Cooking Schools</th>
      <th>Herbal Shops</th>
      <th>Brewpubs</th>
      <th>Local Flavor</th>
      <th>Sports Bars</th>
      <th>Portuguese</th>
      <th>Lebanese</th>
      <th>Zoos</th>
      <th>Barbeque</th>
      <th>Studio Taping</th>
      <th>Herbs &amp; Spices</th>
      <th>Rock Climbing</th>
      <th>Italian</th>
      <th>Sporting Goods</th>
      <th>Specialty Food</th>
      <th>Auto Detailing</th>
      <th>Boat Tours</th>
      <th>Food Tours</th>
      <th>Divorce &amp; Family Law</th>
      <th>Smokehouse</th>
      <th>Bangladeshi</th>
      <th>Eyelash Service</th>
      <th>Towing</th>
      <th>Modern European</th>
      <th>Buddhist Temples</th>
      <th>Accessories</th>
      <th>Bike Rentals</th>
      <th>Dance Clubs</th>
      <th>Couriers &amp; Delivery Services</th>
      <th>Go Karts</th>
      <th>Surf Schools</th>
      <th>Party Supplies</th>
      <th>Event Planning &amp; Services</th>
      <th>Outlet Stores</th>
      <th>Sewing &amp; Alterations</th>
      <th>Farmers Market</th>
      <th>Office Equipment</th>
      <th>Dominican</th>
      <th>Climbing</th>
      <th>Cafes</th>
      <th>Pumpkin Patches</th>
      <th>Tiki Bars</th>
      <th>Used</th>
      <th>Bartenders</th>
      <th>Wraps</th>
      <th>Motorcycle Dealers</th>
      <th>Mobile Phone Accessories</th>
      <th>Flooring</th>
      <th>Internet Service Providers</th>
      <th>Grilling Equipment</th>
      <th>Hair Removal</th>
      <th>Physical Therapy</th>
      <th>Home Inspectors</th>
      <th>Balloon Services</th>
      <th>Amateur Sports Teams</th>
      <th>Airport Lounges</th>
      <th>Automotive</th>
      <th>Blow Dry/Out Services</th>
      <th>Korean</th>
      <th>Cheesesteaks</th>
      <th>Cosmetics &amp; Beauty Supply</th>
      <th>Moroccan</th>
      <th>Mailbox Centers</th>
      <th>Beach Bars</th>
      <th>Auto Upholstery</th>
      <th>Food Stands</th>
      <th>Party Characters</th>
      <th>Fitness &amp; Instruction</th>
      <th>Fitness/Exercise Equipment</th>
      <th>Gay Bars</th>
      <th>Placenta Encapsulations</th>
      <th>Utilities</th>
      <th>Juice Bars &amp; Smoothies</th>
      <th>Movers</th>
      <th>Property Management</th>
      <th>Sailing</th>
      <th>Plumbing</th>
      <th>Taiwanese</th>
      <th>Childrens Clothing</th>
      <th>Indoor Playcentre</th>
      <th>Animal Assisted Therapy</th>
      <th>Public Relations</th>
      <th>South African</th>
      <th>Water Delivery</th>
      <th>Dog Walkers</th>
      <th>Trinidadian</th>
      <th>Laser Tag</th>
      <th>Child Care &amp; Day Care</th>
      <th>Pet Groomers</th>
      <th>Health Markets</th>
      <th>Currency Exchange</th>
      <th>Tax Law</th>
      <th>Public Transportation</th>
      <th>Pool &amp; Billiards</th>
      <th>Street Vendors</th>
      <th>Food Trucks</th>
      <th>Business Consulting</th>
      <th>Religious Organizations</th>
      <th>Chicken Wings</th>
      <th>Baby Gear &amp; Furniture</th>
      <th>Wedding Chapels</th>
      <th>Security Systems</th>
      <th>Bubble Tea</th>
      <th>Mens Clothing</th>
      <th>Reiki</th>
      <th>Seafood</th>
      <th>Scottish</th>
      <th>Pet Sitting</th>
      <th>Izakaya</th>
      <th>Pretzels</th>
      <th>Customized Merchandise</th>
      <th>Newspapers &amp; Magazines</th>
      <th>Auto Insurance</th>
      <th>Real Estate</th>
      <th>Threading Services</th>
      <th>Mortgage Brokers</th>
      <th>Roofing</th>
      <th>Wills</th>
      <th>Hats</th>
      <th>Lighting Fixtures &amp; Equipment</th>
      <th>Radio Stations</th>
      <th>Arts &amp; Crafts</th>
      <th>Public Markets</th>
      <th>Health Retreats</th>
      <th>Bike tours</th>
      <th>Cocktail Bars</th>
      <th>Hungarian</th>
      <th>Building Supplies</th>
      <th>Churros</th>
      <th>Local Services</th>
      <th>Diagnostic Services</th>
      <th>Parenting Classes</th>
      <th>Auto Repair</th>
      <th>Laundromat</th>
      <th>Shared Office Spaces</th>
      <th>Historical Tours</th>
      <th>Kitchen Incubators</th>
      <th>Spray Tanning</th>
      <th>Imported Food</th>
      <th>Himalayan/Nepalese</th>
      <th>Drugstores</th>
      <th>Comic Books</th>
      <th>Optometrists</th>
      <th>Champagne Bars</th>
      <th>Gun/Rifle Ranges</th>
      <th>Professional Sports Teams</th>
      <th>Music Venues</th>
      <th>Karaoke</th>
      <th>Synagogues</th>
      <th>Seafood Markets</th>
      <th>Screen Printing</th>
      <th>Virtual Reality Centers</th>
      <th>Golf Cart Dealers</th>
      <th>Museums</th>
      <th>Hainan</th>
      <th>Medical Cannabis Referrals</th>
      <th>Bulgarian</th>
      <th>Commercial Real Estate</th>
      <th>Escape Games</th>
      <th>Japanese Curry</th>
      <th>Water Purification Services</th>
      <th>Food Delivery Services</th>
      <th>Malaysian</th>
      <th>Cinema</th>
      <th>Car Wash</th>
      <th>Swimming Pools</th>
      <th>Car Dealers</th>
      <th>Team Building Activities</th>
      <th>Musical Instruments &amp; Teachers</th>
      <th>Education</th>
      <th>Home Health Care</th>
      <th>Swimwear</th>
      <th>Nutritionists</th>
      <th>Recreation Centers</th>
      <th>Interval Training Gyms</th>
      <th>Pediatricians</th>
      <th>Pool Halls</th>
      <th>Reunion</th>
      <th>Mattresses</th>
      <th>Country Dance Halls</th>
      <th>Soup</th>
      <th>Piano Bars</th>
      <th>Scavenger Hunts</th>
      <th>Festivals</th>
      <th>Self Storage</th>
      <th>Engraving</th>
      <th>Marketing</th>
      <th>Meditation Centers</th>
      <th>Indonesian</th>
      <th>Landmarks &amp; Historical Buildings</th>
      <th>Outdoor Movies</th>
      <th>Departments of Motor Vehicles</th>
      <th>Tempura</th>
      <th>Damage Restoration</th>
      <th>Popcorn Shops</th>
      <th>Shaved Ice</th>
      <th>Furniture Repair</th>
      <th>Latin American</th>
      <th>Sports Wear</th>
      <th>Magicians</th>
      <th>Coffee &amp; Tea Supplies</th>
      <th>Food Banks</th>
      <th>Teppanyaki</th>
      <th>Chicken Shop</th>
      <th>Orthodontists</th>
      <th>Personal Care Services</th>
      <th>Mini Golf</th>
      <th>Wigs</th>
      <th>Themed Cafes</th>
      <th>Bagels</th>
      <th>Dry Cleaning &amp; Laundry</th>
      <th>Swiss Food</th>
      <th>Beer Gardens</th>
      <th>Electronics Repair</th>
      <th>Tea Rooms</th>
      <th>Nail Salons</th>
      <th>Beer Garden</th>
      <th>Pet Training</th>
      <th>Milkshake Bars</th>
      <th>Adult Entertainment</th>
      <th>Waxing</th>
      <th>Poke</th>
      <th>Acupuncture</th>
      <th>Farms</th>
      <th>Souvenir Shops</th>
      <th>Horseback Riding</th>
      <th>Personal Assistants</th>
      <th>Party Bike Rentals</th>
      <th>Cycling Classes</th>
      <th>Accountants</th>
      <th>Fondue</th>
      <th>Airports</th>
      <th>Mags</th>
      <th>Photography Stores &amp; Services</th>
      <th>Bookstores</th>
      <th>Irish</th>
      <th>Czech/Slovakian</th>
      <th>CSA</th>
      <th>Police Departments</th>
      <th>Allergists</th>
      <th>Department Stores</th>
      <th>Train Stations</th>
      <th>Pizza</th>
      <th>Bowling</th>
      <th>Music &amp; Video</th>
      <th>Trainers</th>
      <th>Golf Equipment</th>
      <th>Rest Stops</th>
      <th>Bocce Ball</th>
      <th>Cafeteria</th>
      <th>ATV Rentals/Tours</th>
      <th>Float Spa</th>
      <th>Gluten-Free</th>
      <th>Sandwiches</th>
      <th>Banks &amp; Credit Unions</th>
      <th>Coffee Roasteries</th>
      <th>Middle Schools &amp; High Schools</th>
      <th>Hot Dogs</th>
      <th>Bridal</th>
      <th>Breweries</th>
      <th>Observatories</th>
      <th>Pasta Shops</th>
      <th>Cheese Tasting Classes</th>
      <th>Horse Racing</th>
      <th>Golf Lessons</th>
      <th>Performing Arts</th>
      <th>Organic Stores</th>
      <th>Cambodian</th>
      <th>Toy Stores</th>
      <th>Barre Classes</th>
      <th>Air Duct Cleaning</th>
      <th>Counseling &amp; Mental Health</th>
      <th>Cigar Bars</th>
      <th>Fabric Stores</th>
      <th>Wine &amp; Spirits</th>
      <th>Estate Planning Law</th>
      <th>Paint-Your-Own Pottery</th>
      <th>Kitchen Supplies</th>
      <th>Caribbean</th>
      <th>Landscape Architects</th>
      <th>Medical Centers</th>
      <th>Tacos</th>
      <th>Brazilian Jiu-jitsu</th>
      <th>Art Schools</th>
      <th>Chinese</th>
      <th>Uzbek</th>
      <th>Cajun/Creole</th>
      <th>Waffles</th>
      <th>Tennis</th>
      <th>French</th>
      <th>Boot Camps</th>
      <th>Data Recovery</th>
      <th>Tai Chi</th>
      <th>Gardeners</th>
      <th>Cosmetic Surgeons</th>
      <th>Office Cleaning</th>
      <th>Aquariums</th>
      <th>Party Bus Rentals</th>
      <th>Laser Hair Removal</th>
      <th>Cultural Center</th>
      <th>Cideries</th>
      <th>Tuscan</th>
      <th>Ramen</th>
      <th>Steakhouses</th>
      <th>Polish</th>
      <th>Guest Houses</th>
      <th>Club Crawl</th>
      <th>Speakeasies</th>
      <th>Traditional Clothing</th>
      <th>Distilleries</th>
      <th>Session Photography</th>
      <th>Security Services</th>
      <th>Beaches</th>
      <th>Haitian</th>
      <th>Music &amp; DVDs</th>
      <th>Community Centers</th>
      <th>Dim Sum</th>
      <th>Saunas</th>
      <th>Door Sales/Installation</th>
      <th>Window Washing</th>
      <th>Apartments</th>
      <th>New Mexican Cuisine</th>
      <th>Comfort Food</th>
      <th>Spanish</th>
      <th>Investing</th>
      <th>Trampoline Parks</th>
      <th>Burgers</th>
      <th>Lakes</th>
      <th>Pest Control</th>
      <th>Hardware Stores</th>
      <th>Leather Goods</th>
      <th>Software Development</th>
      <th>Fashion</th>
      <th>Acne Treatment</th>
      <th>Gutter Services</th>
      <th>Bakeries</th>
      <th>Elementary Schools</th>
      <th>Lawyers</th>
      <th>Knife Sharpening</th>
      <th>Ski Schools</th>
      <th>Interior Design</th>
      <th>Hostels</th>
      <th>Tree Services</th>
      <th>Funeral Services &amp; Cemeteries</th>
      <th>Vacation Rentals</th>
      <th>Shipping Centers</th>
      <th>Vegetarian</th>
      <th>Outdoor Gear</th>
      <th>Life Coach</th>
      <th>Bartending Schools</th>
      <th>American (New)</th>
      <th>Christmas Trees</th>
      <th>Japanese</th>
      <th>Dinner Theater</th>
      <th>Car Window Tinting</th>
      <th>Candle Stores</th>
      <th>Hunting &amp; Fishing Supplies</th>
      <th>Brasseries</th>
      <th>Chocolatiers &amp; Shops</th>
      <th>Tapas Bars</th>
      <th>Cabaret</th>
      <th>Fur Clothing</th>
      <th>Mosques</th>
      <th>Guns &amp; Ammo</th>
      <th>Venues &amp; Event Spaces</th>
      <th>Discount Store</th>
      <th>Calabrian</th>
      <th>Wholesale Stores</th>
      <th>Art Galleries</th>
      <th>Community Gardens</th>
      <th>Drive-Thru Bars</th>
      <th>Aquarium Services</th>
      <th>Bankruptcy Law</th>
      <th>Registration Services</th>
      <th>Medical Transportation</th>
      <th>Szechuan</th>
      <th>Public Services &amp; Government</th>
      <th>Womens Clothing</th>
      <th>Creperies</th>
      <th>Kiosk</th>
      <th>Ski Resorts</th>
      <th>Sugar Shacks</th>
      <th>Community Service/Non-Profit</th>
      <th>Country Clubs</th>
      <th>Mobile Phones</th>
      <th>German</th>
      <th>Videos &amp; Video Game Rental</th>
      <th>Auto Customization</th>
      <th>Traditional Chinese Medicine</th>
      <th>Furniture Rental</th>
      <th>Acai Bowls</th>
      <th>Ukrainian</th>
      <th>Slovakian</th>
      <th>RV Repair</th>
      <th>Laboratory Testing</th>
      <th>Grocery</th>
      <th>Skating Rinks</th>
      <th>Water Stores</th>
      <th>Local Fish Stores</th>
      <th>Formal Wear</th>
      <th>Trophy Shops</th>
      <th>African</th>
      <th>Cannabis Collective</th>
      <th>Aircraft Repairs</th>
      <th>Kids Hair Salons</th>
      <th>Tableware</th>
      <th>Feng Shui</th>
      <th>Doctors</th>
      <th>Resorts</th>
      <th>Vinyl Records</th>
      <th>Bed &amp; Breakfast</th>
      <th>Reflexology</th>
      <th>Pita</th>
      <th>Bespoke Clothing</th>
      <th>Drywall Installation &amp; Repair</th>
      <th>Estate Liquidation</th>
      <th>Thrift Stores</th>
      <th>Cheese Shops</th>
      <th>Playgrounds</th>
      <th>Post Offices</th>
      <th>Audio/Visual Equipment Rental</th>
      <th>Casinos</th>
      <th>Party &amp; Event Planning</th>
      <th>Web Design</th>
      <th>Clothing Rental</th>
      <th>Gift Shops</th>
      <th>Truck Rental</th>
      <th>Canadian (New)</th>
      <th>Social Clubs</th>
      <th>International Grocery</th>
      <th>Nail Technicians</th>
      <th>Japanese Sweets</th>
      <th>Parking</th>
      <th>Colombian</th>
      <th>Hong Kong Style Cafe</th>
      <th>Asian Fusion</th>
      <th>Adult Education</th>
      <th>Payroll Services</th>
      <th>Marinas</th>
      <th>Translation Services</th>
      <th>Amusement Parks</th>
      <th>Pick Your Own Farms</th>
      <th>Cuban</th>
      <th>Food Court</th>
      <th>Basque</th>
      <th>Bike Repair/Maintenance</th>
      <th>LAN Centers</th>
      <th>Floral Designers</th>
      <th>Car Rental</th>
      <th>Art Museums</th>
      <th>Commercial Truck Repair</th>
      <th>Brewing Supplies</th>
      <th>Middle Eastern</th>
      <th>Yelp Events</th>
      <th>Junk Removal &amp; Hauling</th>
      <th>Attraction Farms</th>
      <th>Shoe Repair</th>
      <th>Delis</th>
      <th>&amp; Probates</th>
      <th>Squash</th>
      <th>Tex-Mex</th>
      <th>Coffee &amp; Tea</th>
      <th>Duty-Free Shops</th>
      <th>American (Traditional)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16000</th>
      <td>41597</td>
      <td>41687</td>
      <td>7k0PCx_t-emVk-sCzQ2cVQ</td>
      <td>Hero Certified Burgers</td>
      <td>Vaughan</td>
      <td>ON</td>
      <td>3.0</td>
      <td>4</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16001</th>
      <td>41599</td>
      <td>41689</td>
      <td>0XdKw7peQu6r_CC9L7ZyfQ</td>
      <td>NDG Hotdog</td>
      <td>Montr├йal</td>
      <td>QC</td>
      <td>4.0</td>
      <td>8</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16002</th>
      <td>41601</td>
      <td>41691</td>
      <td>VY6TmD3FE2rxtK5fNaoGcA</td>
      <td>Carolina Soda Shoppe</td>
      <td>Charlotte</td>
      <td>NC</td>
      <td>3.5</td>
      <td>15</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16003</th>
      <td>41604</td>
      <td>41694</td>
      <td>OAL8ewSpHm4A26aCNeKSuA</td>
      <td>Church Bella Variety</td>
      <td>Toronto</td>
      <td>ON</td>
      <td>3.0</td>
      <td>3</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>16004</th>
      <td>41605</td>
      <td>41695</td>
      <td>lcaGKzC5YEqjIFzuxlBq2A</td>
      <td>Dunkin' Donuts</td>
      <td>Charlotte</td>
      <td>NC</td>
      <td>2.5</td>
      <td>11</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
restaurants_df.shape
```




    (74587, 886)




```python
restaurants_df['state'].unique()
```




    array(['ON', 'NC', 'AZ', 'OH', 'NV', 'PA', 'AB', 'QC', 'WI', 'IL', 'NY',
           'SC', 'NM', 'VA', 'BAS', 'NE', 'XGM', 'CA', 'WA', 'XWY', 'CON',
           'TX', 'BC', 'VT', 'AL', 'AR', 'FL', 'XGL'], dtype=object)




```python
len(restaurants_df['state'].unique())
```




    28




```python
len(restaurants_df['city'].unique())
```




    844




```python

```


```python
pd.set_option('display.max_columns', 900)
```


```python
restaurants_df.head(2)
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
      <th>level_0</th>
      <th>index</th>
      <th>business_id</th>
      <th>name</th>
      <th>city</th>
      <th>state</th>
      <th>stars</th>
      <th>review_count</th>
      <th>is_restaurant</th>
      <th>Mass Media</th>
      <th>Hawaiian</th>
      <th>Libraries</th>
      <th>Speech Therapists</th>
      <th>Southern</th>
      <th>Pet Stores</th>
      <th>Masonry/Concrete</th>
      <th>Eastern European</th>
      <th>Eyebrow Services</th>
      <th>Firewood</th>
      <th>Sushi Bars</th>
      <th>Wineries</th>
      <th>Egyptian</th>
      <th>Island Pub</th>
      <th>Wine Bars</th>
      <th>Family Practice</th>
      <th>Sicilian</th>
      <th>Officiants</th>
      <th>Eatertainment</th>
      <th>Hookah Bars</th>
      <th>Computers</th>
      <th>Fast Food</th>
      <th>Rotisserie Chicken</th>
      <th>Dive Bars</th>
      <th>Teeth Whitening</th>
      <th>Trusts</th>
      <th>Golf Cart Rentals</th>
      <th>Holiday Decorations</th>
      <th>Airsoft</th>
      <th>Transmission Repair</th>
      <th>Furniture Reupholstery</th>
      <th>Personal Shopping</th>
      <th>Hospitals</th>
      <th>Boat Dealers</th>
      <th>Ethnic Food</th>
      <th>Medical Spas</th>
      <th>Oil Change Stations</th>
      <th>Video Game Stores</th>
      <th>Backshop</th>
      <th>Electronics</th>
      <th>Bar Crawl</th>
      <th>Limos</th>
      <th>Senegalese</th>
      <th>Airport Shuttles</th>
      <th>Gastropubs</th>
      <th>Supernatural Readings</th>
      <th>Spiritual Shop</th>
      <th>Eyewear &amp; Opticians</th>
      <th>Medical Supplies</th>
      <th>Iberian</th>
      <th>Plus Size Fashion</th>
      <th>Macarons</th>
      <th>Real Estate Services</th>
      <th>Tax Services</th>
      <th>Knitting Supplies</th>
      <th>Pan Asian</th>
      <th>Foundation Repair</th>
      <th>Electricians</th>
      <th>Burmese</th>
      <th>Pop-Up Restaurants</th>
      <th>Art Classes</th>
      <th>Udon</th>
      <th>Skydiving</th>
      <th>Luggage</th>
      <th>Czech</th>
      <th>Massage Therapy</th>
      <th>Nicaraguan</th>
      <th>Thai</th>
      <th>Caterers</th>
      <th>Wine Tours</th>
      <th>Check Cashing/Pay-day Loans</th>
      <th>Hobby Shops</th>
      <th>Mongolian</th>
      <th>Vape Shops</th>
      <th>Fruits &amp; Veggies</th>
      <th>Market Stalls</th>
      <th>Beer Tours</th>
      <th>Golf</th>
      <th>RV Parks</th>
      <th>Stadiums &amp; Arenas</th>
      <th>Chiropractors</th>
      <th>Shopping</th>
      <th>Patisserie/Cake Shop</th>
      <th>Hydrotherapy</th>
      <th>Printing Services</th>
      <th>Emergency Medicine</th>
      <th>Trailer Repair</th>
      <th>Tobacco Shops</th>
      <th>Airport Terminals</th>
      <th>Turkish</th>
      <th>Tanning Beds</th>
      <th>Summer Camps</th>
      <th>Churches</th>
      <th>Septic Services</th>
      <th>Musicians</th>
      <th>Colleges &amp; Universities</th>
      <th>Restaurant Supplies</th>
      <th>Personal Chefs</th>
      <th>Pilates</th>
      <th>Pubs</th>
      <th>Russian</th>
      <th>Tickets</th>
      <th>Architectural Tours</th>
      <th>Signmaking</th>
      <th>Laotian</th>
      <th>Kitchen &amp; Bath</th>
      <th>Donairs</th>
      <th>Strip Clubs</th>
      <th>Peruvian</th>
      <th>Hiking</th>
      <th>General Dentistry</th>
      <th>Donuts</th>
      <th>University Housing</th>
      <th>Dance Schools</th>
      <th>Dance Wear</th>
      <th>Fireplace Services</th>
      <th>Pressure Washers</th>
      <th>Campgrounds</th>
      <th>Falafel</th>
      <th>Fences &amp; Gates</th>
      <th>Roadside Assistance</th>
      <th>Pakistani</th>
      <th>Home Window Tinting</th>
      <th>Persian/Iranian</th>
      <th>Body Shops</th>
      <th>Cardiologists</th>
      <th>Antiques</th>
      <th>Indian</th>
      <th>Adult</th>
      <th>Haunted Houses</th>
      <th>Glass &amp; Mirrors</th>
      <th>Oaxacan</th>
      <th>Watches</th>
      <th>Irish Pub</th>
      <th>Pharmacy</th>
      <th>Shoe Stores</th>
      <th>Rehabilitation Center</th>
      <th>Day Spas</th>
      <th>Empanadas</th>
      <th>Real Estate Agents</th>
      <th>Laundry Services</th>
      <th>Afghan</th>
      <th>Psychics</th>
      <th>Water Suppliers</th>
      <th>Hot Tub &amp; Pool</th>
      <th>Cards &amp; Stationery</th>
      <th>Ticket Sales</th>
      <th>Venezuelan</th>
      <th>Graphic Design</th>
      <th>Halal</th>
      <th>Screen Printing/T-Shirt Printing</th>
      <th>Shanghainese</th>
      <th>Scandinavian</th>
      <th>Hotels</th>
      <th>Comedy Clubs</th>
      <th>Tasting Classes</th>
      <th>Gas Stations</th>
      <th>Car Buyers</th>
      <th>Sleep Specialists</th>
      <th>Tours</th>
      <th>Landscaping</th>
      <th>Propane</th>
      <th>Smog Check Stations</th>
      <th>Holistic Animal Care</th>
      <th>Bistros</th>
      <th>Arcades</th>
      <th>Ophthalmologists</th>
      <th>Health Coach</th>
      <th>Bars</th>
      <th>Meat Shops</th>
      <th>Wholesalers</th>
      <th>British</th>
      <th>Australian</th>
      <th>Notaries</th>
      <th>Professional Services</th>
      <th>Town Car Service</th>
      <th>Doulas</th>
      <th>Anesthesiologists</th>
      <th>Party Equipment Rentals</th>
      <th>Shopping Centers</th>
      <th>Belgian</th>
      <th>Home Cleaning</th>
      <th>Tattoo</th>
      <th>Mauritius</th>
      <th>Brazilian</th>
      <th>Midwives</th>
      <th>Photographers</th>
      <th>Ice Cream &amp; Frozen Yogurt</th>
      <th>Batting Cages</th>
      <th>Vehicle Wraps</th>
      <th>Courthouses</th>
      <th>Transportation</th>
      <th>Vietnamese</th>
      <th>Alternative Medicine</th>
      <th>Barbers</th>
      <th>Bus Tours</th>
      <th>Ethiopian</th>
      <th>Syrian</th>
      <th>Hair Stylists</th>
      <th>Metal Fabricators</th>
      <th>Coffeeshops</th>
      <th>Breakfast &amp; Brunch</th>
      <th>Kombucha</th>
      <th>Puerto Rican</th>
      <th>Recording &amp; Rehearsal Studios</th>
      <th>Furniture Stores</th>
      <th>Shaved Snow</th>
      <th>General Festivals</th>
      <th>Auto Parts &amp; Supplies</th>
      <th>Financial Services</th>
      <th>Cupcakes</th>
      <th>Pets</th>
      <th>Filipino</th>
      <th>Wedding Planning</th>
      <th>Cosmetic Dentists</th>
      <th>Argentine</th>
      <th>Dry Cleaning</th>
      <th>Race Tracks</th>
      <th>Jewelry</th>
      <th>Legal Services</th>
      <th>Tanning</th>
      <th>Poutineries</th>
      <th>Appliances &amp; Repair</th>
      <th>Cardio Classes</th>
      <th>Costumes</th>
      <th>Contractors</th>
      <th>Animal Physical Therapy</th>
      <th>Wheel &amp; Rim Repair</th>
      <th>Mobile Phone Repair</th>
      <th>Salvadoran</th>
      <th>Bounce House Rentals</th>
      <th>Event Photography</th>
      <th>Olive Oil</th>
      <th>Cannabis Clinics</th>
      <th>Auto Glass Services</th>
      <th>Head Shops</th>
      <th>Honey</th>
      <th>International</th>
      <th>Butcher</th>
      <th>Candy Stores</th>
      <th>Skin Care</th>
      <th>Art Supplies</th>
      <th>Hair Extensions</th>
      <th>Bird Shops</th>
      <th>Mountain Biking</th>
      <th>Game Meat</th>
      <th>Tabletop Games</th>
      <th>Sports Clubs</th>
      <th>Customs Brokers</th>
      <th>Used Bookstore</th>
      <th>Christmas Markets</th>
      <th>Car Share Services</th>
      <th>Pain Management</th>
      <th>Beauty &amp; Spas</th>
      <th>Pet Services</th>
      <th>Cannabis Dispensaries</th>
      <th>Beer</th>
      <th>Vintage &amp; Consignment</th>
      <th>Delicatessen</th>
      <th>Paintball</th>
      <th>Austrian</th>
      <th>Hair Salons</th>
      <th>Bingo Halls</th>
      <th>Vocational &amp; Technical School</th>
      <th>Clowns</th>
      <th>Print Media</th>
      <th>Singaporean</th>
      <th>Conveyor Belt Sushi</th>
      <th>Used Car Dealers</th>
      <th>Weight Loss Centers</th>
      <th>Ice Delivery</th>
      <th>Noodles</th>
      <th>Pet Boarding</th>
      <th>Home &amp; Garden</th>
      <th>Photo Booth Rentals</th>
      <th>Personal Injury Law</th>
      <th>DJs</th>
      <th>Bikes</th>
      <th>Appliances</th>
      <th>Animal Shelters</th>
      <th>Visitor Centers</th>
      <th>Siding</th>
      <th>Home Decor</th>
      <th>Vegan</th>
      <th>Lounges</th>
      <th>Convenience Stores</th>
      <th>Internet Cafes</th>
      <th>Flea Markets</th>
      <th>Preschools</th>
      <th>Soba</th>
      <th>Opera &amp; Ballet</th>
      <th>Makeup Artists</th>
      <th>Educational Services</th>
      <th>Dermatologists</th>
      <th>Salad</th>
      <th>Gelato</th>
      <th>Massage</th>
      <th>Armenian</th>
      <th>Honduran</th>
      <th>Soul Food</th>
      <th>Beer Bar</th>
      <th>Windshield Installation &amp; Repair</th>
      <th>Soccer</th>
      <th>Boating</th>
      <th>Sports Medicine</th>
      <th>Heating &amp; Air Conditioning/HVAC</th>
      <th>Product Design</th>
      <th>Leisure Centers</th>
      <th>Pawn Shops</th>
      <th>Immigration Law</th>
      <th>Hot Pot</th>
      <th>Boat Charters</th>
      <th>Passport &amp; Visa Services</th>
      <th>Handyman</th>
      <th>Bookkeepers</th>
      <th>Cooking Classes</th>
      <th>Custom Cakes</th>
      <th>Books</th>
      <th>Mediterranean</th>
      <th>Windows Installation</th>
      <th>Arabian</th>
      <th>Vitamins &amp; Supplements</th>
      <th>Motorcycle Repair</th>
      <th>Rolfing</th>
      <th>Tires</th>
      <th>Kosher</th>
      <th>Kids Activities</th>
      <th>Paint &amp; Sip</th>
      <th>Guamanian</th>
      <th>Do-It-Yourself Food</th>
      <th>Hotels &amp; Travel</th>
      <th>Pet Adoption</th>
      <th>Spin Classes</th>
      <th>Sri Lankan</th>
      <th>Psychologists</th>
      <th>Whiskey Bars</th>
      <th>Dentists</th>
      <th>Fish &amp; Chips</th>
      <th>Taxis</th>
      <th>Mexican</th>
      <th>Ethical Grocery</th>
      <th>Service Stations</th>
      <th>Pediatric Dentists</th>
      <th>Fire Protection Services</th>
      <th>Walking Tours</th>
      <th>Florists</th>
      <th>Hakka</th>
      <th>Botanical Gardens</th>
      <th>Occupational Therapy</th>
      <th>Keys &amp; Locksmiths</th>
      <th>Tonkatsu</th>
      <th>Beer Hall</th>
      <th>Water Heater Installation/Repair</th>
      <th>Boat Repair</th>
      <th>Home Services</th>
      <th>Kebab</th>
      <th>Active Life</th>
      <th>Outdoor Furniture Stores</th>
      <th>Nurseries &amp; Gardening</th>
      <th>Specialty Schools</th>
      <th>Cabinetry</th>
      <th>Wine Tasting Classes</th>
      <th>Minho</th>
      <th>Special Education</th>
      <th>Carpet Installation</th>
      <th>Awnings</th>
      <th>Day Camps</th>
      <th>Beverage Store</th>
      <th>Wine Tasting Room</th>
      <th>Martial Arts</th>
      <th>Health &amp; Medical</th>
      <th>Supper Clubs</th>
      <th>Tapas/Small Plates</th>
      <th>Greek</th>
      <th>Archery</th>
      <th>Childbirth Education</th>
      <th>Unofficial Yelp Events</th>
      <th>Private Tutors</th>
      <th>Jazz &amp; Blues</th>
      <th>Livestock Feed &amp; Supply</th>
      <th>Pub Food</th>
      <th>Fishing</th>
      <th>Hotel bar</th>
      <th>Ethnic Grocery</th>
      <th>Pool &amp; Hot Tub Service</th>
      <th>Buffets</th>
      <th>Cantonese</th>
      <th>Gyms</th>
      <th>Naturopathic/Holistic</th>
      <th>Pet Photography</th>
      <th>Advertising</th>
      <th>Parks</th>
      <th>Employment Agencies</th>
      <th>Live/Raw Food</th>
      <th>Nightlife</th>
      <th>Pop-up Shops</th>
      <th>Street Art</th>
      <th>IT Services &amp; Computer Repair</th>
      <th>Insurance</th>
      <th>Arts &amp; Entertainment</th>
      <th>Flowers</th>
      <th>Colonics</th>
      <th>Yoga</th>
      <th>Dance Studios</th>
      <th>Veterinarians</th>
      <th>Flowers &amp; Gifts</th>
      <th>Pedicabs</th>
      <th>Piercing</th>
      <th>Travel Services</th>
      <th>Painters</th>
      <th>Golf Equipment Shops</th>
      <th>Desserts</th>
      <th>Diners</th>
      <th>Cooking Schools</th>
      <th>Herbal Shops</th>
      <th>Brewpubs</th>
      <th>Local Flavor</th>
      <th>Sports Bars</th>
      <th>Portuguese</th>
      <th>Lebanese</th>
      <th>Zoos</th>
      <th>Barbeque</th>
      <th>Studio Taping</th>
      <th>Herbs &amp; Spices</th>
      <th>Rock Climbing</th>
      <th>Italian</th>
      <th>Sporting Goods</th>
      <th>Specialty Food</th>
      <th>Auto Detailing</th>
      <th>Boat Tours</th>
      <th>Food Tours</th>
      <th>Divorce &amp; Family Law</th>
      <th>Smokehouse</th>
      <th>Bangladeshi</th>
      <th>Eyelash Service</th>
      <th>Towing</th>
      <th>Modern European</th>
      <th>Buddhist Temples</th>
      <th>Accessories</th>
      <th>Bike Rentals</th>
      <th>Dance Clubs</th>
      <th>Couriers &amp; Delivery Services</th>
      <th>Go Karts</th>
      <th>Surf Schools</th>
      <th>Party Supplies</th>
      <th>Event Planning &amp; Services</th>
      <th>Outlet Stores</th>
      <th>Sewing &amp; Alterations</th>
      <th>Farmers Market</th>
      <th>Office Equipment</th>
      <th>Dominican</th>
      <th>Climbing</th>
      <th>Cafes</th>
      <th>Pumpkin Patches</th>
      <th>Tiki Bars</th>
      <th>Used</th>
      <th>Bartenders</th>
      <th>Wraps</th>
      <th>Motorcycle Dealers</th>
      <th>Mobile Phone Accessories</th>
      <th>Flooring</th>
      <th>Internet Service Providers</th>
      <th>Grilling Equipment</th>
      <th>Hair Removal</th>
      <th>Physical Therapy</th>
      <th>Home Inspectors</th>
      <th>Balloon Services</th>
      <th>Amateur Sports Teams</th>
      <th>Airport Lounges</th>
      <th>Automotive</th>
      <th>Blow Dry/Out Services</th>
      <th>Korean</th>
      <th>Cheesesteaks</th>
      <th>Cosmetics &amp; Beauty Supply</th>
      <th>Moroccan</th>
      <th>Mailbox Centers</th>
      <th>Beach Bars</th>
      <th>Auto Upholstery</th>
      <th>Food Stands</th>
      <th>Party Characters</th>
      <th>Fitness &amp; Instruction</th>
      <th>Fitness/Exercise Equipment</th>
      <th>Gay Bars</th>
      <th>Placenta Encapsulations</th>
      <th>Utilities</th>
      <th>Juice Bars &amp; Smoothies</th>
      <th>Movers</th>
      <th>Property Management</th>
      <th>Sailing</th>
      <th>Plumbing</th>
      <th>Taiwanese</th>
      <th>Childrens Clothing</th>
      <th>Indoor Playcentre</th>
      <th>Animal Assisted Therapy</th>
      <th>Public Relations</th>
      <th>South African</th>
      <th>Water Delivery</th>
      <th>Dog Walkers</th>
      <th>Trinidadian</th>
      <th>Laser Tag</th>
      <th>Child Care &amp; Day Care</th>
      <th>Pet Groomers</th>
      <th>Health Markets</th>
      <th>Currency Exchange</th>
      <th>Tax Law</th>
      <th>Public Transportation</th>
      <th>Pool &amp; Billiards</th>
      <th>Street Vendors</th>
      <th>Food Trucks</th>
      <th>Business Consulting</th>
      <th>Religious Organizations</th>
      <th>Chicken Wings</th>
      <th>Baby Gear &amp; Furniture</th>
      <th>Wedding Chapels</th>
      <th>Security Systems</th>
      <th>Bubble Tea</th>
      <th>Mens Clothing</th>
      <th>Reiki</th>
      <th>Seafood</th>
      <th>Scottish</th>
      <th>Pet Sitting</th>
      <th>Izakaya</th>
      <th>Pretzels</th>
      <th>Customized Merchandise</th>
      <th>Newspapers &amp; Magazines</th>
      <th>Auto Insurance</th>
      <th>Real Estate</th>
      <th>Threading Services</th>
      <th>Mortgage Brokers</th>
      <th>Roofing</th>
      <th>Wills</th>
      <th>Hats</th>
      <th>Lighting Fixtures &amp; Equipment</th>
      <th>Radio Stations</th>
      <th>Arts &amp; Crafts</th>
      <th>Public Markets</th>
      <th>Health Retreats</th>
      <th>Bike tours</th>
      <th>Cocktail Bars</th>
      <th>Hungarian</th>
      <th>Building Supplies</th>
      <th>Churros</th>
      <th>Local Services</th>
      <th>Diagnostic Services</th>
      <th>Parenting Classes</th>
      <th>Auto Repair</th>
      <th>Laundromat</th>
      <th>Shared Office Spaces</th>
      <th>Historical Tours</th>
      <th>Kitchen Incubators</th>
      <th>Spray Tanning</th>
      <th>Imported Food</th>
      <th>Himalayan/Nepalese</th>
      <th>Drugstores</th>
      <th>Comic Books</th>
      <th>Optometrists</th>
      <th>Champagne Bars</th>
      <th>Gun/Rifle Ranges</th>
      <th>Professional Sports Teams</th>
      <th>Music Venues</th>
      <th>Karaoke</th>
      <th>Synagogues</th>
      <th>Seafood Markets</th>
      <th>Screen Printing</th>
      <th>Virtual Reality Centers</th>
      <th>Golf Cart Dealers</th>
      <th>Museums</th>
      <th>Hainan</th>
      <th>Medical Cannabis Referrals</th>
      <th>Bulgarian</th>
      <th>Commercial Real Estate</th>
      <th>Escape Games</th>
      <th>Japanese Curry</th>
      <th>Water Purification Services</th>
      <th>Food Delivery Services</th>
      <th>Malaysian</th>
      <th>Cinema</th>
      <th>Car Wash</th>
      <th>Swimming Pools</th>
      <th>Car Dealers</th>
      <th>Team Building Activities</th>
      <th>Musical Instruments &amp; Teachers</th>
      <th>Education</th>
      <th>Home Health Care</th>
      <th>Swimwear</th>
      <th>Nutritionists</th>
      <th>Recreation Centers</th>
      <th>Interval Training Gyms</th>
      <th>Pediatricians</th>
      <th>Pool Halls</th>
      <th>Reunion</th>
      <th>Mattresses</th>
      <th>Country Dance Halls</th>
      <th>Soup</th>
      <th>Piano Bars</th>
      <th>Scavenger Hunts</th>
      <th>Festivals</th>
      <th>Self Storage</th>
      <th>Engraving</th>
      <th>Marketing</th>
      <th>Meditation Centers</th>
      <th>Indonesian</th>
      <th>Landmarks &amp; Historical Buildings</th>
      <th>Outdoor Movies</th>
      <th>Departments of Motor Vehicles</th>
      <th>Tempura</th>
      <th>Damage Restoration</th>
      <th>Popcorn Shops</th>
      <th>Shaved Ice</th>
      <th>Furniture Repair</th>
      <th>Latin American</th>
      <th>Sports Wear</th>
      <th>Magicians</th>
      <th>Coffee &amp; Tea Supplies</th>
      <th>Food Banks</th>
      <th>Teppanyaki</th>
      <th>Chicken Shop</th>
      <th>Orthodontists</th>
      <th>Personal Care Services</th>
      <th>Mini Golf</th>
      <th>Wigs</th>
      <th>Themed Cafes</th>
      <th>Bagels</th>
      <th>Dry Cleaning &amp; Laundry</th>
      <th>Swiss Food</th>
      <th>Beer Gardens</th>
      <th>Electronics Repair</th>
      <th>Tea Rooms</th>
      <th>Nail Salons</th>
      <th>Beer Garden</th>
      <th>Pet Training</th>
      <th>Milkshake Bars</th>
      <th>Adult Entertainment</th>
      <th>Waxing</th>
      <th>Poke</th>
      <th>Acupuncture</th>
      <th>Farms</th>
      <th>Souvenir Shops</th>
      <th>Horseback Riding</th>
      <th>Personal Assistants</th>
      <th>Party Bike Rentals</th>
      <th>Cycling Classes</th>
      <th>Accountants</th>
      <th>Fondue</th>
      <th>Airports</th>
      <th>Mags</th>
      <th>Photography Stores &amp; Services</th>
      <th>Bookstores</th>
      <th>Irish</th>
      <th>Czech/Slovakian</th>
      <th>CSA</th>
      <th>Police Departments</th>
      <th>Allergists</th>
      <th>Department Stores</th>
      <th>Train Stations</th>
      <th>Pizza</th>
      <th>Bowling</th>
      <th>Music &amp; Video</th>
      <th>Trainers</th>
      <th>Golf Equipment</th>
      <th>Rest Stops</th>
      <th>Bocce Ball</th>
      <th>Cafeteria</th>
      <th>ATV Rentals/Tours</th>
      <th>Float Spa</th>
      <th>Gluten-Free</th>
      <th>Sandwiches</th>
      <th>Banks &amp; Credit Unions</th>
      <th>Coffee Roasteries</th>
      <th>Middle Schools &amp; High Schools</th>
      <th>Hot Dogs</th>
      <th>Bridal</th>
      <th>Breweries</th>
      <th>Observatories</th>
      <th>Pasta Shops</th>
      <th>Cheese Tasting Classes</th>
      <th>Horse Racing</th>
      <th>Golf Lessons</th>
      <th>Performing Arts</th>
      <th>Organic Stores</th>
      <th>Cambodian</th>
      <th>Toy Stores</th>
      <th>Barre Classes</th>
      <th>Air Duct Cleaning</th>
      <th>Counseling &amp; Mental Health</th>
      <th>Cigar Bars</th>
      <th>Fabric Stores</th>
      <th>Wine &amp; Spirits</th>
      <th>Estate Planning Law</th>
      <th>Paint-Your-Own Pottery</th>
      <th>Kitchen Supplies</th>
      <th>Caribbean</th>
      <th>Landscape Architects</th>
      <th>Medical Centers</th>
      <th>Tacos</th>
      <th>Brazilian Jiu-jitsu</th>
      <th>Art Schools</th>
      <th>Chinese</th>
      <th>Uzbek</th>
      <th>Cajun/Creole</th>
      <th>Waffles</th>
      <th>Tennis</th>
      <th>French</th>
      <th>Boot Camps</th>
      <th>Data Recovery</th>
      <th>Tai Chi</th>
      <th>Gardeners</th>
      <th>Cosmetic Surgeons</th>
      <th>Office Cleaning</th>
      <th>Aquariums</th>
      <th>Party Bus Rentals</th>
      <th>Laser Hair Removal</th>
      <th>Cultural Center</th>
      <th>Cideries</th>
      <th>Tuscan</th>
      <th>Ramen</th>
      <th>Steakhouses</th>
      <th>Polish</th>
      <th>Guest Houses</th>
      <th>Club Crawl</th>
      <th>Speakeasies</th>
      <th>Traditional Clothing</th>
      <th>Distilleries</th>
      <th>Session Photography</th>
      <th>Security Services</th>
      <th>Beaches</th>
      <th>Haitian</th>
      <th>Music &amp; DVDs</th>
      <th>Community Centers</th>
      <th>Dim Sum</th>
      <th>Saunas</th>
      <th>Door Sales/Installation</th>
      <th>Window Washing</th>
      <th>Apartments</th>
      <th>New Mexican Cuisine</th>
      <th>Comfort Food</th>
      <th>Spanish</th>
      <th>Investing</th>
      <th>Trampoline Parks</th>
      <th>Burgers</th>
      <th>Lakes</th>
      <th>Pest Control</th>
      <th>Hardware Stores</th>
      <th>Leather Goods</th>
      <th>Software Development</th>
      <th>Fashion</th>
      <th>Acne Treatment</th>
      <th>Gutter Services</th>
      <th>Bakeries</th>
      <th>Elementary Schools</th>
      <th>Lawyers</th>
      <th>Knife Sharpening</th>
      <th>Ski Schools</th>
      <th>Interior Design</th>
      <th>Hostels</th>
      <th>Tree Services</th>
      <th>Funeral Services &amp; Cemeteries</th>
      <th>Vacation Rentals</th>
      <th>Shipping Centers</th>
      <th>Vegetarian</th>
      <th>Outdoor Gear</th>
      <th>Life Coach</th>
      <th>Bartending Schools</th>
      <th>American (New)</th>
      <th>Christmas Trees</th>
      <th>Japanese</th>
      <th>Dinner Theater</th>
      <th>Car Window Tinting</th>
      <th>Candle Stores</th>
      <th>Hunting &amp; Fishing Supplies</th>
      <th>Brasseries</th>
      <th>Chocolatiers &amp; Shops</th>
      <th>Tapas Bars</th>
      <th>Cabaret</th>
      <th>Fur Clothing</th>
      <th>Mosques</th>
      <th>Guns &amp; Ammo</th>
      <th>Venues &amp; Event Spaces</th>
      <th>Discount Store</th>
      <th>Calabrian</th>
      <th>Wholesale Stores</th>
      <th>Art Galleries</th>
      <th>Community Gardens</th>
      <th>Drive-Thru Bars</th>
      <th>Aquarium Services</th>
      <th>Bankruptcy Law</th>
      <th>Registration Services</th>
      <th>Medical Transportation</th>
      <th>Szechuan</th>
      <th>Public Services &amp; Government</th>
      <th>Womens Clothing</th>
      <th>Creperies</th>
      <th>Kiosk</th>
      <th>Ski Resorts</th>
      <th>Sugar Shacks</th>
      <th>Community Service/Non-Profit</th>
      <th>Country Clubs</th>
      <th>Mobile Phones</th>
      <th>German</th>
      <th>Videos &amp; Video Game Rental</th>
      <th>Auto Customization</th>
      <th>Traditional Chinese Medicine</th>
      <th>Furniture Rental</th>
      <th>Acai Bowls</th>
      <th>Ukrainian</th>
      <th>Slovakian</th>
      <th>RV Repair</th>
      <th>Laboratory Testing</th>
      <th>Grocery</th>
      <th>Skating Rinks</th>
      <th>Water Stores</th>
      <th>Local Fish Stores</th>
      <th>Formal Wear</th>
      <th>Trophy Shops</th>
      <th>African</th>
      <th>Cannabis Collective</th>
      <th>Aircraft Repairs</th>
      <th>Kids Hair Salons</th>
      <th>Tableware</th>
      <th>Feng Shui</th>
      <th>Doctors</th>
      <th>Resorts</th>
      <th>Vinyl Records</th>
      <th>Bed &amp; Breakfast</th>
      <th>Reflexology</th>
      <th>Pita</th>
      <th>Bespoke Clothing</th>
      <th>Drywall Installation &amp; Repair</th>
      <th>Estate Liquidation</th>
      <th>Thrift Stores</th>
      <th>Cheese Shops</th>
      <th>Playgrounds</th>
      <th>Post Offices</th>
      <th>Audio/Visual Equipment Rental</th>
      <th>Casinos</th>
      <th>Party &amp; Event Planning</th>
      <th>Web Design</th>
      <th>Clothing Rental</th>
      <th>Gift Shops</th>
      <th>Truck Rental</th>
      <th>Canadian (New)</th>
      <th>Social Clubs</th>
      <th>International Grocery</th>
      <th>Nail Technicians</th>
      <th>Japanese Sweets</th>
      <th>Parking</th>
      <th>Colombian</th>
      <th>Hong Kong Style Cafe</th>
      <th>Asian Fusion</th>
      <th>Adult Education</th>
      <th>Payroll Services</th>
      <th>Marinas</th>
      <th>Translation Services</th>
      <th>Amusement Parks</th>
      <th>Pick Your Own Farms</th>
      <th>Cuban</th>
      <th>Food Court</th>
      <th>Basque</th>
      <th>Bike Repair/Maintenance</th>
      <th>LAN Centers</th>
      <th>Floral Designers</th>
      <th>Car Rental</th>
      <th>Art Museums</th>
      <th>Commercial Truck Repair</th>
      <th>Brewing Supplies</th>
      <th>Middle Eastern</th>
      <th>Yelp Events</th>
      <th>Junk Removal &amp; Hauling</th>
      <th>Attraction Farms</th>
      <th>Shoe Repair</th>
      <th>Delis</th>
      <th>&amp; Probates</th>
      <th>Squash</th>
      <th>Tex-Mex</th>
      <th>Coffee &amp; Tea</th>
      <th>Duty-Free Shops</th>
      <th>American (Traditional)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>QXAEGFB4oINsVuTFxEYKFQ</td>
      <td>Emerald Chinese Restaurant</td>
      <td>Mississauga</td>
      <td>ON</td>
      <td>2.5</td>
      <td>128</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>2</td>
      <td>gnKjwL_1w79qoiV3IC_xQQ</td>
      <td>Musashi Japanese Restaurant</td>
      <td>Charlotte</td>
      <td>NC</td>
      <td>4.0</td>
      <td>170</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
restaurants_df.shape
```




    (74587, 786)




```python
restaurants_df['Active Life'].sum()
```




    447




```python
restaurants_df.iloc[:,8:].head()
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
      <th>is_restaurant</th>
      <th>Hawaiian</th>
      <th>Southern</th>
      <th>Eastern European</th>
      <th>Sushi Bars</th>
      <th>Wineries</th>
      <th>Egyptian</th>
      <th>Island Pub</th>
      <th>Wine Bars</th>
      <th>Family Practice</th>
      <th>Sicilian</th>
      <th>Officiants</th>
      <th>Eatertainment</th>
      <th>Hookah Bars</th>
      <th>Fast Food</th>
      <th>Rotisserie Chicken</th>
      <th>Dive Bars</th>
      <th>Trusts</th>
      <th>Golf Cart Rentals</th>
      <th>Hospitals</th>
      <th>Ethnic Food</th>
      <th>Bar Crawl</th>
      <th>Limos</th>
      <th>Senegalese</th>
      <th>Gastropubs</th>
      <th>Iberian</th>
      <th>Macarons</th>
      <th>Pan Asian</th>
      <th>Foundation Repair</th>
      <th>Electricians</th>
      <th>Burmese</th>
      <th>Pop-Up Restaurants</th>
      <th>Art Classes</th>
      <th>Udon</th>
      <th>Skydiving</th>
      <th>Luggage</th>
      <th>Czech</th>
      <th>Massage Therapy</th>
      <th>Nicaraguan</th>
      <th>Thai</th>
      <th>Caterers</th>
      <th>Wine Tours</th>
      <th>Check Cashing/Pay-day Loans</th>
      <th>Mongolian</th>
      <th>Fruits &amp; Veggies</th>
      <th>Market Stalls</th>
      <th>Beer Tours</th>
      <th>Golf</th>
      <th>RV Parks</th>
      <th>Stadiums &amp; Arenas</th>
      <th>Chiropractors</th>
      <th>Shopping</th>
      <th>Patisserie/Cake Shop</th>
      <th>Hydrotherapy</th>
      <th>Printing Services</th>
      <th>Emergency Medicine</th>
      <th>Trailer Repair</th>
      <th>Airport Terminals</th>
      <th>Turkish</th>
      <th>Tanning Beds</th>
      <th>Summer Camps</th>
      <th>Churches</th>
      <th>Septic Services</th>
      <th>Musicians</th>
      <th>Colleges &amp; Universities</th>
      <th>Restaurant Supplies</th>
      <th>Personal Chefs</th>
      <th>Pilates</th>
      <th>Pubs</th>
      <th>Russian</th>
      <th>Tickets</th>
      <th>Architectural Tours</th>
      <th>Signmaking</th>
      <th>Laotian</th>
      <th>Donairs</th>
      <th>Strip Clubs</th>
      <th>Peruvian</th>
      <th>Hiking</th>
      <th>General Dentistry</th>
      <th>Donuts</th>
      <th>University Housing</th>
      <th>Dance Wear</th>
      <th>Fireplace Services</th>
      <th>Pressure Washers</th>
      <th>Campgrounds</th>
      <th>Falafel</th>
      <th>Fences &amp; Gates</th>
      <th>Roadside Assistance</th>
      <th>Pakistani</th>
      <th>Home Window Tinting</th>
      <th>Persian/Iranian</th>
      <th>Cardiologists</th>
      <th>Antiques</th>
      <th>Indian</th>
      <th>Adult</th>
      <th>Haunted Houses</th>
      <th>Glass &amp; Mirrors</th>
      <th>Oaxacan</th>
      <th>Watches</th>
      <th>Irish Pub</th>
      <th>Shoe Stores</th>
      <th>Rehabilitation Center</th>
      <th>Day Spas</th>
      <th>Empanadas</th>
      <th>Laundry Services</th>
      <th>Afghan</th>
      <th>Psychics</th>
      <th>Water Suppliers</th>
      <th>Hot Tub &amp; Pool</th>
      <th>Cards &amp; Stationery</th>
      <th>Ticket Sales</th>
      <th>Venezuelan</th>
      <th>Graphic Design</th>
      <th>Halal</th>
      <th>Screen Printing/T-Shirt Printing</th>
      <th>Shanghainese</th>
      <th>Scandinavian</th>
      <th>Hotels</th>
      <th>Comedy Clubs</th>
      <th>Tasting Classes</th>
      <th>Car Buyers</th>
      <th>Sleep Specialists</th>
      <th>Tours</th>
      <th>Landscaping</th>
      <th>Propane</th>
      <th>Smog Check Stations</th>
      <th>Holistic Animal Care</th>
      <th>Bistros</th>
      <th>Arcades</th>
      <th>Ophthalmologists</th>
      <th>Health Coach</th>
      <th>Bars</th>
      <th>Meat Shops</th>
      <th>British</th>
      <th>Australian</th>
      <th>Notaries</th>
      <th>Professional Services</th>
      <th>Town Car Service</th>
      <th>Doulas</th>
      <th>Anesthesiologists</th>
      <th>Shopping Centers</th>
      <th>Belgian</th>
      <th>Home Cleaning</th>
      <th>Mauritius</th>
      <th>Brazilian</th>
      <th>Midwives</th>
      <th>Photographers</th>
      <th>Ice Cream &amp; Frozen Yogurt</th>
      <th>Batting Cages</th>
      <th>Vehicle Wraps</th>
      <th>Courthouses</th>
      <th>Transportation</th>
      <th>Vietnamese</th>
      <th>Alternative Medicine</th>
      <th>Barbers</th>
      <th>Ethiopian</th>
      <th>Syrian</th>
      <th>Hair Stylists</th>
      <th>Metal Fabricators</th>
      <th>Coffeeshops</th>
      <th>Breakfast &amp; Brunch</th>
      <th>Kombucha</th>
      <th>Puerto Rican</th>
      <th>Recording &amp; Rehearsal Studios</th>
      <th>Furniture Stores</th>
      <th>Shaved Snow</th>
      <th>General Festivals</th>
      <th>Auto Parts &amp; Supplies</th>
      <th>Financial Services</th>
      <th>Cupcakes</th>
      <th>Pets</th>
      <th>Filipino</th>
      <th>Cosmetic Dentists</th>
      <th>Argentine</th>
      <th>Dry Cleaning</th>
      <th>Race Tracks</th>
      <th>Legal Services</th>
      <th>Tanning</th>
      <th>Poutineries</th>
      <th>Cardio Classes</th>
      <th>Costumes</th>
      <th>Animal Physical Therapy</th>
      <th>Wheel &amp; Rim Repair</th>
      <th>Mobile Phone Repair</th>
      <th>Salvadoran</th>
      <th>Bounce House Rentals</th>
      <th>Event Photography</th>
      <th>Olive Oil</th>
      <th>Cannabis Clinics</th>
      <th>Honey</th>
      <th>International</th>
      <th>Butcher</th>
      <th>Candy Stores</th>
      <th>Skin Care</th>
      <th>Art Supplies</th>
      <th>Hair Extensions</th>
      <th>Bird Shops</th>
      <th>Mountain Biking</th>
      <th>Game Meat</th>
      <th>Tabletop Games</th>
      <th>Sports Clubs</th>
      <th>Customs Brokers</th>
      <th>Used Bookstore</th>
      <th>Christmas Markets</th>
      <th>Car Share Services</th>
      <th>Pain Management</th>
      <th>Cannabis Dispensaries</th>
      <th>Beer</th>
      <th>Vintage &amp; Consignment</th>
      <th>Delicatessen</th>
      <th>Paintball</th>
      <th>Austrian</th>
      <th>Hair Salons</th>
      <th>Bingo Halls</th>
      <th>Vocational &amp; Technical School</th>
      <th>Clowns</th>
      <th>Print Media</th>
      <th>Singaporean</th>
      <th>Conveyor Belt Sushi</th>
      <th>Used Car Dealers</th>
      <th>Ice Delivery</th>
      <th>Noodles</th>
      <th>Pet Boarding</th>
      <th>Photo Booth Rentals</th>
      <th>Personal Injury Law</th>
      <th>DJs</th>
      <th>Bikes</th>
      <th>Appliances</th>
      <th>Visitor Centers</th>
      <th>Siding</th>
      <th>Vegan</th>
      <th>Lounges</th>
      <th>Internet Cafes</th>
      <th>Flea Markets</th>
      <th>Preschools</th>
      <th>Soba</th>
      <th>Opera &amp; Ballet</th>
      <th>Makeup Artists</th>
      <th>Educational Services</th>
      <th>Dermatologists</th>
      <th>Salad</th>
      <th>Gelato</th>
      <th>Massage</th>
      <th>Armenian</th>
      <th>Honduran</th>
      <th>Soul Food</th>
      <th>Beer Bar</th>
      <th>Windshield Installation &amp; Repair</th>
      <th>Soccer</th>
      <th>Boating</th>
      <th>Sports Medicine</th>
      <th>Heating &amp; Air Conditioning/HVAC</th>
      <th>Product Design</th>
      <th>Leisure Centers</th>
      <th>Pawn Shops</th>
      <th>Immigration Law</th>
      <th>Hot Pot</th>
      <th>Boat Charters</th>
      <th>Passport &amp; Visa Services</th>
      <th>Handyman</th>
      <th>Bookkeepers</th>
      <th>Cooking Classes</th>
      <th>Custom Cakes</th>
      <th>Books</th>
      <th>Mediterranean</th>
      <th>Windows Installation</th>
      <th>Arabian</th>
      <th>Vitamins &amp; Supplements</th>
      <th>Motorcycle Repair</th>
      <th>Rolfing</th>
      <th>Tires</th>
      <th>Kosher</th>
      <th>Kids Activities</th>
      <th>Paint &amp; Sip</th>
      <th>Guamanian</th>
      <th>Do-It-Yourself Food</th>
      <th>Hotels &amp; Travel</th>
      <th>Spin Classes</th>
      <th>Sri Lankan</th>
      <th>Psychologists</th>
      <th>Whiskey Bars</th>
      <th>Dentists</th>
      <th>Fish &amp; Chips</th>
      <th>Taxis</th>
      <th>Mexican</th>
      <th>Ethical Grocery</th>
      <th>Service Stations</th>
      <th>Pediatric Dentists</th>
      <th>Fire Protection Services</th>
      <th>Florists</th>
      <th>Hakka</th>
      <th>Botanical Gardens</th>
      <th>Occupational Therapy</th>
      <th>Keys &amp; Locksmiths</th>
      <th>Tonkatsu</th>
      <th>Beer Hall</th>
      <th>Water Heater Installation/Repair</th>
      <th>Boat Repair</th>
      <th>Home Services</th>
      <th>Kebab</th>
      <th>Active Life</th>
      <th>Outdoor Furniture Stores</th>
      <th>Specialty Schools</th>
      <th>Cabinetry</th>
      <th>Wine Tasting Classes</th>
      <th>Minho</th>
      <th>Special Education</th>
      <th>Carpet Installation</th>
      <th>Awnings</th>
      <th>Day Camps</th>
      <th>Beverage Store</th>
      <th>Wine Tasting Room</th>
      <th>Martial Arts</th>
      <th>Supper Clubs</th>
      <th>Tapas/Small Plates</th>
      <th>Greek</th>
      <th>Archery</th>
      <th>Childbirth Education</th>
      <th>Unofficial Yelp Events</th>
      <th>Private Tutors</th>
      <th>Jazz &amp; Blues</th>
      <th>Livestock Feed &amp; Supply</th>
      <th>Pub Food</th>
      <th>Fishing</th>
      <th>Hotel bar</th>
      <th>Ethnic Grocery</th>
      <th>Pool &amp; Hot Tub Service</th>
      <th>Buffets</th>
      <th>Cantonese</th>
      <th>Naturopathic/Holistic</th>
      <th>Pet Photography</th>
      <th>Advertising</th>
      <th>Parks</th>
      <th>Employment Agencies</th>
      <th>Live/Raw Food</th>
      <th>Nightlife</th>
      <th>Pop-up Shops</th>
      <th>Street Art</th>
      <th>IT Services &amp; Computer Repair</th>
      <th>Insurance</th>
      <th>Arts &amp; Entertainment</th>
      <th>Flowers</th>
      <th>Colonics</th>
      <th>Dance Studios</th>
      <th>Veterinarians</th>
      <th>Flowers &amp; Gifts</th>
      <th>Pedicabs</th>
      <th>Piercing</th>
      <th>Travel Services</th>
      <th>Painters</th>
      <th>Golf Equipment Shops</th>
      <th>Desserts</th>
      <th>Diners</th>
      <th>Cooking Schools</th>
      <th>Herbal Shops</th>
      <th>Brewpubs</th>
      <th>Local Flavor</th>
      <th>Sports Bars</th>
      <th>Portuguese</th>
      <th>Lebanese</th>
      <th>Zoos</th>
      <th>Barbeque</th>
      <th>Studio Taping</th>
      <th>Herbs &amp; Spices</th>
      <th>Rock Climbing</th>
      <th>Italian</th>
      <th>Specialty Food</th>
      <th>Boat Tours</th>
      <th>Food Tours</th>
      <th>Divorce &amp; Family Law</th>
      <th>Smokehouse</th>
      <th>Bangladeshi</th>
      <th>Eyelash Service</th>
      <th>Towing</th>
      <th>Modern European</th>
      <th>Buddhist Temples</th>
      <th>Bike Rentals</th>
      <th>Dance Clubs</th>
      <th>Couriers &amp; Delivery Services</th>
      <th>Go Karts</th>
      <th>Surf Schools</th>
      <th>Party Supplies</th>
      <th>Outlet Stores</th>
      <th>Sewing &amp; Alterations</th>
      <th>Farmers Market</th>
      <th>Office Equipment</th>
      <th>Dominican</th>
      <th>Climbing</th>
      <th>Cafes</th>
      <th>Pumpkin Patches</th>
      <th>Tiki Bars</th>
      <th>Used</th>
      <th>Bartenders</th>
      <th>Wraps</th>
      <th>Motorcycle Dealers</th>
      <th>Mobile Phone Accessories</th>
      <th>Flooring</th>
      <th>Internet Service Providers</th>
      <th>Grilling Equipment</th>
      <th>Hair Removal</th>
      <th>Physical Therapy</th>
      <th>Balloon Services</th>
      <th>Amateur Sports Teams</th>
      <th>Airport Lounges</th>
      <th>Blow Dry/Out Services</th>
      <th>Korean</th>
      <th>Cheesesteaks</th>
      <th>Moroccan</th>
      <th>Mailbox Centers</th>
      <th>Beach Bars</th>
      <th>Auto Upholstery</th>
      <th>Food Stands</th>
      <th>Party Characters</th>
      <th>Fitness/Exercise Equipment</th>
      <th>Gay Bars</th>
      <th>Placenta Encapsulations</th>
      <th>Utilities</th>
      <th>Juice Bars &amp; Smoothies</th>
      <th>Movers</th>
      <th>Property Management</th>
      <th>Sailing</th>
      <th>Taiwanese</th>
      <th>Childrens Clothing</th>
      <th>Indoor Playcentre</th>
      <th>Animal Assisted Therapy</th>
      <th>Public Relations</th>
      <th>South African</th>
      <th>Water Delivery</th>
      <th>Trinidadian</th>
      <th>Laser Tag</th>
      <th>Child Care &amp; Day Care</th>
      <th>Health Markets</th>
      <th>Currency Exchange</th>
      <th>Tax Law</th>
      <th>Public Transportation</th>
      <th>Pool &amp; Billiards</th>
      <th>Street Vendors</th>
      <th>Food Trucks</th>
      <th>Religious Organizations</th>
      <th>Chicken Wings</th>
      <th>Baby Gear &amp; Furniture</th>
      <th>Wedding Chapels</th>
      <th>Security Systems</th>
      <th>Bubble Tea</th>
      <th>Mens Clothing</th>
      <th>Reiki</th>
      <th>Seafood</th>
      <th>Scottish</th>
      <th>Pet Sitting</th>
      <th>Izakaya</th>
      <th>Pretzels</th>
      <th>Newspapers &amp; Magazines</th>
      <th>Auto Insurance</th>
      <th>Real Estate</th>
      <th>Threading Services</th>
      <th>Mortgage Brokers</th>
      <th>Roofing</th>
      <th>Wills</th>
      <th>Hats</th>
      <th>Lighting Fixtures &amp; Equipment</th>
      <th>Radio Stations</th>
      <th>Arts &amp; Crafts</th>
      <th>Public Markets</th>
      <th>Health Retreats</th>
      <th>Bike tours</th>
      <th>Cocktail Bars</th>
      <th>Hungarian</th>
      <th>Building Supplies</th>
      <th>Churros</th>
      <th>Local Services</th>
      <th>Diagnostic Services</th>
      <th>Parenting Classes</th>
      <th>Laundromat</th>
      <th>Shared Office Spaces</th>
      <th>Historical Tours</th>
      <th>Kitchen Incubators</th>
      <th>Spray Tanning</th>
      <th>Imported Food</th>
      <th>Himalayan/Nepalese</th>
      <th>Comic Books</th>
      <th>Optometrists</th>
      <th>Champagne Bars</th>
      <th>Gun/Rifle Ranges</th>
      <th>Professional Sports Teams</th>
      <th>Music Venues</th>
      <th>Karaoke</th>
      <th>Synagogues</th>
      <th>Seafood Markets</th>
      <th>Screen Printing</th>
      <th>Virtual Reality Centers</th>
      <th>Golf Cart Dealers</th>
      <th>Museums</th>
      <th>Hainan</th>
      <th>Medical Cannabis Referrals</th>
      <th>Bulgarian</th>
      <th>Japanese Curry</th>
      <th>Water Purification Services</th>
      <th>Food Delivery Services</th>
      <th>Malaysian</th>
      <th>Cinema</th>
      <th>Swimming Pools</th>
      <th>Car Dealers</th>
      <th>Team Building Activities</th>
      <th>Musical Instruments &amp; Teachers</th>
      <th>Education</th>
      <th>Home Health Care</th>
      <th>Swimwear</th>
      <th>Recreation Centers</th>
      <th>Interval Training Gyms</th>
      <th>Pediatricians</th>
      <th>Pool Halls</th>
      <th>Reunion</th>
      <th>Mattresses</th>
      <th>Country Dance Halls</th>
      <th>Soup</th>
      <th>Piano Bars</th>
      <th>Festivals</th>
      <th>Self Storage</th>
      <th>Engraving</th>
      <th>Marketing</th>
      <th>Meditation Centers</th>
      <th>Indonesian</th>
      <th>Landmarks &amp; Historical Buildings</th>
      <th>Outdoor Movies</th>
      <th>Departments of Motor Vehicles</th>
      <th>Tempura</th>
      <th>Damage Restoration</th>
      <th>Popcorn Shops</th>
      <th>Shaved Ice</th>
      <th>Furniture Repair</th>
      <th>Latin American</th>
      <th>Sports Wear</th>
      <th>Magicians</th>
      <th>Coffee &amp; Tea Supplies</th>
      <th>Food Banks</th>
      <th>Teppanyaki</th>
      <th>Chicken Shop</th>
      <th>Orthodontists</th>
      <th>Personal Care Services</th>
      <th>Mini Golf</th>
      <th>Wigs</th>
      <th>Themed Cafes</th>
      <th>Bagels</th>
      <th>Swiss Food</th>
      <th>Beer Gardens</th>
      <th>Electronics Repair</th>
      <th>Tea Rooms</th>
      <th>Nail Salons</th>
      <th>Beer Garden</th>
      <th>Pet Training</th>
      <th>Milkshake Bars</th>
      <th>Adult Entertainment</th>
      <th>Waxing</th>
      <th>Poke</th>
      <th>Acupuncture</th>
      <th>Farms</th>
      <th>Horseback Riding</th>
      <th>Personal Assistants</th>
      <th>Party Bike Rentals</th>
      <th>Cycling Classes</th>
      <th>Accountants</th>
      <th>Fondue</th>
      <th>Airports</th>
      <th>Mags</th>
      <th>Bookstores</th>
      <th>Irish</th>
      <th>Czech/Slovakian</th>
      <th>CSA</th>
      <th>Police Departments</th>
      <th>Allergists</th>
      <th>Train Stations</th>
      <th>Pizza</th>
      <th>Bowling</th>
      <th>Golf Equipment</th>
      <th>Rest Stops</th>
      <th>Bocce Ball</th>
      <th>Cafeteria</th>
      <th>ATV Rentals/Tours</th>
      <th>Float Spa</th>
      <th>Sandwiches</th>
      <th>Banks &amp; Credit Unions</th>
      <th>Coffee Roasteries</th>
      <th>Middle Schools &amp; High Schools</th>
      <th>Hot Dogs</th>
      <th>Bridal</th>
      <th>Breweries</th>
      <th>Observatories</th>
      <th>Pasta Shops</th>
      <th>Cheese Tasting Classes</th>
      <th>Horse Racing</th>
      <th>Golf Lessons</th>
      <th>Organic Stores</th>
      <th>Cambodian</th>
      <th>Barre Classes</th>
      <th>Air Duct Cleaning</th>
      <th>Counseling &amp; Mental Health</th>
      <th>Cigar Bars</th>
      <th>Fabric Stores</th>
      <th>Wine &amp; Spirits</th>
      <th>Paint-Your-Own Pottery</th>
      <th>Kitchen Supplies</th>
      <th>Caribbean</th>
      <th>Landscape Architects</th>
      <th>Medical Centers</th>
      <th>Tacos</th>
      <th>Brazilian Jiu-jitsu</th>
      <th>Art Schools</th>
      <th>Chinese</th>
      <th>Uzbek</th>
      <th>Cajun/Creole</th>
      <th>Waffles</th>
      <th>Tennis</th>
      <th>French</th>
      <th>Boot Camps</th>
      <th>Data Recovery</th>
      <th>Tai Chi</th>
      <th>Gardeners</th>
      <th>Cosmetic Surgeons</th>
      <th>Office Cleaning</th>
      <th>Aquariums</th>
      <th>Party Bus Rentals</th>
      <th>Laser Hair Removal</th>
      <th>Cultural Center</th>
      <th>Cideries</th>
      <th>Tuscan</th>
      <th>Ramen</th>
      <th>Steakhouses</th>
      <th>Polish</th>
      <th>Guest Houses</th>
      <th>Club Crawl</th>
      <th>Speakeasies</th>
      <th>Traditional Clothing</th>
      <th>Distilleries</th>
      <th>Session Photography</th>
      <th>Security Services</th>
      <th>Beaches</th>
      <th>Haitian</th>
      <th>Music &amp; DVDs</th>
      <th>Community Centers</th>
      <th>Dim Sum</th>
      <th>Saunas</th>
      <th>Door Sales/Installation</th>
      <th>Window Washing</th>
      <th>Apartments</th>
      <th>New Mexican Cuisine</th>
      <th>Comfort Food</th>
      <th>Spanish</th>
      <th>Investing</th>
      <th>Trampoline Parks</th>
      <th>Burgers</th>
      <th>Lakes</th>
      <th>Pest Control</th>
      <th>Hardware Stores</th>
      <th>Leather Goods</th>
      <th>Software Development</th>
      <th>Acne Treatment</th>
      <th>Gutter Services</th>
      <th>Bakeries</th>
      <th>Elementary Schools</th>
      <th>Knife Sharpening</th>
      <th>Ski Schools</th>
      <th>Interior Design</th>
      <th>Hostels</th>
      <th>Tree Services</th>
      <th>Vacation Rentals</th>
      <th>Shipping Centers</th>
      <th>Vegetarian</th>
      <th>Outdoor Gear</th>
      <th>Life Coach</th>
      <th>Bartending Schools</th>
      <th>American (New)</th>
      <th>Christmas Trees</th>
      <th>Japanese</th>
      <th>Dinner Theater</th>
      <th>Candle Stores</th>
      <th>Hunting &amp; Fishing Supplies</th>
      <th>Brasseries</th>
      <th>Chocolatiers &amp; Shops</th>
      <th>Tapas Bars</th>
      <th>Cabaret</th>
      <th>Fur Clothing</th>
      <th>Mosques</th>
      <th>Guns &amp; Ammo</th>
      <th>Venues &amp; Event Spaces</th>
      <th>Calabrian</th>
      <th>Art Galleries</th>
      <th>Community Gardens</th>
      <th>Drive-Thru Bars</th>
      <th>Aquarium Services</th>
      <th>Registration Services</th>
      <th>Medical Transportation</th>
      <th>Szechuan</th>
      <th>Public Services &amp; Government</th>
      <th>Womens Clothing</th>
      <th>Creperies</th>
      <th>Kiosk</th>
      <th>Ski Resorts</th>
      <th>Sugar Shacks</th>
      <th>Community Service/Non-Profit</th>
      <th>Country Clubs</th>
      <th>German</th>
      <th>Videos &amp; Video Game Rental</th>
      <th>Auto Customization</th>
      <th>Traditional Chinese Medicine</th>
      <th>Furniture Rental</th>
      <th>Acai Bowls</th>
      <th>Ukrainian</th>
      <th>Slovakian</th>
      <th>RV Repair</th>
      <th>Laboratory Testing</th>
      <th>Skating Rinks</th>
      <th>Water Stores</th>
      <th>Local Fish Stores</th>
      <th>Formal Wear</th>
      <th>Trophy Shops</th>
      <th>African</th>
      <th>Cannabis Collective</th>
      <th>Aircraft Repairs</th>
      <th>Kids Hair Salons</th>
      <th>Tableware</th>
      <th>Feng Shui</th>
      <th>Doctors</th>
      <th>Resorts</th>
      <th>Vinyl Records</th>
      <th>Bed &amp; Breakfast</th>
      <th>Reflexology</th>
      <th>Pita</th>
      <th>Bespoke Clothing</th>
      <th>Drywall Installation &amp; Repair</th>
      <th>Estate Liquidation</th>
      <th>Thrift Stores</th>
      <th>Cheese Shops</th>
      <th>Playgrounds</th>
      <th>Post Offices</th>
      <th>Audio/Visual Equipment Rental</th>
      <th>Casinos</th>
      <th>Party &amp; Event Planning</th>
      <th>Web Design</th>
      <th>Clothing Rental</th>
      <th>Gift Shops</th>
      <th>Truck Rental</th>
      <th>Canadian (New)</th>
      <th>Social Clubs</th>
      <th>International Grocery</th>
      <th>Nail Technicians</th>
      <th>Japanese Sweets</th>
      <th>Parking</th>
      <th>Colombian</th>
      <th>Hong Kong Style Cafe</th>
      <th>Asian Fusion</th>
      <th>Adult Education</th>
      <th>Payroll Services</th>
      <th>Marinas</th>
      <th>Translation Services</th>
      <th>Amusement Parks</th>
      <th>Pick Your Own Farms</th>
      <th>Cuban</th>
      <th>Food Court</th>
      <th>Basque</th>
      <th>Bike Repair/Maintenance</th>
      <th>LAN Centers</th>
      <th>Floral Designers</th>
      <th>Car Rental</th>
      <th>Art Museums</th>
      <th>Commercial Truck Repair</th>
      <th>Brewing Supplies</th>
      <th>Middle Eastern</th>
      <th>Yelp Events</th>
      <th>Junk Removal &amp; Hauling</th>
      <th>Attraction Farms</th>
      <th>Shoe Repair</th>
      <th>Delis</th>
      <th>&amp; Probates</th>
      <th>Squash</th>
      <th>Tex-Mex</th>
      <th>Coffee &amp; Tea</th>
      <th>Duty-Free Shops</th>
      <th>American (Traditional)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
##indexing to the ctaegories and dropping categories that have less than 100 because of false categories grroupng like signmaking on some food and restaurant lablled ones


drop_uselss_cat= restaurants_df.iloc[:,8:].drop([col for col, val in restaurants_df.iloc[:,8:].sum().iteritems() if val < 100], axis=1)
```


```python
drop_uselss_cat.shape
```




    (74587, 162)




```python
#merge on business id only with extra categories

rest_cat_bus_id = pd.concat([restaurants_df['business_id'], drop_uselss_cat], axis=1, sort=False)
```


```python
rest_cat_bus_id.head()
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
      <th>business_id</th>
      <th>is_restaurant</th>
      <th>Hawaiian</th>
      <th>Southern</th>
      <th>Sushi Bars</th>
      <th>Wineries</th>
      <th>Wine Bars</th>
      <th>Hookah Bars</th>
      <th>Fast Food</th>
      <th>Dive Bars</th>
      <th>Ethnic Food</th>
      <th>Gastropubs</th>
      <th>Thai</th>
      <th>Caterers</th>
      <th>Fruits &amp; Veggies</th>
      <th>Golf</th>
      <th>Shopping</th>
      <th>Patisserie/Cake Shop</th>
      <th>Turkish</th>
      <th>Pubs</th>
      <th>Peruvian</th>
      <th>Donuts</th>
      <th>Falafel</th>
      <th>Pakistani</th>
      <th>Persian/Iranian</th>
      <th>Indian</th>
      <th>Afghan</th>
      <th>Halal</th>
      <th>Hotels</th>
      <th>Tours</th>
      <th>Bistros</th>
      <th>Arcades</th>
      <th>Bars</th>
      <th>Meat Shops</th>
      <th>British</th>
      <th>Professional Services</th>
      <th>Shopping Centers</th>
      <th>Ice Cream &amp; Frozen Yogurt</th>
      <th>Vietnamese</th>
      <th>Breakfast &amp; Brunch</th>
      <th>Cupcakes</th>
      <th>Filipino</th>
      <th>Poutineries</th>
      <th>International</th>
      <th>Butcher</th>
      <th>Candy Stores</th>
      <th>Beer</th>
      <th>Delicatessen</th>
      <th>Noodles</th>
      <th>Vegan</th>
      <th>Lounges</th>
      <th>Internet Cafes</th>
      <th>Salad</th>
      <th>Gelato</th>
      <th>Soul Food</th>
      <th>Beer Bar</th>
      <th>Hot Pot</th>
      <th>Custom Cakes</th>
      <th>Books</th>
      <th>Mediterranean</th>
      <th>Vitamins &amp; Supplements</th>
      <th>Kosher</th>
      <th>Do-It-Yourself Food</th>
      <th>Hotels &amp; Travel</th>
      <th>Fish &amp; Chips</th>
      <th>Mexican</th>
      <th>Florists</th>
      <th>Home Services</th>
      <th>Active Life</th>
      <th>Tapas/Small Plates</th>
      <th>Greek</th>
      <th>Buffets</th>
      <th>Cantonese</th>
      <th>Nightlife</th>
      <th>Arts &amp; Entertainment</th>
      <th>Flowers</th>
      <th>Flowers &amp; Gifts</th>
      <th>Desserts</th>
      <th>Diners</th>
      <th>Local Flavor</th>
      <th>Sports Bars</th>
      <th>Portuguese</th>
      <th>Lebanese</th>
      <th>Barbeque</th>
      <th>Herbs &amp; Spices</th>
      <th>Italian</th>
      <th>Specialty Food</th>
      <th>Modern European</th>
      <th>Dance Clubs</th>
      <th>Farmers Market</th>
      <th>Cafes</th>
      <th>Wraps</th>
      <th>Korean</th>
      <th>Cheesesteaks</th>
      <th>Food Stands</th>
      <th>Juice Bars &amp; Smoothies</th>
      <th>Taiwanese</th>
      <th>Health Markets</th>
      <th>Street Vendors</th>
      <th>Food Trucks</th>
      <th>Chicken Wings</th>
      <th>Bubble Tea</th>
      <th>Seafood</th>
      <th>Arts &amp; Crafts</th>
      <th>Cocktail Bars</th>
      <th>Local Services</th>
      <th>Imported Food</th>
      <th>Music Venues</th>
      <th>Karaoke</th>
      <th>Seafood Markets</th>
      <th>Food Delivery Services</th>
      <th>Education</th>
      <th>Soup</th>
      <th>Festivals</th>
      <th>Shaved Ice</th>
      <th>Latin American</th>
      <th>Chicken Shop</th>
      <th>Bagels</th>
      <th>Tea Rooms</th>
      <th>Poke</th>
      <th>Mags</th>
      <th>Irish</th>
      <th>Pizza</th>
      <th>Sandwiches</th>
      <th>Coffee Roasteries</th>
      <th>Hot Dogs</th>
      <th>Breweries</th>
      <th>Organic Stores</th>
      <th>Wine &amp; Spirits</th>
      <th>Caribbean</th>
      <th>Tacos</th>
      <th>Chinese</th>
      <th>Cajun/Creole</th>
      <th>Waffles</th>
      <th>French</th>
      <th>Ramen</th>
      <th>Steakhouses</th>
      <th>Dim Sum</th>
      <th>Comfort Food</th>
      <th>Spanish</th>
      <th>Burgers</th>
      <th>Bakeries</th>
      <th>Vegetarian</th>
      <th>Japanese</th>
      <th>Brasseries</th>
      <th>Chocolatiers &amp; Shops</th>
      <th>Tapas Bars</th>
      <th>Venues &amp; Event Spaces</th>
      <th>Creperies</th>
      <th>German</th>
      <th>Acai Bowls</th>
      <th>African</th>
      <th>Cheese Shops</th>
      <th>Casinos</th>
      <th>Party &amp; Event Planning</th>
      <th>Gift Shops</th>
      <th>International Grocery</th>
      <th>Asian Fusion</th>
      <th>Food Court</th>
      <th>Middle Eastern</th>
      <th>Delis</th>
      <th>Tex-Mex</th>
      <th>Coffee &amp; Tea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>QXAEGFB4oINsVuTFxEYKFQ</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gnKjwL_1w79qoiV3IC_xQQ</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1Dfx3zM-rW4n-31KeC8sJg</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fweCYi8FmbJXHCqLnwuk8w</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-K4gAv8_vjx8-2BxkVeRkA</td>
      <td>1</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
rest_cat_bus_id.drop(['is_restaurant'], axis=1, inplace=True)
```


```python
rest_cat_bus_id.head(2)
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
      <th>business_id</th>
      <th>Hawaiian</th>
      <th>Southern</th>
      <th>Sushi Bars</th>
      <th>Wineries</th>
      <th>Wine Bars</th>
      <th>Hookah Bars</th>
      <th>Fast Food</th>
      <th>Dive Bars</th>
      <th>Ethnic Food</th>
      <th>Gastropubs</th>
      <th>Thai</th>
      <th>Caterers</th>
      <th>Fruits &amp; Veggies</th>
      <th>Golf</th>
      <th>Shopping</th>
      <th>Patisserie/Cake Shop</th>
      <th>Turkish</th>
      <th>Pubs</th>
      <th>Peruvian</th>
      <th>Donuts</th>
      <th>Falafel</th>
      <th>Pakistani</th>
      <th>Persian/Iranian</th>
      <th>Indian</th>
      <th>Afghan</th>
      <th>Halal</th>
      <th>Hotels</th>
      <th>Tours</th>
      <th>Bistros</th>
      <th>Arcades</th>
      <th>Bars</th>
      <th>Meat Shops</th>
      <th>British</th>
      <th>Professional Services</th>
      <th>Shopping Centers</th>
      <th>Ice Cream &amp; Frozen Yogurt</th>
      <th>Vietnamese</th>
      <th>Breakfast &amp; Brunch</th>
      <th>Cupcakes</th>
      <th>Filipino</th>
      <th>Poutineries</th>
      <th>International</th>
      <th>Butcher</th>
      <th>Candy Stores</th>
      <th>Beer</th>
      <th>Delicatessen</th>
      <th>Noodles</th>
      <th>Vegan</th>
      <th>Lounges</th>
      <th>Internet Cafes</th>
      <th>Salad</th>
      <th>Gelato</th>
      <th>Soul Food</th>
      <th>Beer Bar</th>
      <th>Hot Pot</th>
      <th>Custom Cakes</th>
      <th>Books</th>
      <th>Mediterranean</th>
      <th>Vitamins &amp; Supplements</th>
      <th>Kosher</th>
      <th>Do-It-Yourself Food</th>
      <th>Hotels &amp; Travel</th>
      <th>Fish &amp; Chips</th>
      <th>Mexican</th>
      <th>Florists</th>
      <th>Home Services</th>
      <th>Active Life</th>
      <th>Tapas/Small Plates</th>
      <th>Greek</th>
      <th>Buffets</th>
      <th>Cantonese</th>
      <th>Nightlife</th>
      <th>Arts &amp; Entertainment</th>
      <th>Flowers</th>
      <th>Flowers &amp; Gifts</th>
      <th>Desserts</th>
      <th>Diners</th>
      <th>Local Flavor</th>
      <th>Sports Bars</th>
      <th>Portuguese</th>
      <th>Lebanese</th>
      <th>Barbeque</th>
      <th>Herbs &amp; Spices</th>
      <th>Italian</th>
      <th>Specialty Food</th>
      <th>Modern European</th>
      <th>Dance Clubs</th>
      <th>Farmers Market</th>
      <th>Cafes</th>
      <th>Wraps</th>
      <th>Korean</th>
      <th>Cheesesteaks</th>
      <th>Food Stands</th>
      <th>Juice Bars &amp; Smoothies</th>
      <th>Taiwanese</th>
      <th>Health Markets</th>
      <th>Street Vendors</th>
      <th>Food Trucks</th>
      <th>Chicken Wings</th>
      <th>Bubble Tea</th>
      <th>Seafood</th>
      <th>Arts &amp; Crafts</th>
      <th>Cocktail Bars</th>
      <th>Local Services</th>
      <th>Imported Food</th>
      <th>Music Venues</th>
      <th>Karaoke</th>
      <th>Seafood Markets</th>
      <th>Food Delivery Services</th>
      <th>Education</th>
      <th>Soup</th>
      <th>Festivals</th>
      <th>Shaved Ice</th>
      <th>Latin American</th>
      <th>Chicken Shop</th>
      <th>Bagels</th>
      <th>Tea Rooms</th>
      <th>Poke</th>
      <th>Mags</th>
      <th>Irish</th>
      <th>Pizza</th>
      <th>Sandwiches</th>
      <th>Coffee Roasteries</th>
      <th>Hot Dogs</th>
      <th>Breweries</th>
      <th>Organic Stores</th>
      <th>Wine &amp; Spirits</th>
      <th>Caribbean</th>
      <th>Tacos</th>
      <th>Chinese</th>
      <th>Cajun/Creole</th>
      <th>Waffles</th>
      <th>French</th>
      <th>Ramen</th>
      <th>Steakhouses</th>
      <th>Dim Sum</th>
      <th>Comfort Food</th>
      <th>Spanish</th>
      <th>Burgers</th>
      <th>Bakeries</th>
      <th>Vegetarian</th>
      <th>Japanese</th>
      <th>Brasseries</th>
      <th>Chocolatiers &amp; Shops</th>
      <th>Tapas Bars</th>
      <th>Venues &amp; Event Spaces</th>
      <th>Creperies</th>
      <th>German</th>
      <th>Acai Bowls</th>
      <th>African</th>
      <th>Cheese Shops</th>
      <th>Casinos</th>
      <th>Party &amp; Event Planning</th>
      <th>Gift Shops</th>
      <th>International Grocery</th>
      <th>Asian Fusion</th>
      <th>Food Court</th>
      <th>Middle Eastern</th>
      <th>Delis</th>
      <th>Tex-Mex</th>
      <th>Coffee &amp; Tea</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>QXAEGFB4oINsVuTFxEYKFQ</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>gnKjwL_1w79qoiV3IC_xQQ</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



# Bringing in User Dataset and starting to clean 


```python
user = pd.read_csv('user.csv')
```

    /home/mubarakb/miniconda3/lib/python3.7/site-packages/IPython/core/interactiveshell.py:3020: DtypeWarning: Columns (7) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)



```python
user.head()
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
      <th>user_id</th>
      <th>name</th>
      <th>review_count</th>
      <th>yelping_since</th>
      <th>useful</th>
      <th>funny</th>
      <th>cool</th>
      <th>elite</th>
      <th>friends</th>
      <th>fans</th>
      <th>...</th>
      <th>compliment_more</th>
      <th>compliment_profile</th>
      <th>compliment_cute</th>
      <th>compliment_list</th>
      <th>compliment_note</th>
      <th>compliment_plain</th>
      <th>compliment_cool</th>
      <th>compliment_funny</th>
      <th>compliment_writer</th>
      <th>compliment_photos</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>l6BmjZMeQD3rDxWUbiAiow</td>
      <td>Rashmi</td>
      <td>95</td>
      <td>2013-10-08 23:11:33</td>
      <td>84</td>
      <td>17</td>
      <td>25</td>
      <td>2015,2016,2017</td>
      <td>c78V-rj8NQcQjOI8KP3UEA, alRMgPcngYSCJ5naFRBz5g...</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4XChL029mKr5hydo79Ljxg</td>
      <td>Jenna</td>
      <td>33</td>
      <td>2013-02-21 22:29:06</td>
      <td>48</td>
      <td>22</td>
      <td>16</td>
      <td>NaN</td>
      <td>kEBTgDvFX754S68FllfCaA, aB2DynOxNOJK9st2ZeGTPg...</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bc8C_eETBWL0olvFSJJd0w</td>
      <td>David</td>
      <td>16</td>
      <td>2013-10-04 00:16:10</td>
      <td>28</td>
      <td>8</td>
      <td>10</td>
      <td>NaN</td>
      <td>4N-HU_T32hLENLntsNKNBg, pSY2vwWLgWfGVAAiKQzMng...</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dD0gZpBctWGdWo9WlGuhlA</td>
      <td>Angela</td>
      <td>17</td>
      <td>2014-05-22 15:57:30</td>
      <td>30</td>
      <td>4</td>
      <td>14</td>
      <td>NaN</td>
      <td>RZ6wS38wnlXyj-OOdTzBxA, l5jxZh1KsgI8rMunm-GN6A...</td>
      <td>5</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MM4RJAeH6yuaN8oZDSt0RA</td>
      <td>Nancy</td>
      <td>361</td>
      <td>2013-10-23 07:02:50</td>
      <td>1114</td>
      <td>279</td>
      <td>665</td>
      <td>2015,2016,2017,2018</td>
      <td>mbwrZ-RS76V1HoJ0bF_Geg, g64lOV39xSLRZO0aQQ6DeQ...</td>
      <td>39</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>16</td>
      <td>57</td>
      <td>80</td>
      <td>80</td>
      <td>25</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
user.shape
```




    (1637138, 22)




```python
user.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1637138 entries, 0 to 1637137
    Data columns (total 22 columns):
    user_id               1637138 non-null object
    name                  1637135 non-null object
    review_count          1637138 non-null int64
    yelping_since         1637138 non-null object
    useful                1637138 non-null int64
    funny                 1637138 non-null int64
    cool                  1637138 non-null int64
    elite                 71377 non-null object
    friends               1637138 non-null object
    fans                  1637138 non-null int64
    average_stars         1637138 non-null float64
    compliment_hot        1637138 non-null int64
    compliment_more       1637138 non-null int64
    compliment_profile    1637138 non-null int64
    compliment_cute       1637138 non-null int64
    compliment_list       1637138 non-null int64
    compliment_note       1637138 non-null int64
    compliment_plain      1637138 non-null int64
    compliment_cool       1637138 non-null int64
    compliment_funny      1637138 non-null int64
    compliment_writer     1637138 non-null int64
    compliment_photos     1637138 non-null int64
    dtypes: float64(1), int64(16), object(5)
    memory usage: 274.8+ MB



```python
user.columns
```




    Index(['user_id', 'name', 'review_count', 'yelping_since', 'useful', 'funny',
           'cool', 'elite', 'friends', 'fans', 'average_stars', 'compliment_hot',
           'compliment_more', 'compliment_profile', 'compliment_cute',
           'compliment_list', 'compliment_note', 'compliment_plain',
           'compliment_cool', 'compliment_funny', 'compliment_writer',
           'compliment_photos'],
          dtype='object')




```python
user1= user[['user_id','review_count','average_stars']]
```


```python
user1.head()
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
      <th>user_id</th>
      <th>review_count</th>
      <th>average_stars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>l6BmjZMeQD3rDxWUbiAiow</td>
      <td>95</td>
      <td>4.03</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4XChL029mKr5hydo79Ljxg</td>
      <td>33</td>
      <td>3.63</td>
    </tr>
    <tr>
      <th>2</th>
      <td>bc8C_eETBWL0olvFSJJd0w</td>
      <td>16</td>
      <td>3.71</td>
    </tr>
    <tr>
      <th>3</th>
      <td>dD0gZpBctWGdWo9WlGuhlA</td>
      <td>17</td>
      <td>4.85</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MM4RJAeH6yuaN8oZDSt0RA</td>
      <td>361</td>
      <td>4.08</td>
    </tr>
  </tbody>
</table>
</div>




```python
user1.isna().sum()
```




    user_id          0
    review_count     0
    average_stars    0
    dtype: int64




```python
user1.shape
```




    (1637138, 3)




```python
user1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1637138 entries, 0 to 1637137
    Data columns (total 3 columns):
    user_id          1637138 non-null object
    review_count     1637138 non-null int64
    average_stars    1637138 non-null float64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 37.5+ MB



```python

```


```python

```


```python

```


```python

```

# Create SQLlite Tables in memory 


```python
from sqlalchemy import create_engine
```


```python
engine = create_engine('sqlite:///reviews.db', echo=False)
```


```python

```


```python

```

### USER1 SQL Table/   Table Name= 'Users'


```python
user1.to_sql('users', con=engine)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-62-1025f52c1c72> in <module>
    ----> 1 user1.to_sql('users', con=engine)
    

    ~/miniconda3/lib/python3.7/site-packages/pandas/core/generic.py in to_sql(self, name, con, schema, if_exists, index, index_label, chunksize, dtype, method)
       2520         sql.to_sql(self, name, con, schema=schema, if_exists=if_exists,
       2521                    index=index, index_label=index_label, chunksize=chunksize,
    -> 2522                    dtype=dtype, method=method)
       2523 
       2524     def to_pickle(self, path, compression='infer',


    ~/miniconda3/lib/python3.7/site-packages/pandas/io/sql.py in to_sql(frame, name, con, schema, if_exists, index, index_label, chunksize, dtype, method)
        457     pandas_sql.to_sql(frame, name, if_exists=if_exists, index=index,
        458                       index_label=index_label, schema=schema,
    --> 459                       chunksize=chunksize, dtype=dtype, method=method)
        460 
        461 


    ~/miniconda3/lib/python3.7/site-packages/pandas/io/sql.py in to_sql(self, frame, name, if_exists, index, index_label, schema, chunksize, dtype, method)
       1170                          if_exists=if_exists, index_label=index_label,
       1171                          schema=schema, dtype=dtype)
    -> 1172         table.create()
       1173         table.insert(chunksize, method=method)
       1174         if (not name.isdigit() and not name.islower()):


    ~/miniconda3/lib/python3.7/site-packages/pandas/io/sql.py in create(self)
        572             if self.if_exists == 'fail':
        573                 raise ValueError(
    --> 574                     "Table '{name}' already exists.".format(name=self.name))
        575             elif self.if_exists == 'replace':
        576                 self.pd_sql.drop_table(self.name, self.schema)


    ValueError: Table 'users' already exists.



```python
engine.execute("SELECT * FROM users").fetchall()
```


```python

```

### Business2 SQL Table/  Table Name= 'business'


```python
business2.to_sql('business', con=engine)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-4-6bde95831e0f> in <module>
    ----> 1 business2.to_sql('business', con=engine)
    

    NameError: name 'business2' is not defined



```python
engine.execute("SELECT * FROM business").fetchall()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-5-b06ffa03720a> in <module>
    ----> 1 engine.execute("SELECT * FROM business").fetchall()
    

    NameError: name 'engine' is not defined


### Create categories business table that have bus id and categories/ Table name='categories'


```python
rest_cat_bus_id.to_sql('categories', con=engine)
```


```python
engine.execute("SELECT * FROM categories").fetchone()
```




    (0, 'QXAEGFB4oINsVuTFxEYKFQ', 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)



### Reviews1 SQL Table/ Table Name= 'reviews'


```python
reviews1.to_sql('reviews', con=engine)
```


```python
engine.execute("SELECT * FROM reviews").fetchone()
```


```python
engine.execute("SELECT stars FROM reviews").fetchone()
```




    (1.0,)




```python

```


```python

```


```python
# 
```


```python

```


```python

```


```python

```

### Query Sql table to give me only restaurant reviews. 



```python
#got just the reviews
rest_reviews= engine.execute('SELECT text FROM reviews INNER JOIN business ON reviews.business_id=business.business_id').fetchall()
```


```python
len(rest_reviews)
```




    4580299




```python
#got the reviews and ratings to match up 
rest_reviews_rating= engine.execute('SELECT reviews.stars, text FROM reviews INNER JOIN business ON reviews.business_id=business.business_id').fetchall()
```


```python
len(rest_reviews_rating)
```




    4580299




```python
rest_reviews_rating[345]
```




    (4.0, "I recently visited Pieology for the first time with friends recently and was really pleased with the selection and taste of the pies. For a set price ... (279 characters truncated) ...  and will return in the future.  Staff is friendly and efficient. It was a relatively quiet late afternoon so the wait was negligible.  Check it out.")




```python

```


```python

```


```python
#GET reviews with only 1 stars and 2 and 3 and 4 and 5 seperatly 
```


```python
one_star_reviews= engine.execute('SELECT reviews.stars, text FROM reviews INNER JOIN business ON reviews.business_id=business.business_id WHERE reviews.stars= 1.0').fetchall()
```


```python
one_star_reviews[0:2]
```




    [(1.0, "This place has gone down hill.  Clearly they have cut back on staff and food quality\n\nMany of the reviews were written before the menu changed.  I' ... (82 characters truncated) ...  slow & my salad, which was $15, was as bad as it gets.\n\nIt's just not worth spending the money on this place when there are so many other options."),
     (1.0, "Walked in around 4 on a Friday afternoon, we sat at a table just off the bar and walked out after 5 min or so. Don't even think they realized we walk ... (136 characters truncated) ... out. Oh well, the location they are at has been about 5 different things over the past several years, so they will just be added to the list. SMDH!!!")]




```python
len(one_star_reviews)
```




    544139




```python

```


```python
two_star_reviews= engine.execute('SELECT reviews.stars, text FROM reviews INNER JOIN business ON reviews.business_id=business.business_id WHERE reviews.stars= 2.0').fetchall()
```


```python
two_star_reviews[0:2]
```




    [(2.0, 'I was really looking forward to visiting after having some of their beers. The "Man O\'War" quickly became my favorite DIPA; the Rusulka Vanilla Stou ... (1136 characters truncated) ... ast & cheese was good, but by the time we were able to dig into their heartiest offering of food, saltines and butter may have been equally pleasing.'),
     (2.0, 'Went here last weekend and was pretty disappointed. They did not have one thing that was pictured and recommended on yelp as being good. We started o ... (517 characters truncated) ... name attached to this restaurant and going on such an empty stomach we had such high hopes. The service was great which is why i gave it three stars.')]




```python
len(two_star_reviews)
```




    420637




```python

```


```python
three_star_reviews= engine.execute('SELECT reviews.stars, text FROM reviews INNER JOIN business ON reviews.business_id=business.business_id WHERE reviews.stars= 3.0').fetchall()
```


```python
three_star_reviews[0:2]
```




    [(3.0, "Tracy dessert had a big name in Hong Kong and the one in First Markham place has been here for many years now! \n\nCame in for some Chinese dessert,  ... (984 characters truncated) ... f tables they had were just perfect because no one really waited for seats for a long time, but the tables kept filling up once a table was finished."),
     (3.0, 'I love chinese food and I love mexican food. What can go wrong? A couple of things. First things first, this place is more of a "rice bowl" kind of p ... (1739 characters truncated) ... g and throwing molotov cocktails inside. I used the bathroom like 5 times. I don\'t recommend eating this place if you have a lot to do the next day.')]




```python
len(three_star_reviews)
```




    604359




```python

```


```python
four_star_reviews= engine.execute('SELECT reviews.stars, text FROM reviews INNER JOIN business ON reviews.business_id=business.business_id WHERE reviews.stars= 4.0').fetchall()
```


```python
four_star_reviews[0:2]
```




    [(4.0, 'I\'ll be the first to admit that I was not excited about going to La Tavolta. Being a food snob, when a group of friends suggested we go for dinner I ... (1266 characters truncated) ... on\'t go with a date unless you\'re fighting and don\'t feel like hearing anything they have to say.  Ask to sit in the side room if it\'s available.'),
     (4.0, 'Like walking back in time, every Saturday morning my sister and I was in a bowling league and after we were done, we\'d spend a few quarters playing  ... (362 characters truncated) ...  80\'s theme.  There is even a Ms pac man!  It was fun to spend an afternoon playing the machines and remembering all the fun of my early teen years.')]




```python
len(four_star_reviews)
```




    1191087




```python

```


```python
five_star_reviews= engine.execute('SELECT reviews.stars, text FROM reviews INNER JOIN business ON reviews.business_id=business.business_id WHERE reviews.stars= 5.0').fetchall()
```


```python
five_star_reviews[0:2]
```




    [(5.0, "Went in for a lunch. Steak sandwich was delicious, and the Caesar salad had an absolutely delicious dressing, with a perfect amount of dressing, and  ... (115 characters truncated) ...  were pretty good.\n\nThe Server, Dawn, was friendly and accommodating. Very happy with her.\n\nIn summation, a great pub experience. Would go again!"),
     (5.0, "You can't really find anything wrong with this place, the pastas and pizzas are both amazing and high quality, the price is very reasonable, the owne ... (107 characters truncated) ... ecause it's downtown there are lots of options around but that's not always the case as there is also a lot of poor quality food in downtown as well.")]




```python
len(five_star_reviews)
```




    1820077




```python

```


```python

```


```python

```


```python
## Taking a 1-5 star random of 200,000
```


```python
rand_items_one_star = random.sample(one_star_reviews, 40000)
```


```python
rand_items_two_star = random.sample(two_star_reviews, 40000)
```


```python
rand_items_three_star = random.sample(three_star_reviews, 40000)
```


```python
rand_items_four_star = random.sample(four_star_reviews, 40000)
```


```python
rand_items_five_star = random.sample(five_star_reviews, 40000)
```


```python
#function to make into a list. Could have done many other ways
rand_list=[]
def combine(list):
    for i in list:
        rand_list.append(i)
    
```


```python
combine(rand_items_one_star)
```


```python
combine(rand_items_two_star)
```


```python
combine(rand_items_three_star)
```


```python
combine(rand_items_four_star)
```


```python
combine(rand_items_five_star)
```


```python
len(rand_list)
```




    200000




```python
# Making a df with the random downsampled 
```


```python
ds_df= pd.DataFrame(rand_list, columns=['rating', 'review'])
```


```python
ds_df.head()
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
      <th>rating</th>
      <th>review</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>This review is for a Saturday night dinner ser...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>I'm am still in disbelief at noon today and tr...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>Owner is EXTREMELY rude, ordered food and told...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>Based on the reviews, I was surprised at how d...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>Let me start by saying the food was A1. Everyo...</td>
    </tr>
  </tbody>
</table>
</div>




```python
ds_df.shape
```




    (200000, 2)




```python
df_df1= ds_df
```


```python
#counting words in each review
df_df1['count_rev'] = [len(x.split()) for x in ds_df.review]
```


```python
df_df1.head()
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
      <th>rating</th>
      <th>review</th>
      <th>count_rev</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>This review is for a Saturday night dinner ser...</td>
      <td>320</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>I'm am still in disbelief at noon today and tr...</td>
      <td>274</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.0</td>
      <td>Owner is EXTREMELY rude, ordered food and told...</td>
      <td>115</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>Based on the reviews, I was surprised at how d...</td>
      <td>125</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>Let me start by saying the food was A1. Everyo...</td>
      <td>169</td>
    </tr>
  </tbody>
</table>
</div>




```python
#summing total words
df_df1['count_rev'].sum()
```




    23439714




```python

```


```python
df_df1.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 200000 entries, 0 to 199999
    Data columns (total 3 columns):
    rating       200000 non-null float64
    review       200000 non-null object
    count_rev    200000 non-null int64
    dtypes: float64(1), int64(1), object(1)
    memory usage: 4.6+ MB


# NLTK


```python
#Creating regex pattern to parse the each overview
from nltk.stem.snowball import SnowballStemmer
pattern = "([a-zA-Z]+(?:'[a-z]+)?)"

#Stemming words using SnowballStemmer
stemmer = SnowballStemmer("english")

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#Creating unique list of available stopwords in the NLTK corpus
from nltk.corpus import stopwords
# this line below opened up the dowloader for me to go in download this specific stopwords 
# nltk.download()
stop_words = set(stopwords.words('english'))
```


```python

```


```python

```


```python
giant_string = []
all_words = []
for review in df_df1.review: 
    cleaned = nltk.regexp_tokenize(str(review), pattern)
    tokens = [i.lower() for i in cleaned]
    tokens_stopped = [w for w in tokens if not w in stop_words]
    meta_stemmed = [stemmer.stem(word) for word in tokens_stopped]
    string = ' '.join(meta_stemmed)
    giant_string.append(string)
    [all_words.append(w) for w in meta_stemmed]
```


```python
len(all_words)
```




    11781773




```python
giant_string[0]
```




    "review saturday night dinner servic happi hour servic guess definit minor thought best thing visit serendipit meet eric lisa r join us dinner beyond place mediocr food terribl servic moment walk door noth wait wait host ess anyon acknowledg us tell us put name etc seat order wait ice tea came everyon els drink serv prompt waitress wait everyon entre order bun bo hue green papaya salad came order green papaya salad larg steam shrimp instead dehydr small bay shrimp add addit salti dish okay bun bo hue good robust beef flavor spici miss pig hock pig blood ask forgot waitress repli gee tell us order can't send someon across street market buy add insult injuri wait waitress refil ice tea see waiter carri pitcher ice tea look empti glass walk tabl fill glass ice tea walk away set pitcher realli dude know tabl guess would'v die spot fill even inquir anoth custom ice tea end meal wait could leav place fast enough"




```python
all_words_fd = nltk.FreqDist(all_words)
```


```python
all_words_fd
```




    FreqDist({'food': 158656, 'place': 127027, 'good': 126143, 'order': 118778, 'like': 95494, 'time': 89564, 'servic': 85376, 'go': 81945, 'get': 81450, 'one': 76051, ...})




```python

```


```python

```


```python
# So changing the order here and split the test and train data before we go into the vectorization process.
```


```python
from sklearn.model_selection import train_test_split
X = giant_string
targets= df_df1['rating']
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.2)
```


```python

```


```python

```


```python
# tfidf on features
#in this second case we are now vectorizing after splitting the data instead of vectorizing before splitting.  
```


```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
```


```python
tfidf = TfidfVectorizer(max_features=5000)
x_train = tfidf.fit_transform(X_train)
x_test= tfidf.transform(X_test)
x_train_features_tfidf = pd.DataFrame(x_train.toarray(), columns=tfidf.get_feature_names())
x_test_features_tfidf = pd.DataFrame(x_test.toarray(), columns=tfidf.get_feature_names())
```


```python
x_train_features_tfidf.shape
```




    (160000, 5000)




```python

```


```python
x_train_features_tfidf.head()
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
      <th>abandon</th>
      <th>abil</th>
      <th>abl</th>
      <th>abrupt</th>
      <th>absent</th>
      <th>absolut</th>
      <th>absorb</th>
      <th>absurd</th>
      <th>abund</th>
      <th>abus</th>
      <th>...</th>
      <th>yummi</th>
      <th>yup</th>
      <th>yuzu</th>
      <th>zen</th>
      <th>zero</th>
      <th>zing</th>
      <th>zipp</th>
      <th>ziti</th>
      <th>zone</th>
      <th>zucchini</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.068623</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 5000 columns</p>
</div>




```python
x_test_features_tfidf.head()
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
      <th>abandon</th>
      <th>abil</th>
      <th>abl</th>
      <th>abrupt</th>
      <th>absent</th>
      <th>absolut</th>
      <th>absorb</th>
      <th>absurd</th>
      <th>abund</th>
      <th>abus</th>
      <th>...</th>
      <th>yummi</th>
      <th>yup</th>
      <th>yuzu</th>
      <th>zen</th>
      <th>zero</th>
      <th>zing</th>
      <th>zipp</th>
      <th>ziti</th>
      <th>zone</th>
      <th>zucchini</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 5000 columns</p>
</div>




```python

```

#### hasing vectorizer below is like a count vectorizer. 


```python
# from sklearn.feature_extraction.text import HashingVectorizer

```


```python
# vectorizer = HashingVectorizer(n_features=5000)
# v= vectorizer.fit_transform(giant_string)
# features_hashing= pd.DataFrame(v.toarray(), columns= v.get_feature_names())

```


```python


#the 5 lines below are using a count_vecotrizer. 


# bow_transformer = CountVectorizer().fit(giant_string)
```


```python
# len(bow_transformer.vocabulary_)
```


```python
# bow_25 = bow_transformer.transform([giant_string[25]])
```


```python

```


```python
# review25= df_df1['review'][25]
```


```python
# bow_25 = bow_transformer.transform([review25])
```


```python
# bow_25
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

## Model Running



```python
from sklearn import linear_model
```


```python
lm = linear_model.LinearRegression()
model = lm.fit(x_train,y_train)
```


```python
from sklearn.metrics import mean_squared_error
yhat = model.predict(x_test)

RMSE = np.sqrt(mean_squared_error(y_test, yhat))
z_score = RMSE/np.std(y_test)
R_squared = model.score(x_test, y_test)
print("RMSE = {}, z_score = {}, R_squared = {}".format(RMSE, z_score, R_squared))
```

    RMSE = 0.8245592614468864, z_score = 0.5825635792299276, R_squared = 0.660619676154816



```python

```


```python
#pickle the linreg model for rating. named it log_reg_rating but its lin_reg_rating 
import pickle
pickling_on = open("log_reg_rating","wb")
pickle.dump(model, pickling_on)
pickling_on.close()

```


```python
pickle_off = open("log_reg_rating","rb")
lin_reg_rating = pickle.load(pickle_off)

```


```python

```


```python
#second model for running
```


```python

from sklearn.svm import SVR
svr_lin = SVR(kernel='linear', verbose=10)
rat_svr = svr_lin.fit(x_train, y_train)
```

    [LibSVM]


```python

```


```python
#model grid search with pipeline


```


```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

pipe_rf = Pipeline([('tf_idf_vectorizer',TfidfVectorizer()),
                   ('rf', RandomForestRegressor(n_jobs=-1))])

parameters = {'tf_idf_vectorizer__ngram_range': [(1,1),(1,2)],
              'tf_idf_vectorizer__max_features': [1000,2000,5000],
              'rf__n_estimators':[10,20,50,100],
             'rf__max_features':['auto',5,20]}

model_grid = GridSearchCV(pipe_rf,parameters,n_jobs=-1, verbose=10)
```


```python
#passing in X_train because this is before we vecotrized earlier for the logreg. this will do it for us. 
model_grid.fit(X_train,y_train)
```

    Fitting 3 folds for each of 72 candidates, totalling 216 fits


    /home/mubarakb/miniconda3/lib/python3.7/site-packages/sklearn/model_selection/_split.py:2053: FutureWarning: You should specify a value for 'cv' instead of relying on the default value. The default value will change from 3 to 5 in version 0.22.
      warnings.warn(CV_WARNING, FutureWarning)
    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 22 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   6 tasks      | elapsed: 51.6min
    [Parallel(n_jobs=-1)]: Done  17 tasks      | elapsed: 65.3min



```python

```


```python
###### testing code dont know how it works #####
```


```python
rev1 = input('')
```

    



```python





for overview in [rev1]: 
    rev_string = []
    cleaned = nltk.regexp_tokenize(str(overview), pattern)
    tokens = [i.lower() for i in cleaned]
    tokens_stopped = [w for w in tokens if not w in stop_words]
    meta_stemmed = [stemmer.stem(word) for word in tokens_stopped]
    string = ' '.join(meta_stemmed)
    rev_string.append(string)
tt = tfidf.transform(rev_string)


print(rev1)
print('')
rev_pred = lin_reg_rating.predict(tt)
for i in rev_pred:
    if i < 1.5:
        print ('1 Star')
    else:
        if i < 2.5:
            print ('2 star')
        else:
            if i < 3.5:
                print ('3 star')
            else:
                if i < 4.5:
                    print ('4 stars')
                else:
                    print ('5 stars')


```

    fuck uou
    
    2 star



```python
rev_pred
```




    array([2.37453486])




```python

```


```python

```


```python

```


```python
from tpot import TPOTRegressor
from tpot.config import regressor_config_sparse
```


```python
pipeline_optimizer = TPOTRegressor(config_dict=regressor_config_sparse, verbosity=3,periodic_checkpoint_folder='checkpoint',generations=10, population_size=10)


```


```python
pipeline_optimizer.fit(x_train, y_train)
# pipeline_optimizer.export('best_pipe.py')
```

    11 operators have been imported by TPOT.
    Warning: Since the input matrix is a sparse matrix, please makes sure all the operators in the customized config dictionary supports sparse matriies.



    HBox(children=(IntProgress(value=0, description='Optimization Progress', max=110, style=ProgressStyle(descript…


    Skipped pipeline #5 due to time out. Continuing to the next pipeline.
    Skipped pipeline #7 due to time out. Continuing to the next pipeline.
    Skipped pipeline #9 due to time out. Continuing to the next pipeline.
    Skipped pipeline #11 due to time out. Continuing to the next pipeline.
    Skipped pipeline #14 due to time out. Continuing to the next pipeline.
    


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    ~/miniconda3/lib/python3.7/site-packages/tpot/base.py in fit(self, features, target, sample_weight, groups)
        660                     verbose=self.verbosity,
    --> 661                     per_generation_function=self._check_periodic_pipeline
        662                 )


    ~/miniconda3/lib/python3.7/site-packages/tpot/gp_deap.py in eaMuPlusLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, pbar, stats, halloffame, verbose, per_generation_function)
        243         if per_generation_function is not None:
    --> 244             per_generation_function()
        245 


    ~/miniconda3/lib/python3.7/site-packages/tpot/base.py in _check_periodic_pipeline(self)
        937         """
    --> 938         self._update_top_pipeline()
        939         if self.periodic_checkpoint_folder is not None:


    ~/miniconda3/lib/python3.7/site-packages/tpot/base.py in _update_top_pipeline(self)
        739             if not self._optimized_pipeline:
    --> 740                 raise RuntimeError('There was an error in the TPOT optimization '
        741                                    'process. This could be because the data was '


    RuntimeError: There was an error in the TPOT optimization process. This could be because the data was not formatted properly, or because data for a regression problem was provided to the TPOTClassifier object. Please make sure you passed the data to TPOT correctly.

    
    During handling of the above exception, another exception occurred:


    RuntimeError                              Traceback (most recent call last)

    <ipython-input-225-e6e9dde440ef> in <module>
    ----> 1 pipeline_optimizer.fit(x_train, y_train)
          2 # pipeline_optimizer.export('best_pipe.py')


    ~/miniconda3/lib/python3.7/site-packages/tpot/base.py in fit(self, features, target, sample_weight, groups)
        691                     # raise the exception if it's our last attempt
        692                     if attempt == (attempts - 1):
    --> 693                         raise e
        694             return self
        695 


    ~/miniconda3/lib/python3.7/site-packages/tpot/base.py in fit(self, features, target, sample_weight, groups)
        682                         self._pbar.close()
        683 
    --> 684                     self._update_top_pipeline()
        685                     self._summary_of_best_pipeline(features, target)
        686                     # Delete the temporary cache before exiting


    ~/miniconda3/lib/python3.7/site-packages/tpot/base.py in _update_top_pipeline(self)
        738 
        739             if not self._optimized_pipeline:
    --> 740                 raise RuntimeError('There was an error in the TPOT optimization '
        741                                    'process. This could be because the data was '
        742                                    'not formatted properly, or because data for '


    RuntimeError: There was an error in the TPOT optimization process. This could be because the data was not formatted properly, or because data for a regression problem was provided to the TPOTClassifier object. Please make sure you passed the data to TPOT correctly.



```python

```


```python

```

# USEFULNESS 


```python
#### Starting  SQl   ##### SQL QUERY for usefulness######
```


```python
Useful = engine.execute('SELECT reviews.stars AS review_stars, reviews.useful AS reviews_useful, reviews.text AS review_text, users.review_count AS user_rev_count,business.stars AS bus_avg_rating From reviews INNER JOIN business ON business.business_id = reviews.business_id INNER JOIN users ON users.user_id = reviews.user_id')
#could write straight to pd
# Useful= pd.read_sql("SQLQuery", engine)

```


```python

```


```python
use_df= pd.DataFrame(Useful, columns=['review_stars','useful_rev','review_text','user_rev_count','bus_stars'])
```


```python
use_df.head(5)
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
      <th>review_stars</th>
      <th>useful_rev</th>
      <th>review_text</th>
      <th>user_rev_count</th>
      <th>bus_stars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.0</td>
      <td>0</td>
      <td>Went in for a lunch. Steak sandwich was delici...</td>
      <td>4</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>0</td>
      <td>I'll be the first to admit that I was not exci...</td>
      <td>1</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>5</td>
      <td>Tracy dessert had a big name in Hong Kong and ...</td>
      <td>600</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>3</td>
      <td>This place has gone down hill.  Clearly they h...</td>
      <td>88</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2.0</td>
      <td>1</td>
      <td>I was really looking forward to visiting after...</td>
      <td>13</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
use_df.shape
```




    (4580299, 5)




```python


```


```python
usefuldf= use_df
```


```python
usefuldf.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4580299 entries, 0 to 4580298
    Data columns (total 5 columns):
    review_stars      float64
    useful_rev        int64
    review_text       object
    user_rev_count    int64
    bus_stars         float64
    dtypes: float64(2), int64(2), object(1)
    memory usage: 174.7+ MB



```python

```


```python
use_df = usefuldf[usefuldf.useful_rev >= 2]
```


```python
use_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 1009932 entries, 2 to 4580298
    Data columns (total 5 columns):
    review_stars      1009932 non-null float64
    useful_rev        1009932 non-null int64
    review_text       1009932 non-null object
    user_rev_count    1009932 non-null int64
    bus_stars         1009932 non-null float64
    dtypes: float64(2), int64(2), object(1)
    memory usage: 46.2+ MB



```python
use_df.head()
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
      <th>review_stars</th>
      <th>useful_rev</th>
      <th>review_text</th>
      <th>user_rev_count</th>
      <th>bus_stars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>5</td>
      <td>Tracy dessert had a big name in Hong Kong and ...</td>
      <td>600</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>3</td>
      <td>This place has gone down hill.  Clearly they h...</td>
      <td>88</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5.0</td>
      <td>9</td>
      <td>If you are looking for the best pierogies in P...</td>
      <td>776</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1.0</td>
      <td>6</td>
      <td>Met a friend for dinner there tonight. The ser...</td>
      <td>16</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>30</th>
      <td>3.0</td>
      <td>2</td>
      <td>One day after I satisfy my frozen yogurt cravi...</td>
      <td>444</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
use_df.shape
```




    (1009932, 5)




```python
use_df1= use_df.sample(n=40000)
```


```python
use_df1.shape
```




    (40000, 5)




```python

```


```python
non_useful_df= usefuldf[usefuldf.useful_rev == 0]
```


```python
non_useful_df.head()
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
      <th>review_stars</th>
      <th>useful_rev</th>
      <th>review_text</th>
      <th>user_rev_count</th>
      <th>bus_stars</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.0</td>
      <td>0</td>
      <td>Went in for a lunch. Steak sandwich was delici...</td>
      <td>4</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.0</td>
      <td>0</td>
      <td>I'll be the first to admit that I was not exci...</td>
      <td>1</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.0</td>
      <td>0</td>
      <td>Like walking back in time, every Saturday morn...</td>
      <td>866</td>
      <td>4.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.0</td>
      <td>0</td>
      <td>Walked in around 4 on a Friday afternoon, we s...</td>
      <td>2</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>7</th>
      <td>4.0</td>
      <td>0</td>
      <td>Wow. So surprised at the one and two star revi...</td>
      <td>12</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
non_useful_df.shape

```




    (2579691, 5)




```python
non_useful_df1= non_useful_df.sample(n=40000)
```


```python
non_useful_df1.shape
```




    (40000, 5)




```python
# use_and_non_df= use_df[use_df.useful_rev == 0 & use_df.useful_rev>= 2]
```


```python
#merge useful 1 and non useful df 1 we downsampled both for computational purposes
use_and_non_df= pd.concat([use_df1, non_useful_df1])

```


```python
use_and_non_df.shape
```




    (80000, 5)




```python

```


```python
use_and_non_df.shape
```




    (80000, 5)




```python
#set empty column to binarize useful into 0 or 1 
use_and_non_df['useful']  = 0
```


```python
#if useful is over 0 then label 1 if not it keeps the same 
use_and_non_df.loc[use_and_non_df['useful_rev'] > 0 , 'useful'] = 1
```


```python
#spelled revieiew text wrong so if error pops up or there is a change look here 
#this is counting text and in a review 
use_and_non_df['count_rev'] = [len(x.split()) for x in use_and_non_df.review_text]
```


```python
# create category for difference between user rating and business stars 
use_and_non_df['rev_bus_rat_diff'] =  (use_and_non_df['review_stars']-use_and_non_df['bus_stars']).abs()
```


```python
use_and_non_df= use_and_non_df.reset_index()
```


```python
use_and_non_df.head()
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
      <th>index</th>
      <th>review_stars</th>
      <th>useful_rev</th>
      <th>review_text</th>
      <th>user_rev_count</th>
      <th>bus_stars</th>
      <th>useful</th>
      <th>count_rev</th>
      <th>rev_bus_rat_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4015789</td>
      <td>5.0</td>
      <td>2</td>
      <td>If you want a healthy choice that taste good, ...</td>
      <td>7</td>
      <td>4.5</td>
      <td>1</td>
      <td>30</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3300746</td>
      <td>1.0</td>
      <td>2</td>
      <td>Took over an hour and half to get dinner. In o...</td>
      <td>9</td>
      <td>3.5</td>
      <td>1</td>
      <td>25</td>
      <td>2.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3277908</td>
      <td>4.0</td>
      <td>3</td>
      <td>Ohh man the beer selection here is amazing! So...</td>
      <td>146</td>
      <td>4.5</td>
      <td>1</td>
      <td>155</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3333522</td>
      <td>4.0</td>
      <td>50</td>
      <td>So I'm price conscious about salads.  But I ca...</td>
      <td>538</td>
      <td>3.5</td>
      <td>1</td>
      <td>268</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>119208</td>
      <td>5.0</td>
      <td>2</td>
      <td>We stayed at the hotel on Business...went down...</td>
      <td>85</td>
      <td>3.0</td>
      <td>1</td>
      <td>75</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
use_and_non_df1= use_and_non_df.drop(['index','review_stars','useful_rev','bus_stars'], axis=1)
```


```python
use_and_non_df1.tail(2)
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
      <th>review_text</th>
      <th>user_rev_count</th>
      <th>useful</th>
      <th>count_rev</th>
      <th>rev_bus_rat_diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>79998</th>
      <td>A mom-and-pop hole-in-the-wall that offers del...</td>
      <td>28</td>
      <td>0</td>
      <td>217</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>79999</th>
      <td>Last Friday was my first time visiting this pl...</td>
      <td>16</td>
      <td>0</td>
      <td>741</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
use_and_non_df1.shape
```




    (80000, 5)




```python
use_and_non_df1.isna().sum()
```




    review_text         0
    user_rev_count      0
    useful              0
    count_rev           0
    rev_bus_rat_diff    0
    dtype: int64




```python
### so i have the data like i want and now i want to vectorize text and add it back to the df so some of the same 
#     code might be used before 


## use_and_non_df1 is the one being used 
```


```python
from nltk.stem.snowball import SnowballStemmer
pattern = "([a-zA-Z]+(?:'[a-z]+)?)"

#Stemming words using SnowballStemmer
stemmer = SnowballStemmer("english")

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#Creating unique list of available stopwords in the NLTK corpus
from nltk.corpus import stopwords
# this line below opened up the dowloader for me to go in download this specific stopwords 
# nltk.download()
stop_words = set(stopwords.words('english'))
```


```python
useful_giant_string = []
useful_all_words = []
for review in use_and_non_df1.review_text: 
    cleaned = nltk.regexp_tokenize(str(review), pattern)
    tokens = [i.lower() for i in cleaned]
    tokens_stopped = [w for w in tokens if not w in stop_words]
    meta_stemmed = [stemmer.stem(word) for word in tokens_stopped]
    string = ' '.join(meta_stemmed)
    useful_giant_string.append(string)
    [useful_all_words.append(w) for w in meta_stemmed]
```


```python
## next steps are to put this data back together and then test train split and then tfidf the columns get 5000 
## and then put back in the dataframe and scale it.
len(useful_giant_string)
```




    80000




```python

```


```python
use_and_non_df1['tok_text']= useful_giant_string
```


```python
use_and_non_df1.drop(['review_text'], axis=1, inplace=True)     
```


```python
use_and_non_df1.head(2)
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
      <th>user_rev_count</th>
      <th>useful</th>
      <th>count_rev</th>
      <th>rev_bus_rat_diff</th>
      <th>tok_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7</td>
      <td>1</td>
      <td>30</td>
      <td>0.5</td>
      <td>want healthi choic tast good hk way go fresh h...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>1</td>
      <td>25</td>
      <td>2.5</td>
      <td>took hour half get dinner group dinner came di...</td>
    </tr>
  </tbody>
</table>
</div>




```python
use_and_non_df1.isna().sum()

```




    user_rev_count      0
    useful              0
    count_rev           0
    rev_bus_rat_diff    0
    tok_text            0
    dtype: int64




```python
#cut this into a better chunk of samples 
```


```python
#so im splitting the test and train data 
# i am also dropping user rev count becuase of model training with it, seemed like it wasnt an important feature
from sklearn.model_selection import train_test_split

X1= use_and_non_df1.drop(['useful'], axis=1)
y1= use_and_non_df1['useful']

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2)





# # Build Models
# # Logistic Regression
# from sklearn.linear_model import LogisticRegression
# logreg = LogisticRegression()
# logreg.fit(X_train, y_train)

# print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(logreg.score(X_train, y_train)))
# print('Accuracy of Logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
```


```python
X_test1.shape
```




    (16000, 4)




```python
X_train1.head(2)
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
      <th>user_rev_count</th>
      <th>count_rev</th>
      <th>rev_bus_rat_diff</th>
      <th>tok_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65891</th>
      <td>47</td>
      <td>21</td>
      <td>1.0</td>
      <td>sushi burrito realli good staff super nice att...</td>
    </tr>
    <tr>
      <th>27095</th>
      <td>9</td>
      <td>94</td>
      <td>1.0</td>
      <td>hey last review filter posit one first restaur...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#now i am vectorizing the text data 
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_useful = TfidfVectorizer(max_features=5000)
x_train1 = tfidf_useful.fit_transform(X_train1.tok_text)
x_test1= tfidf_useful.transform(X_test1.tok_text)
x_train1_features_tfidf = pd.DataFrame(x_train1.toarray(), columns=tfidf_useful.get_feature_names())
x_test1_features_tfidf = pd.DataFrame(x_test1.toarray(), columns=tfidf_useful.get_feature_names())
```


```python
x_train1_features_tfidf.head(2)

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
      <th>abandon</th>
      <th>abil</th>
      <th>abl</th>
      <th>abrupt</th>
      <th>absolut</th>
      <th>absorb</th>
      <th>absurd</th>
      <th>abund</th>
      <th>ac</th>
      <th>acai</th>
      <th>...</th>
      <th>zen</th>
      <th>zero</th>
      <th>zest</th>
      <th>zesti</th>
      <th>zing</th>
      <th>zip</th>
      <th>zombi</th>
      <th>zone</th>
      <th>zoo</th>
      <th>zucchini</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 5000 columns</p>
</div>




```python
## now take these merge them back with the training ones
```


```python
#had to match the index because the tfidf made new indexes and joining thme would be a problem. so for this and test
x_train1_features_tfidf.index = X_train1.index
```


```python
X_train12 = X_train1.join(x_train1_features_tfidf)


```


```python
x_test1_features_tfidf.index = X_test1.index
X_test12= X_test1.join(x_test1_features_tfidf)
```


```python
X_test12.head(2)
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
      <th>user_rev_count</th>
      <th>count_rev</th>
      <th>rev_bus_rat_diff</th>
      <th>tok_text</th>
      <th>abandon</th>
      <th>abil</th>
      <th>abl</th>
      <th>abrupt</th>
      <th>absolut</th>
      <th>absorb</th>
      <th>...</th>
      <th>zen</th>
      <th>zero</th>
      <th>zest</th>
      <th>zesti</th>
      <th>zing</th>
      <th>zip</th>
      <th>zombi</th>
      <th>zone</th>
      <th>zoo</th>
      <th>zucchini</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>66949</th>
      <td>39</td>
      <td>75</td>
      <td>0.5</td>
      <td>park free small park lot venu medium size brin...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>53438</th>
      <td>73</td>
      <td>73</td>
      <td>0.5</td>
      <td>great fresh food pita probabl best thing menu ...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 5004 columns</p>
</div>




```python
X_train12.head()
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
      <th>user_rev_count</th>
      <th>count_rev</th>
      <th>rev_bus_rat_diff</th>
      <th>tok_text</th>
      <th>abandon</th>
      <th>abil</th>
      <th>abl</th>
      <th>abrupt</th>
      <th>absolut</th>
      <th>absorb</th>
      <th>...</th>
      <th>zen</th>
      <th>zero</th>
      <th>zest</th>
      <th>zesti</th>
      <th>zing</th>
      <th>zip</th>
      <th>zombi</th>
      <th>zone</th>
      <th>zoo</th>
      <th>zucchini</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65891</th>
      <td>47</td>
      <td>21</td>
      <td>1.0</td>
      <td>sushi burrito realli good staff super nice att...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>27095</th>
      <td>9</td>
      <td>94</td>
      <td>1.0</td>
      <td>hey last review filter posit one first restaur...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>40456</th>
      <td>107</td>
      <td>158</td>
      <td>1.5</td>
      <td>believ use schezuan restaur made beauti quaint...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>32461</th>
      <td>192</td>
      <td>172</td>
      <td>1.5</td>
      <td>fanci look spot fanci look food restaur chef d...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>66337</th>
      <td>1</td>
      <td>119</td>
      <td>1.0</td>
      <td>boyfriend went dailo group peopl tast menu foo...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 5004 columns</p>
</div>




```python
X_train12.drop(['tok_text'], axis=1, inplace=True)
X_test12.drop(['tok_text'],axis=1, inplace=True)
```


```python
X_train12.head(2)
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
      <th>user_rev_count</th>
      <th>count_rev</th>
      <th>rev_bus_rat_diff</th>
      <th>abandon</th>
      <th>abil</th>
      <th>abl</th>
      <th>abrupt</th>
      <th>absolut</th>
      <th>absorb</th>
      <th>absurd</th>
      <th>...</th>
      <th>zen</th>
      <th>zero</th>
      <th>zest</th>
      <th>zesti</th>
      <th>zing</th>
      <th>zip</th>
      <th>zombi</th>
      <th>zone</th>
      <th>zoo</th>
      <th>zucchini</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>65891</th>
      <td>47</td>
      <td>21</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>27095</th>
      <td>9</td>
      <td>94</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 5003 columns</p>
</div>




```python
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train13 = scaler.fit_transform(X_train12)
X_test13 = scaler.transform(X_test12)
```

    /home/mubarakb/miniconda3/lib/python3.7/site-packages/sklearn/preprocessing/data.py:323: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by MinMaxScaler.
      return self.partial_fit(X, y)



```python
X_train13
```




    array([[3.71297118e-03, 1.94741967e-02, 2.50000000e-01, ...,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
           [6.45734119e-04, 9.05550146e-02, 2.50000000e-01, ...,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
           [8.55597708e-03, 1.52872444e-01, 3.75000000e-01, ...,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
           ...,
           [4.68157236e-03, 7.10808179e-02, 0.00000000e+00, ...,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
           [4.03583824e-04, 2.72638754e-02, 3.75000000e-01, ...,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
           [8.07167649e-05, 1.85004869e-02, 2.50000000e-01, ...,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00]])




```python
# Logistic Regression
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train13, y_train1)
```

    /home/mubarakb/miniconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)





    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='warn',
              n_jobs=None, penalty='l2', random_state=None, solver='warn',
              tol=0.0001, verbose=0, warm_start=False)




```python
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train13, y_train1)))
print('Accuracy of Logistic regression classifier on test set: {:.2f}'
     .format(logreg.score(X_test13, y_test1)))
```

    Accuracy of Logistic regression classifier on training set: 0.75
    Accuracy of Logistic regression classifier on test set: 0.70



```python
ypred= logreg.predict(X_test13)
```


```python
#code for printing confusion matrix

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    #Add Normalization Option
    '''prints pretty confusion metric with normalization option '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True Lable')
    plt.xlabel('Predicted Label')
```


```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test1, ypred))
cm = confusion_matrix(y_test1, ypred)
plot_confusion_matrix(cm, [0.0, 1.0])
```

                  precision    recall  f1-score   support
    
               0       0.68      0.74      0.71      7945
               1       0.72      0.66      0.69      8055
    
       micro avg       0.70      0.70      0.70     16000
       macro avg       0.70      0.70      0.70     16000
    weighted avg       0.70      0.70      0.70     16000
    
    Confusion matrix, without normalization
    [[5847 2098]
     [2709 5346]]



![png](DataPreprocessing_files/DataPreprocessing_326_1.png)



```python

```


```python
# just getting the features to see with the coefficents. X_train12 is the last time the data was a full df 
my_features = X_train12.columns
```


```python
#### got the least important features
my_features[np.argsort(logreg.coef_)]
```




    Index([['mush', 'diablo', 'butterscotch', 'yr', 'dissatisfi', 'manger', 'altogeth', 'sudden', 'bullshit', 'syrupi', 'yorker', 'overbear', 'touristi', 'rooftop', 'bouch', 'routin', 'toddler', 'elabor', 'curb', 'nous', 'satisfactori', 'essenc', 'manag', 'pate', 'setup', 'qu', 'born', 'retain', 'nickel', 'easter', 'ruth', 'tar', 'tr', 'butteri', 'brais', 'homeless', 'good', 'isol', 'dread', 'truth', 'roy', 'airi', 'complementari', 'flop', 'field', 'carbonara', 'furnish', 'purpos', 'kitschi', 'contribut', 'motel', 'reuben', 'compot', 'ceasar', 'conge', 'hr', 'consumpt', 'van', 'crawfish', 'januari', 'lid', 'pet', 'stronger', 'robin', 'termin', 'defin', 'churro', 'garbag', 'privaci', 'dock', 'miller', 'consum', 'orlean', 'savor', 'throat', 'grate', 'marrow', 'teeter', 'opportun', 'adam', 'shower', 'khao', 'confirm', 'solo', 'nevada', 'vega', 'moco', 'messag', 'spirit', 'alfredo', 'cow', 'jame', 'draw', 'ramsey', 'careless', 'melon', 'lay', 'summerlici', 'caribbean', 'loung', ...]], dtype='object')




```python
#these features were the most important 
my_features[np.argsort(logreg.coef_)][0][::-1]
```




    array(['user_rev_count', 'count_rev', 'wrote', ..., 'butterscotch',
           'diablo', 'mush'], dtype=object)




```python
#sorting out the values it places on the coeffeicnts 
np.argsort(logreg.coef_)
```




    array([[2911, 1242,  625, ..., 4952,    1,    0]])




```python
usefull= input()
```

    Fuck you



```python

use_pred_data= {'user_rev_count':[21],'count_rev': [len(usefull.split())],'rev_bus_rat_diff': [0.5]}
use_pred_df= pd.DataFrame(data= use_pred_data)
for useful in [usefull]: 
    rev_string = []
    cleaned = nltk.regexp_tokenize(str(useful), pattern)
    tokens = [i.lower() for i in cleaned]
    tokens_stopped = [w for w in tokens if not w in stop_words]
    meta_stemmed = [stemmer.stem(word) for word in tokens_stopped]
    string = ' '.join(meta_stemmed)
    rev_string.append(string)
    usett = tfidf_useful.transform(rev_string)
    usett_features_tfidf = pd.DataFrame(usett.toarray(), columns=tfidf_useful.get_feature_names())
    pred_use_df= use_pred_df.join(usett_features_tfidf)
  
    
```


```python
 pred_use_df.head(1)
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
      <th>user_rev_count</th>
      <th>count_rev</th>
      <th>rev_bus_rat_diff</th>
      <th>abandon</th>
      <th>abil</th>
      <th>abl</th>
      <th>abrupt</th>
      <th>absolut</th>
      <th>absorb</th>
      <th>absurd</th>
      <th>...</th>
      <th>zen</th>
      <th>zero</th>
      <th>zest</th>
      <th>zesti</th>
      <th>zing</th>
      <th>zip</th>
      <th>zombi</th>
      <th>zone</th>
      <th>zoo</th>
      <th>zucchini</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21</td>
      <td>2</td>
      <td>0.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 5003 columns</p>
</div>




```python
logreg.predict_log_proba(pred_use_df)
```

    /home/mubarakb/miniconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:1438: RuntimeWarning: divide by zero encountered in log
      return np.log(self.predict_proba(X))





    array([[-inf,   0.]])




```python
use_pred= logreg.predict(pred_use_df)
```


```python
use_pred
```




    array([1])




```python
logreg.predict_proba(pred_use_df)
```




    array([[0., 1.]])




```python

```


```python

```

## Running usefullness with random forest


```python
# Import the model we are using
from sklearn.ensemble import RandomForestClassifier
# Instantiate model with 1000 decision trees
rf_use = RandomForestClassifier(n_estimators = 200, random_state = 42, max_depth=5, n_jobs=-1)
# Train the model on training data
rf_use.fit(X_train13, y_train1);
```


```python
print('Accuracy of Random Forest classifier on training set: {:.2f}'
     .format(rf_use.score(X_train13, y_train1)))
print('Accuracy of Random Forest classifier on test set: {:.2f}'
     .format(rf_use.score(X_test13, y_test1)))
```

    Accuracy of Random Forest classifier on training set: 0.70
    Accuracy of Random Forest classifier on test set: 0.70



```python
usefull=input()

```

    I used to love getting the Mediterranean quinoa bowl from Juicology ( former name). Was sad when they closed. Now that it reopened under a different name I was hoping their grain bowls would be the same. Got the Greek quinoa bowl and it's sadly more like a salad rather than a warm quinoa bowl. Everything tastes fresh but I wish they would bring back original quinoa bowl recipes and preparation. I'll be back to try their wraps and burgers. Please bring back the original Mediterranean bowl.



```python

use_pred_data= {'user_rev_count': [5], 'count_rev': [len(usefull.split(' '))],'rev_bus_rat_diff': [3.0]}
use_pred_df= pd.DataFrame(data= use_pred_data)
for useful in [usefull]: 
    rev_string = []
    cleaned = nltk.regexp_tokenize(str(useful), pattern)
    tokens = [i.lower() for i in cleaned]
    tokens_stopped = [w for w in tokens if not w in stop_words]
    meta_stemmed = [stemmer.stem(word) for word in tokens_stopped]
    string = ' '.join(meta_stemmed)
    rev_string.append(string)
    usett = tfidf_useful.transform(rev_string)
    usett_features_tfidf = pd.DataFrame(usett.toarray(), columns=tfidf_useful.get_feature_names())
    pred_use_df= use_pred_df.join(usett_features_tfidf)
    pred_use_df1= scaler.transform(pred_use_df)
    
```


```python
rf_use.predict_proba(pred_use_df1)
```




    array([[0.51783651, 0.48216349]])




```python
use_pred= rf_use.predict(pred_use_df)
```


```python
use_pred
```




    array([0])




```python
my_features[np.argsort(rf_use.feature_importances_)]
```




    Index(['leather', 'octob', 'odd', 'offend', 'offens', 'offic', 'og', 'ohio',
           'oil', 'oj',
           ...
           'two', 'well', 'servic', 'tabl', 'get', 'want', 'one', 'like',
           'user_rev_count', 'count_rev'],
          dtype='object', length=5003)




```python
my_features[np.argsort(rf_use.feature_importances_)][0][::-1]
```




    'rehtael'




```python
ypred_use= rf_use.predict(X_test13)
```


```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test1, ypred_use))
cm = confusion_matrix(y_test1, ypred_use)
plot_confusion_matrix(cm, [0.0, 1.0])
```

                  precision    recall  f1-score   support
    
               0       0.68      0.74      0.71      7945
               1       0.72      0.65      0.69      8055
    
       micro avg       0.70      0.70      0.70     16000
       macro avg       0.70      0.70      0.70     16000
    weighted avg       0.70      0.70      0.70     16000
    
    Confusion matrix, without normalization
    [[5898 2047]
     [2791 5264]]



![png](DataPreprocessing_files/DataPreprocessing_352_1.png)



```python

```


```python
#pickling random forest classifier for uselfulness
pickling_rf_use = open("rf_class_usef","wb")
pickle.dump(rf_use, pickling_rf_use)
pickling_rf_use.close()
```


```python
pickle_rf_use = open("rf_class_usef","rb")
rf_use = pickle.load(pickle_rf_use)

```


```python

```


```python

```


```python

```


```python
#Making the big prediction of rating and usefullness
```


```python
rev1=input()
use_pred_data= {'user_rev_count': [5], 'count_rev': [len(rev1.split(' '))],'rev_bus_rat_diff': [3.0]}
use_pred_df= pd.DataFrame(data= use_pred_data)
print('')

if len(rev1.split(' ')) < 10:
    print ('2 Stars')
    print('')
    print('Your Review WONT Be Helpful to others.')
    print('')
    print('If you wish to leave a better rating please describe your experience in more words')
else:
    for overview in [rev1]: 
        rev_string = []
        cleaned = nltk.regexp_tokenize(str(overview), pattern)
        tokens = [i.lower() for i in cleaned]
        tokens_stopped = [w for w in tokens if not w in stop_words]
        meta_stemmed = [stemmer.stem(word) for word in tokens_stopped]
        string = ' '.join(meta_stemmed)
        rev_string.append(string)
        tt = tfidf.transform(rev_string)
    # print(rev1)
        print ('')
        rev_pred = lin_reg_rating.predict(tt)

        for i in rev_pred:
            if i < 1.5:
                print ('1 Star')
            elif i < 2.5:
                print ('2 stars')
            elif i < 3.5:
                print ('3 stars')
            elif i < 4.5:
                print ('4 stars')
            else:
                print ('5 stars')
            print ("")
            
            
        usett_features_tfidf = pd.DataFrame(tt.toarray(), columns=tfidf_useful.get_feature_names())
        pred_use_df= use_pred_df.join(usett_features_tfidf)
        pred_use_df1= scaler.transform(pred_use_df)
        use_pred= rf_use.predict(pred_use_df1)
        use_proba= rf_use.predict_proba(pred_use_df1)
        print (use_pred)
        print (rf_use.predict_proba(pred_use_df1))
        for a,b in use_proba:
            if b > 0.46:
                print ("Your review WILL be Helpful to others! ")
            else:
                print ("Your review WONT be Helpful to others! ")

```

    Absolutely horrible service. They are far from the service Juiceology use to have (The previous place, which I believe had the same owners). At first I thought it was just a matter of time until they are situated and get the hang of it, but I realize they are just incompetent. I been going the past month, almost every day, since they open hoping they get it together, giving them the chance to adjust and recognize that I could be a regular, but things never got better, not one bit.   They are SLOW! The girls on the registrar take forever to get transactions done and bag your food. I had seen a few customers saying, "I'll just take it with out it", at the site of their struggle to put things in the paper bags. One of the girls was on her phone texting while she is trying to figure out the machine and giving me change back, making things even slower. Most of the people that work in the neighborhood are in a rush to get food and get back to work, there is no concern for peoples time. The cooks take for ever to prepare the food and the salads are just sad and bland, at least the ones I tried.  The same day that girl was on the phone, I happen to walk in during some sort of inspection, I was the only costumer at the moment. The inspector was looking through the ready to go food area/refrigerator shelves and had the staff take all the food down because there where rotten eggs and old sandwiches sitting there from, what the staff said, the day before. I had actually seen those same shell-less boiled eggs sitting there 2 days prior. Gross! Mean while the full staff is congregated around the inspector I am waiting for my food to be prepared. I'm looking at them like, "Is anyone going to take care of my order?". 30 min had past, I finally get my food, I storm out upset that it took so long to prepare an Acai Bowl and on top of it I was reeking of grease for hours after that.       The food isn't horrible and had potential to be good considering the options in the neighborhood. The only things I will miss from this place is the Avocado Wrap and Express Shake. Other than that this place needs a lot of work and a more alert staff.   Oh! and so much for it being "green" the vegan and vegetarian options are almost non-existent.
    
    
    1 Star
    
    [0]
    [[0.53103891 0.46896109]]
    Your review WILL be Helpful to others! 



```python

```




    array([[0.53103891, 0.46896109]])




```python

```
