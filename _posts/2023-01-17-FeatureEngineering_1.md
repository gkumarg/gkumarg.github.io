# Feature Engineering Notes 1/17/2023

There are times when we need to create a few additional features from categorical data. Below is an example of a dataframe that I was working with:

|    id | gender   |   age |   hypertension |   heart_disease | ever_married   | work_type     | Residence_type   |   avg_glucose_level |   bmi | smoking_status   |   stroke |
|------:|:---------|------:|---------------:|----------------:|:---------------|:--------------|:-----------------|--------------------:|------:|:-----------------|---------:|
| 13244 | Female   |    75 |              0 |               0 | Yes            | Private       | Urban            |              100.29 |  30.5 | never smoked     |        0 |
| 10109 | Male     |    80 |              0 |               0 | Yes            | Self-employed | Rural            |               76.12 |  20.3 | Unknown          |        1 |
|  3111 | Female   |    14 |              0 |               0 | No             | Private       | Urban            |               72.18 |  31.5 | Unknown          |        0 |
| 11751 | Female   |    35 |              0 |               0 | Yes            | Self-employed | Urban            |               89.88 |  28.8 | never smoked     |        0 |
|  8041 | Female   |    49 |              0 |               0 | Yes            | Private       | Urban            |              102.97 |  25.5 | smokes           |        0 |

As we can see, the columns gender, ever_married, work_type, Residence_type and smoking_status are categorical.

1. Value greater than threshold
Let's say we want to create a feature that takes the bmi values (continuous feature). This can be done as below:

    `df['Obese'] = np.where(df['bmi']>25.0, 1, 0)`

    It was quick using the np.where function and passing it a condition to check. If the condition is true, we assign it 1 and if it is not met, we assign it 0.

2. Multiple conditions
Let's say we want to create a feature for male or female with hypertension as another feature. This can be done as below:

    `df['MaleXhypertension']= np.where(((df['gender']=='Male') & (df['hypertension']==1)),1,0)`

    `df['FemaleXhypertension']= np.where(((df['gender']=='Female') & (df['hypertension']==1)),1,0)`
    
In this case, it is checking for the condition that the gender is 'Male' and also that the indicator is 1 for hypertension. Note the use of & to combine the two conditions.

3. Mixing greater than threshold and indicator flag
This is another extension of the first two. We can check for values exceeding threshold and also check a flag.

    `df['OldXObese'] = np.where(((df['age']>45) & (df['Obese']==1)),1,0)`

4. Creating a continuous feature from existing continuous features
This one is very easy with pandas:

    `df['AgeRatioBMI'] = df['age'] / df['bmi']`

    `df['AgeXBMI'] = df['age'] * df['bmi']`


All the above code can be combined into a simple function as below:

```
def make_additional_features(df):
    df['Obese'] = np.where(df['bmi']>25.0, 1, 0)
    df['MaleXhypertension']= np.where(((df['gender']=='Male') & (df['hypertension']==1)),1,0)
    df['FemaleXhypertension']= np.where(((df['gender']=='Female') & (df['hypertension']==1)),1,0)
    df['OldXObese'] = np.where(((df['age']>45) & (df['Obese']==1)),1,0)
    df['AgeRatioBMI'] = df['age'] / df['bmi']
    df['AgeXBMI'] = df['age'] * df['bmi']

    return df


df = make_additional_features(df)
```
Checking the results that we get for some of the samples, we can see  

|    id | gender   |   age |   hypertension |   heart_disease | ever_married   | work_type     | Residence_type   |   avg_glucose_level |   bmi | smoking_status   |   stroke |   Obese |   MaleXhypertension |   FemaleXhypertension |   OldXObese |   AgeRatioBMI |   AgeXBMI |
|------:|:---------|------:|---------------:|----------------:|:---------------|:--------------|:-----------------|--------------------:|------:|:-----------------|---------:|--------:|--------------------:|----------------------:|------------:|--------------:|----------:|
|  2858 | Female   |    79 |              0 |               0 | Yes            | Self-employed | Urban            |               70.58 |  25.6 | Unknown          |        0 |       1 |                   0 |                     0 |           1 |      3.08594  |    2022.4 |
|  7338 | Female   |    82 |              0 |               0 | Yes            | Self-employed | Urban            |               80.43 |  30.3 | smokes           |        0 |       1 |                   0 |                     0 |           1 |      2.70627  |    2484.6 |
|  9404 | Female   |    19 |              0 |               0 | No             | Private       | Rural            |              110.72 |  25.4 | smokes           |        0 |       1 |                   0 |                     0 |           0 |      0.748031 |     482.6 |
| 13868 | Female   |    26 |              0 |               0 | No             | Private       | Urban            |              112.54 |  33.1 | Unknown          |        0 |       1 |                   0 |                     0 |           0 |      0.785498 |     860.6 |
|  2139 | Male     |    67 |              0 |               0 | Yes            | Self-employed | Urban            |               69.61 |  27.3 | formerly smoked  |        0 |       1 |                   0 |                     0 |           1 |      2.45421  |    1829.1 |
|  5439 | Female   |    81 |              0 |               0 | Yes            | Private     | Rural            |               78.16 |  29.6 | formerly smoked  |        1 |       1 |                   0 |                     0 |           1 |      2.73649  |    2397.6 |
|  5323 | Female   |    49 |              0 |               0 | Yes            | Private     | Rural            |               85.33 |  25.5 | smokes           |        0 |       1 |                   0 |                     0 |           1 |      1.92157  |    1249.5 |
| 10993 | Male     |     8 |              0 |               0 | No             | children    | Urban            |               89.44 |  18.4 | Unknown          |        0 |       0 |                   0 |                     0 |           0 |      0.434783 |     147.2 |
| 14404 | Female   |    59 |              0 |               0 | No             | Private     | Urban            |               96.26 |  45.7 | never smoked     |        1 |       1 |                   0 |                     0 |           1 |      1.29103  |    2696.3 |
|   884 | Female   |    24 |              0 |               0 | No             | Private       | Rural            |               70.01 |  27.9 | never smoked     |        0 |       1 |                   0 |                     0 |           0 |      0.860215 |     669.6 |
|  3740 | Female   |    63 |              1 |               0 | Yes            | Self-employed | Rural            |               95.06 |  34.3 | never smoked     |        0 |       1 |                   0 |                     1 |           1 |      1.83673  |    2160.9 |

These types of features can help with improving our machine learning models.

#featureengineering  #machinelearning

Gopakumar Gopinathan