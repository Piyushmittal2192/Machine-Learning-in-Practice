## Machine Learning Topics
### Supervised, Unsupervised & Reinforced Learning

-----------------------------------------------------------------------
## Concepts
### Data & Its Type

### Feature Engineering
### Feature Extraction
### Regression
### Classification
### Clustering
### Optimizations

-------------------------------------------------------

### Feature Engineering 
Machine Learning techniques expects data in a format respective to their algorithm. Therfore, we manipulate (engineer) the features, so that we can feed the input to the model and get output. Feature Engineering helps us to :

1. Prepare the features of the input data for machine learning algorithm.
2. Provide correct format of feature to get better results from the machine learning algorithm.


#### Dropping 
When we are cleaning the data set, first thought that comes to our mind is to look for missing values. If there are considerable missing values then we drop the rows or the columns. Usually, when we are dropping we consider a <b> Threshold </b> value, which sets a criteria i.e. if e.g. 70% of the data is not present then drop the records. Threshold can vary based on the dataset.   
```markdown
threshold = 0.7
#Dropping columns with missing value rate higher than threshold
data = data[data.columns[data.isnull().mean() < threshold]]  # default axis is set to 0 for .mean()

#Dropping rows with missing value rate higher than threshold
data = data.loc[data.isnull().mean(axis=1) < threshold]

```
#### Imputation
If the data does not have considerable amount of missing values for the observartions, then we look for <b> Imputation </b> techniques. Imputation means replacing the missing ones with a suitbale value. 
- ##### Pros : 
  - Size of the dataset remain same.
  - Information is not lost when compared to dropping records  
- ##### Cons : 
  - Improper imputation will pass wrong informaiton to model and result in bad predictions. 

1. <b>Numerical Imputation</b> <br>
When a numerical attribute has missing values, we impute them with numbers. There are different numerical imputation methods :
    - Replace with 0 : if there is only single value 1 and other is NA, then we can subsitute the NA with 0.
    - Replace with mean : If outliers are not present
    - Replace with median : If outliers are present then mean values is affected and shifted towards the outliers. Thus is is better to median. 
    - Forecast the missing values using other columns : Use regression models to predict the missing values.


```markdown
#Filling all missing values with 0
data = data.fillna(0)
#Filling missing values with medians of the columns
data = data.fillna(data.median())
```

2. <b>Categorical Imputation</b> <br>
When a categorical attribute has missing values, we impute them with maximum occuring element. But if the values are uniformly distributed i.e. no dominant category, then we can subtitute with <b>"Others"</b>

```markdown
#Max fill function for categorical columns
data['column_name'].fillna(data['column_name'].value_counts().idxmax(), inplace=True)
```
-----------------------------------------------------------
For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

#### Outlier Removal
Outliers are points/obervations that are significantly different from other observations in data. Outliers can occur due to 
error in recording the observation or the occurance of event that lead to deviation in the observation.
<b> Measures to Identify Outliers </b>
  - Standard Deviation
  - Percentiles
    - <b>Standard Deviation</b> : Any observation that is at a distance greater that x * standard deviation, where x is generally 3 or 4, is an outlier.
    ```markdown
    #Dropping the outlier rows with standard deviation
    factor = 3
    upper_lim = data['column'].mean () + data['column'].std () * factor
    lower_lim = data['column'].mean () - data['column'].std () * factor

    data = data[(data['column'] < upper_lim) & (data['column'] > lower_lim)]
1
    ```
    - <b>Percentiles</b> : 
    ```markdown
    #Dropping the outlier rows with Percentiles
    upper_lim = data['column'].quantile(.95)
    lower_lim = data['column'].quantile(.05)

    data = data[(data['column'] < upper_lim) & (data['column'] > lower_lim)]
    ```
  
  - <b> Capping</b> : Instead of drop we can cap the observations 
    ```markdown
      #Capping the outlier rows with Percentiles
      upper_lim = data['column'].quantile(.95)
      lower_lim = data['column'].quantile(.05)
      data.loc[(df[column] > upper_lim),column] = upper_lim
      data.loc[(df[column] < lower_lim),column] = lower_lim
    ```
#### Binning
Data binning is a data pre-processing technique used to reduce the effects of minor observation errors. The original data values which fall into a given small interval, a bin, are replaced by a value representative of that interval, often the central value. It is a form
of quantization. The intent of binning is to make the model more robust and prevent overfitting. The con of binning is that, it leads to loss of information. Binning labels to <b>"others"</b> makes sense when there are 100,000 observations and among them there are labels that have count less than 100. 

```markdown
#Numerical Binning Example
data['bin'] = pd.cut(data['value'], bins=[0,30,70,100], labels=["Low", "Mid", "High"])
   value   bin
0      2   Low
1     45   Mid
2      7   Low
3     85  High
4     28   Low
#Categorical Binning Example
     Country
0      Spain
1      Chile
2  Australia
3      Italy
4     Brazil
conditions = [
    data['Country'].str.contains('Spain'),
    data['Country'].str.contains('Italy'),
    data['Country'].str.contains('Chile'),
    data['Country'].str.contains('Brazil')]

choices = ['Europe', 'Europe', 'South America', 'South America']

data['Continent'] = np.select(conditions, choices, default='Other')
     Country      Continent
0      Spain         Europe
1      Chile  South America
2  Australia          Other
3      Italy         Europe
4     Brazil  South America
```

#### Log Transformation 
Manipulates the given observation using log function. Log transformation scales down the magnitude of the values. e.g. if we want our model to learn that the experience gain from age 15 to 20 is more when compared to age 75 to 80, then simply providing age as it is to model might not be a good idea. As difference between 20-15 and 65-70 is 5. whereas we know that the expereince gain is more from age 15 to 20 than 65 to 70. So, if we perform log over values then we can see that log 20-log15 = 0.124 and log70 - log 65 = 0.03  

```markdown
#Log Transform Example
data = pd.DataFrame({'value':[2,45, -23, 85, 28, 2, 35, -12]})
data['log+1'] = (data['value']+1).transform(np.log)
#Negative Values Handling
#Note that the values are different
data['log'] = (data['value']-data['value'].min()+1) .transform(np.log)
```
#### One Hot Encoding
Depending on the Machine Learning model, we need to feed the data in a specific format. Some alogorithms need the data to be in numerical formal, so if we have categotical observations, then we need to convert them in numerical format. there are certain ways, one of them is One Hot Encoding. We use this when there is no order among the labels of the feature. e.g. if we have a color feature having  Red, gree, blue labels. Then we can't assign numerical values as 0,1,2 respecitvely to the colors, cuz by doing so we are saying Blue is 2 units more than Red and Green is 1 unit more than blue, which is not true. So one hot encode the feature such that we create 3 columns w.r.t Red, Green, Blue color. 
```markdown
df = pd.DataFrame({'color':['r','b','g','r','r','g'], 'count':[1, 2, 3, 4, 5, 6]})
df
    color	count
0	r	1
1	b	2
2	g	3
3	r	4
4	r	5
5	g	6
encoded_col = pd.get_dummies(df['color'])
df = df.join(encoded_col).drop('color', axis = 1)
df
    count	b	g	r
0	1	0	0	1
1	2	1	0	0
2	3	0	1	0
3	4	0	0	1
4	5	0	0	1 
5	6	0	1	0
```

#### Grouping
Tidy datasets -> every instance is represented by row in training data and each column represents the feature of the instance.
Transactional Datasets -> an instance has multiple rows in dataset. We need to group by instances and each instance will be represented by a row.
  - <b> Categorical Column Grouping </b> <br> :
      - Highest Frequency: Select the label with highest frequency.
            ```markdown
                data.groupby('id').agg(lambda x: x.value_counts().index[0])
            ```
      - Pivot Table : 
            ```markdown
                #Pivot table Pandas Example
                 data.pivot_table(index='column_to_group', columns='column_to_encode', values='aggregation_column', aggfunc=np.sum, fill_value = 0)
             ```
      - Group By one-hot-encoding : Apply group by function after one-hot encoding. Usig this we can preserve all the data lost via first option.
          
#### Feature Split
Split the column data to usefull information
```markdown
#String extraction example
data.title.head()
0                      Toy Story (1995)
1                        Jumanji (1995)
2               Grumpier Old Men (1995)
3              Waiting to Exhale (1995)
4    Father of the Bride Part II (1995)
data.title.str.split("(", n=1, expand=True)[1].str.split(")", n=1, expand=True)[0]
0    1995
1    1995
2    1995
3    1995
4    1995
```

        


