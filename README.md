## Machine Learning Topics
### Supervised, Unsupervised & Reinforced Learning

-----------------------------------------------------------------------
### Concepts


```markdown


# Regression
# Classification
# Clustrtering
# Optimizations

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

3. <b>Outliers</b> <br>
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
    ```
    - <b>Percentiles</b> : 
    ```markdown
    #Dropping the outlier rows with Percentiles
    upper_lim = data['column'].quantile(.95)
    lower_lim = data['column'].quantile(.05)

    data = data[(data['column'] < upper_lim) & (data['column'] > lower_lim)]
    ```
  
  <b> Capping</b> : Instead of drop we can cap the observations 
  ```markdown
    #Capping the outlier rows with Percentiles
    upper_lim = data['column'].quantile(.95)
    lower_lim = data['column'].quantile(.05)
    data.loc[(df[column] > upper_lim),column] = upper_lim
    data.loc[(df[column] < lower_lim),column] = lower_lim
    
    
### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Piyushmittal2192/Machine-Learning-in-Practice/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
