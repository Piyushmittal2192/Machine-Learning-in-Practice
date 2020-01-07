## Machine Learning Topics
### Supervised, Unsupervised & Reinforced Learning

-----------------------------------------------------------------------

You can use the [editor on GitHub](https://github.com/Piyushmittal2192/Machine-Learning-in-Practice/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Concepts

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown


# Regression
## Header 2
# Classification
## Header 2
# Clustrtering
### Header 3

# Optimizations

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```
-------------------------------------------------------

### Feature Engineering 
Machine Learning techniques expects data in a format respective to their algorithm. So we manipulate (engineer) the features, so that we can feed the input to the model and get output. So we are doing two things by performing feature engineering :

1. Prepare the features of the input data for machine learning algorithm.
2. By providing correct format of feature, we may get better results from the machine learning algorithm.


##### Imputation 

```markdown
threshold = 0.7
#Dropping columns with missing value rate higher than threshold
data = data[data.columns[data.isnull().mean() < threshold]]

#Dropping rows with missing value rate higher than threshold
data = data.loc[data.isnull().mean(axis=1) < threshold]

```
-----------------------------------------------------------
For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Piyushmittal2192/Machine-Learning-in-Practice/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
