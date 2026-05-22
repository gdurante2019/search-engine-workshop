# Search Engine Workshop (prep for LLM-Zoomcamp)

Reference notebook:  https://github.com/alexeygrigorev/build-your-own-search-engine \
Video: https://www.youtube.com/watch?v=nMrGK5QgPVE

## Overview
In this workshop, we built and compared search algorithms with different tools, including CountVectorizer, TF-IDF, and BERT, using the FAQ documents from three DataTalks.Club Zoomcamps (Data Engineering Zoomcamp, ML Zoomcamp, and MLOps Zoomcamp). We learned about basic text search, vectorization, semantic search, and more. Having experience with these fundamental concepts and tools will be valuable for future search engine builds using more advanced tools.

This notebook provides step-by-step code instructions and explanations of what the code is doing, per instructor Alexey Gregorev. Taking these notes helped remind me of the key concepts; hopefully they will be useful to anyone who is learning about--or seeking a refresher on--search engine fundamentals.

### Prepare Python environment

I prepared an environment on my local machine. The libraries installed include ```requests```, ```pandas```, ```scikit-learn```, and ```jupyter```. The instructor used GitHub Codespaces, which is another great option when you don't want to take the time to set up another environment on your machine. We used Jupyter Notebook.

### Text search:  FAQ docs ingestion and exploration

To start, we download the course FAQs and creat a dataframe to explore the data. We will then vectorize the documents and employ different tools to perform searches.




```python
import pandas as pd
import numpy as np
```


```python
# requests is a library for obtaining documents from URLs
import requests 

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)
```


```python
from sklearn import set_config
set_config(display='text')
```

Let's take a look at an individual item within the documents corpus.


```python
documents[2]
```




    {'text': "Yes, even if you don't register, you're still eligible to submit the homeworks.\nBe aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.",
     'section': 'General course-related questions',
     'question': 'Course - Can I still join the course after the start date?',
     'course': 'data-engineering-zoomcamp'}



Now, let's create a pandas dataframe to look at the structure of the information.


```python
pd.DataFrame(documents)
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
      <th>text</th>
      <th>section</th>
      <th>question</th>
      <th>course</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The purpose of this document is to capture fre...</td>
      <td>General course-related questions</td>
      <td>Course - When will the course start?</td>
      <td>data-engineering-zoomcamp</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GitHub - DataTalksClub data-engineering-zoomca...</td>
      <td>General course-related questions</td>
      <td>Course - What are the prerequisites for this c...</td>
      <td>data-engineering-zoomcamp</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Yes, even if you don't register, you're still ...</td>
      <td>General course-related questions</td>
      <td>Course - Can I still join the course after the...</td>
      <td>data-engineering-zoomcamp</td>
    </tr>
    <tr>
      <th>3</th>
      <td>You don't need it. You're accepted. You can al...</td>
      <td>General course-related questions</td>
      <td>Course - I have registered for the Data Engine...</td>
      <td>data-engineering-zoomcamp</td>
    </tr>
    <tr>
      <th>4</th>
      <td>You can start by installing and setting up all...</td>
      <td>General course-related questions</td>
      <td>Course - What can I do before the course starts?</td>
      <td>data-engineering-zoomcamp</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>943</th>
      <td>Problem description\nThis is the step in the c...</td>
      <td>Module 6: Best practices</td>
      <td>Github actions: Permission denied error when e...</td>
      <td>mlops-zoomcamp</td>
    </tr>
    <tr>
      <th>944</th>
      <td>Problem description\nWhen a docker-compose fil...</td>
      <td>Module 6: Best practices</td>
      <td>Managing Multiple Docker Containers with docke...</td>
      <td>mlops-zoomcamp</td>
    </tr>
    <tr>
      <th>945</th>
      <td>Problem description\nIf you are having problem...</td>
      <td>Module 6: Best practices</td>
      <td>AWS regions need to match docker-compose</td>
      <td>mlops-zoomcamp</td>
    </tr>
    <tr>
      <th>946</th>
      <td>Problem description\nPre-commit command was fa...</td>
      <td>Module 6: Best practices</td>
      <td>Isort Pre-commit</td>
      <td>mlops-zoomcamp</td>
    </tr>
    <tr>
      <th>947</th>
      <td>Problem description\nInfrastructure created in...</td>
      <td>Module 6: Best practices</td>
      <td>How to destroy infrastructure created via GitH...</td>
      <td>mlops-zoomcamp</td>
    </tr>
  </tbody>
</table>
<p>948 rows × 4 columns</p>
</div>



Let's reorganize the columns to make the information more readable.


```python
df = pd.DataFrame(documents, columns=['course', 'section', 'question', 'text'])
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
      <th>course</th>
      <th>section</th>
      <th>question</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - When will the course start?</td>
      <td>The purpose of this document is to capture fre...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - What are the prerequisites for this c...</td>
      <td>GitHub - DataTalksClub data-engineering-zoomca...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - Can I still join the course after the...</td>
      <td>Yes, even if you don't register, you're still ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - I have registered for the Data Engine...</td>
      <td>You don't need it. You're accepted. You can al...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - What can I do before the course starts?</td>
      <td>You can start by installing and setting up all...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
# Note that the questions at the end of the document are for a different course (mlops-zoomcamp) than those at the beginning (data-engineering-zoomcamp) 
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
      <th>course</th>
      <th>section</th>
      <th>question</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>943</th>
      <td>mlops-zoomcamp</td>
      <td>Module 6: Best practices</td>
      <td>Github actions: Permission denied error when e...</td>
      <td>Problem description\nThis is the step in the c...</td>
    </tr>
    <tr>
      <th>944</th>
      <td>mlops-zoomcamp</td>
      <td>Module 6: Best practices</td>
      <td>Managing Multiple Docker Containers with docke...</td>
      <td>Problem description\nWhen a docker-compose fil...</td>
    </tr>
    <tr>
      <th>945</th>
      <td>mlops-zoomcamp</td>
      <td>Module 6: Best practices</td>
      <td>AWS regions need to match docker-compose</td>
      <td>Problem description\nIf you are having problem...</td>
    </tr>
    <tr>
      <th>946</th>
      <td>mlops-zoomcamp</td>
      <td>Module 6: Best practices</td>
      <td>Isort Pre-commit</td>
      <td>Problem description\nPre-commit command was fa...</td>
    </tr>
    <tr>
      <th>947</th>
      <td>mlops-zoomcamp</td>
      <td>Module 6: Best practices</td>
      <td>How to destroy infrastructure created via GitH...</td>
      <td>Problem description\nInfrastructure created in...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# To filter by any column value (keyword search), use the syntax below:
df[df.course == 'data-engineering-zoomcamp']
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
      <th>course</th>
      <th>section</th>
      <th>question</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - When will the course start?</td>
      <td>The purpose of this document is to capture fre...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - What are the prerequisites for this c...</td>
      <td>GitHub - DataTalksClub data-engineering-zoomca...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - Can I still join the course after the...</td>
      <td>Yes, even if you don't register, you're still ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - I have registered for the Data Engine...</td>
      <td>You don't need it. You're accepted. You can al...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - What can I do before the course starts?</td>
      <td>You can start by installing and setting up all...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>430</th>
      <td>data-engineering-zoomcamp</td>
      <td>Workshop 2 - RisingWave</td>
      <td>Unable to Open Dashboard as xdg-open doesn’t o...</td>
      <td>Refer to the solution given in the first solut...</td>
    </tr>
    <tr>
      <th>431</th>
      <td>data-engineering-zoomcamp</td>
      <td>Workshop 2 - RisingWave</td>
      <td>Resolving Python Interpreter Path Inconsistenc...</td>
      <td>Example Error:\nWhen attempting to execute a P...</td>
    </tr>
    <tr>
      <th>432</th>
      <td>data-engineering-zoomcamp</td>
      <td>Workshop 2 - RisingWave</td>
      <td>How does windowing work in Sql?</td>
      <td>Ans : Windowing in streaming SQL involves defi...</td>
    </tr>
    <tr>
      <th>433</th>
      <td>data-engineering-zoomcamp</td>
      <td>Triggers in Mage via CLI</td>
      <td>Encountering the error "ModuleNotFoundError: N...</td>
      <td>Python 3.12.1, is not compatible with kafka-py...</td>
    </tr>
    <tr>
      <th>434</th>
      <td>data-engineering-zoomcamp</td>
      <td>Triggers in Mage via CLI</td>
      <td>Basic Commands</td>
      <td>Docker Commands\n# Create a Docker Image from ...</td>
    </tr>
  </tbody>
</table>
<p>435 rows × 4 columns</p>
</div>



### Vector spaces 

To more efficiently process text documents, we use vector spaces.  This involves turning documents into vectors and encoding the text into a matrix.  This matrix will have a column for each word/term and rows representing the contents of documents.  For each word in a document (row), the number of times the word appears will be marked in the column corresponding to that word.  This is called a term-document matrix.  Since most of the columns will be zeros for any given row, the matrix is referred to as a sparse matrix. 

We'll first use CountVectorizer and create a simple text representation referred to as "bag-of-words".  It disregards grammar and word order but keeps track of the number of times a word appears in a document.  

To summarize:
- Turn the docs into vectors
- This creates the term-document matrix:
    - Rows: documents
    - Columns: words/tokens
- Key features of bag of words:
    - word order is lost
    - sparse matrix


```python
from sklearn.feature_extraction.text import CountVectorizer
```


```python
cv = CountVectorizer()
```


```python
cv.fit(df.text)
```




    CountVectorizer()



Let's check the terms (features) included in the CountVectorizer model using ```get_feature_names_out()```:


```python
cv.get_feature_names_out()
```




    array(['00', '00000000e', '0002', ..., '要了解键盘快捷键', '要启用屏幕阅读器支持', '请按ctrl'],
          shape=(6711,), dtype=object)



Notice that we have a lot of noise here.  Each term/word/token is represented as a feature.  In our case, this leads to us having 6711 features!  

To help improve the quality of results and speed of execution, we can limit the terms that are included to only those found in five "documents" or more. (In this case, a "document" is a single FAQ.)


```python
cv = CountVectorizer(min_df=5)
```


```python
cv.fit(df.text)
```




    CountVectorizer(min_df=5)




```python
cv.get_feature_names_out()
```




    array(['01', '02', '03', ..., 'youtube', 'zip', 'zoomcamp'],
          shape=(1524,), dtype=object)



Now we have a more manageable number of features (1524 vs. 6711).

Now, let's run a quick check on the model with a small subset of documents (FAQs)


```python
doc_examples = [
    "Course starts on 15th Jan 2024",
    "Prerequisites listed on GitHub",
    "Submit homeworks after start date",
    "Registration not required for participation",
    "Setup Google Cloud and Python before course"
]
```


```python
cv = CountVectorizer()
```


```python
cv.fit(doc_examples)
```




    CountVectorizer()




```python
cv.get_feature_names_out() 
```




    array(['15th', '2024', 'after', 'and', 'before', 'cloud', 'course',
           'date', 'for', 'github', 'google', 'homeworks', 'jan', 'listed',
           'not', 'on', 'participation', 'prerequisites', 'python',
           'registration', 'required', 'setup', 'start', 'starts', 'submit'],
          dtype=object)



Now, we transform these documents into a matrix.


```python
X = cv.transform(doc_examples)
```


```python
X.todense()

# allows us to see inside of this matrix.  
```




    matrix([[1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
             0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0,
             0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,
             0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
             1, 0, 0, 0]])



Looking into this matrix, we see it's a list of 5 lists.  The first list represents the words in the first document, and each item in the list corresponds to the feature list obtained with ```cv.get_feature_names_out()```.  We can see that the terms '15th', '2024', 'course', 'jan', and 'starts' are present in the first list (document).  

To make it easier to view the matrix, we put it in a dataframe and include the terms (feature names) in the dataframe.


```python
pd.DataFrame(X.todense(), columns=cv.get_feature_names_out())
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
      <th>15th</th>
      <th>2024</th>
      <th>after</th>
      <th>and</th>
      <th>before</th>
      <th>cloud</th>
      <th>course</th>
      <th>date</th>
      <th>for</th>
      <th>github</th>
      <th>...</th>
      <th>on</th>
      <th>participation</th>
      <th>prerequisites</th>
      <th>python</th>
      <th>registration</th>
      <th>required</th>
      <th>setup</th>
      <th>start</th>
      <th>starts</th>
      <th>submit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>
</div>



We can transpose the matrix to make it easier to see the terms (scrolling down vs. across).


```python
pd.DataFrame(X.todense(), columns=cv.get_feature_names_out()).T
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15th</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2024</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>after</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>and</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>before</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>cloud</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>course</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>date</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>for</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>github</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>google</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>homeworks</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>jan</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>listed</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>not</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>on</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>participation</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>prerequisites</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>python</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>registration</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>required</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>setup</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>start</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>starts</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>submit</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



We see words here that aren't really relevant to understanding the relevance of the document--words like 'the', 'and', 'or', 'an', and so on.  To remove these words from the analysis, we use the argument ```stop_words```.


```python
cv = CountVectorizer(stop_words='english')
cv.fit(doc_examples)
```




    CountVectorizer(stop_words='english')




```python
cv.get_feature_names_out()
```




    array(['15th', '2024', 'cloud', 'course', 'date', 'github', 'google',
           'homeworks', 'jan', 'listed', 'participation', 'prerequisites',
           'python', 'registration', 'required', 'setup', 'start', 'starts',
           'submit'], dtype=object)




```python
X = cv.transform(doc_examples)
```


```python
pd.DataFrame(X.todense(), columns=cv.get_feature_names_out()).T
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15th</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2024</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>cloud</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>course</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>date</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>github</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>google</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>homeworks</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>jan</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>listed</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>participation</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>prerequisites</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>python</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>registration</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>required</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>setup</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>start</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>starts</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>submit</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Now we have removed words that we're not interested in.  As a reminder, this is a bag-of-words representation.  Word order is lost, we use sparse matrices.

### TD-IDF Vectorizer

The bag-of-words approach above shows us which terms are included in each document, but doesn't give us information about the relative importance of each term in the overall corpus.  For that, we need a tool that vectorizes the documents and gives a measure of the relative importance of each term.  

TF-IDF is a tool in the scikit-learn library that performs this function.The TD-IDF (Term Frequency-Inverse Document Frequency) Vectorizer is a statistical measure that evaluates the importance of a term in a document in a corpus.  It gives more importance to less frequent terms, on the assumption that less frequent terms (e.g., yaml) are probably more relevant than more frequent terms (e.g., YouTube, Zoomcamp).  Here we run through the same process as we did with CountVectorizer.  We are fitting the model on the entire documents corpus. 


```python
from sklearn.feature_extraction.text import TfidfVectorizer

cv = TfidfVectorizer(stop_words='english', min_df=5)
X = cv.fit_transform(df.text)  #Fit the whole corpus 

names = cv.get_feature_names_out()

df_docs = pd.DataFrame(X.toarray(), columns=names)    #Feature names become column headers
df_docs.round(2)
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
      <th>01</th>
      <th>02</th>
      <th>03</th>
      <th>04</th>
      <th>05</th>
      <th>06</th>
      <th>09</th>
      <th>10</th>
      <th>100</th>
      <th>11</th>
      <th>...</th>
      <th>y_val</th>
      <th>yaml</th>
      <th>year</th>
      <th>yellow</th>
      <th>yellow_tripdata_2021</th>
      <th>yes</th>
      <th>yml</th>
      <th>youtube</th>
      <th>zip</th>
      <th>zoomcamp</th>
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
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
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
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.43</td>
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
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.28</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
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
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
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
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
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
      <th>943</th>
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
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.11</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>944</th>
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
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>945</th>
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
      <td>0.17</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>946</th>
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
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>947</th>
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
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.15</td>
    </tr>
  </tbody>
</table>
<p>948 rows × 1333 columns</p>
</div>




```python
X
```




    <Compressed Sparse Row sparse matrix of dtype 'float64'
    	with 23808 stored elements and shape (948, 1333)>



Let's see what the matrix looks like for a single query:


```python
query = "I just discovered this course; is it too late to join?"

q = cv.transform([query])
q.toarray()
```




    array([[0., 0., 0., ..., 0., 0., 0.]], shape=(1, 1333))




```python
# Source - https://stackoverflow.com/a/78793583
np.set_printoptions(legacy='1.25')

query_dict = dict(zip(names, q.toarray()[0]))
len(query_dict)
```




    1333




```python
from itertools import islice
first_40 = dict(islice(query_dict.items(), 40))
print(first_40)
```

    {'01': 0.0, '02': 0.0, '03': 0.0, '04': 0.0, '05': 0.0, '06': 0.0, '09': 0.0, '10': 0.0, '100': 0.0, '11': 0.0, '12': 0.0, '127': 0.0, '13': 0.0, '14': 0.0, '15': 0.0, '16': 0.0, '17': 0.0, '19': 0.0, '1st': 0.0, '20': 0.0, '2019': 0.0, '2020': 0.0, '2021': 0.0, '2022': 0.0, '2023': 0.0, '2024': 0.0, '21': 0.0, '22': 0.0, '24': 0.0, '25': 0.0, '2pacx': 0.0, '30': 0.0, '35': 0.0, '403': 0.0, '42': 0.0, '50': 0.0, '5000': 0.0, '5431': 0.0, '5432': 0.0, '60': 0.0}



```python
last_40 = dict(islice(query_dict.items(), len(query_dict)-40, len(query_dict)))
print(last_40)
```

    {'ways': 0.0, 'web': 0.0, 'website': 0.0, 'week': 0.0, 'weights': 0.0, 'went': 0.0, 'wget': 0.0, 'whl': 0.0, 'window': 0.0, 'windows': 0.0, 'winpty': 0.0, 'won': 0.0, 'work': 0.0, 'worked': 0.0, 'worker': 0.0, 'working': 0.0, 'works': 0.0, 'workshop': 0.0, 'worry': 0.0, 'write': 0.0, 'writing': 0.0, 'written': 0.0, 'wrong': 0.0, 'wsl': 0.0, 'wsl2': 0.0, 'www': 0.0, 'x_train': 0.0, 'xgboost': 0.0, 'y_pred': 0.0, 'y_train': 0.0, 'y_val': 0.0, 'yaml': 0.0, 'year': 0.0, 'yellow': 0.0, 'yellow_tripdata_2021': 0.0, 'yes': 0.0, 'yml': 0.0, 'youtube': 0.0, 'zip': 0.0, 'zoomcamp': 0.0}


We see most of the term values are 0.0, since those words don't appear in the query.  The word 'course' does appear in the query, so it has a value ```'course': 0.38148200594064524```.

So what we're looking for is where the document and the query contain the same words, then the document (FAQ) is relevant to the query.  Because each of the terms is weighted (TF-IDF weight), we can now see which documents are more important to this query.  Now we can rank the documents according to this criteria.  To do this, we multiply the weight of the term in the query with the weight of the term in a document, and sum across all of the matching terms (dot product in linear algebra).  This gives us a measure of similarity.  If we do this for all the documents, we can score all the documents and then rank by the relevant score.


```python
# Computing similarity between query and document (cosine similarity)
np.set_printoptions(threshold=40, edgeitems=10)

X.dot(q.T).todense()
```




    matrix([[0.48049682],
            [0.        ],
            [0.        ],
            [0.2083882 ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.17557272],
            [0.        ],
            [0.        ],
            ...,
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ]])




```python
# Can now compute similarity for all documents
from sklearn.metrics.pairwise import cosine_similarity
```


```python
cosine_similarity(X, q) 
#Gives us the same result as the measure of similarity above.
```




    array([[0.48049682],
           [0.        ],
           [0.        ],
           [0.2083882 ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.17557272],
           [0.        ],
           [0.        ],
           ...,
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ]])




```python
#turn into a vector using flatten
score = cosine_similarity(X, q).flatten()
```


```python
np.set_printoptions(threshold=40, edgeitems=20)

score
```




    array([0.48049682, 0.        , 0.        , 0.2083882 , 0.        ,
           0.        , 0.        , 0.17557272, 0.        , 0.        ,
           0.        , 0.15870689, 0.        , 0.        , 0.        ,
           0.09680922, 0.        , 0.        , 0.07529201, 0.        , ...,
           0.        , 0.        , 0.        , 0.        , 0.03662924,
           0.0451592 , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ])



To find out the ranking of the documents by importance, we can use numpy's ```argsort()```.  This returns the *index* of the documents, from most importance to least.


```python
# This gives us the the indices of the documents, rather than the scores themselves, and sorts them from lowest to highest.  
# Interested in the 5 highest similarity scores, so the last 5 indices.
np.set_printoptions(threshold=40, edgeitems=50)

np.argsort(score)
```




    array([473, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613,
           614, 615, 616, 617, 618, 620, 621, 622, 624, 625, 627, 601, 628,
           600, 596, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580,
           581, 582, 583, 584, 585, 586, 589, 590, 591, 592, 594, ..., 806,
           328,  18, 327, 142, 826, 226, 100, 741, 411, 193,  15, 915, 342,
            23, 511, 396, 454, 768, 923, 395, 450, 435, 148, 814, 758,  28,
           588, 913, 451, 455, 565, 654,  11, 445, 113,   7, 457, 657, 436,
             3, 287, 764, 452,  27,  22, 448, 449, 440,   0])




```python
np.argsort(score)[-5:]
```




    array([ 22, 448, 449, 440,   0])




```python
df.iloc[449].text
#One related response to our query
```




    'Yes, you can. You won’t be able to submit some of the homeworks, but you can still take part in the course.\nIn order to get a certificate, you need to submit 2 out of 3 course projects and review 3 peers’ Projects by the deadline. It means that if you join the course at the end of November and manage to work on two projects, you will still be eligible for a certificate.'




```python
#We can do this for all of our fields
fields = ['section', 'question', 'text']
```


```python
#Implement search across all fields
matrices = {}  #These are our X's 
vectorizers = {}  #Vectorizers are the cv's 

for f in fields:
    cv = TfidfVectorizer(stop_words='english', min_df=5)
    X = cv.fit_transform(df[f])
    matrices[f] = X
    vectorizers[f] = cv
```


```python
matrices
```




    {'section': <Compressed Sparse Row sparse matrix of dtype 'float64'
     	with 3090 stored elements and shape (948, 66)>,
     'question': <Compressed Sparse Row sparse matrix of dtype 'float64'
     	with 3431 stored elements and shape (948, 291)>,
     'text': <Compressed Sparse Row sparse matrix of dtype 'float64'
     	with 23808 stored elements and shape (948, 1333)>}




```python
vectorizers
```




    {'section': TfidfVectorizer(min_df=5, stop_words='english'),
     'question': TfidfVectorizer(min_df=5, stop_words='english'),
     'text': TfidfVectorizer(min_df=5, stop_words='english')}



First, we'll create an array with all zeros, then loop over all the fields. For each field, will compute the similarity, then sum similarities across all the fields. 

The size of the matrix will be the size of our dataframe (948 documents).


```python
n = len(df)
```


```python
n
```




    948




```python
score = np.zeros(n)

query = "I just discovered the course, is it too late to join?"

for f in fields:
    q = vectorizers[f].transform([query])
    X = matrices[f]

    f_score = cosine_similarity(X, q).flatten()

    score = score + f_score
```

Rather than having FAQs for all courses, let's filter by course (post-processing):


```python
filter = {
    'course': 'data-engineering-zoomcamp'
}
```


```python
for field, value in filter.items():
    mask = (df[field] == value).astype(int).values      #this is a Boolean mask, so a value "True" means that it pertains to the DE Zoomcamp
    score = score * mask
```


```python
mask
```




    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, ..., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])




```python
score
```




    array([1.82703079, 1.49512426, 1.23253339, 1.45772228, 1.49512426,
           1.49512426, 0.97571381, 1.67069698, 1.22089233, 1.49512426,
           1.36407773, 1.25620994, 0.49512426, 0.49512426, 0.49512426,
           0.59193348, 0.49512426, 1.20932345, 0.57041627, 0.49512426,
           0.49512426, 0.49512426, 0.79499188, 0.60033101, 0.49512426,
           0.49512426, 0.49512426, 0.76959902, 0.62340833, 0.49512426,
           0.49512426, 0.49512426, 0.49512426, 0.96108   , 1.49512426,
           0.90368553, 0.49512426, 0.49512426, 0.49512426, 0.52668735,
           0.54427244, 1.04472639, 0.49512426, 0.53842198, 0.        ,
           0.        , 0.        , 0.        , 0.02804374, 0.        , ...,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ])




```python
idx = np.argsort(score)[-5:]
```


```python
#Top 5 documents by similary score:
df.iloc[idx]
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
      <th>course</th>
      <th>section</th>
      <th>question</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - how many Zoomcamps in a year?</td>
      <td>There are 3 Zoom Camps in a year, as of 2024. ...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - What can I do before the course starts?</td>
      <td>You can start by installing and setting up all...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - What are the prerequisites for this c...</td>
      <td>GitHub - DataTalksClub data-engineering-zoomca...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - Can I follow the course after it fini...</td>
      <td>Yes, we will keep all the materials after the ...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - When will the course start?</td>
      <td>The purpose of this document is to capture fre...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# To get better results, we can boost matches where the terms in the question field match the query more closely.  
# We can give a higher score to "question" rather than "text"

score = np.zeros(n)

query = "I just discovered the course, is it too late to join?"

boosts = {
    'question': 3,
    'text': 0.5
}

for f in fields:
    q = vectorizers[f].transform([query])
    X = matrices[f]

    f_score = cosine_similarity(X, q).flatten()

    boost = boosts.get(f, 1.0)

    score = score + boost * f_score
```


```python
score
```




    array([3.28960182, 3.49512426, 2.70735166, 2.86194785, 3.49512426,
           3.49512426, 1.93689291, 3.58291062, 2.67242848, 3.49512426,
           3.10198469, 2.38161407, 0.49512426, 0.49512426, 0.49512426,
           0.54352887, 0.49512426, 2.63772182, 0.53277026, 0.49512426,
           0.49512426, 0.49512426, 0.64505807, 0.54772763, 0.49512426,
           0.49512426, 0.49512426, 0.63236164, 0.55926629, 0.49512426,
           0.49512426, 0.49512426, 0.49512426, 1.76390631, 3.49512426,
           1.72080809, 0.49512426, 0.49512426, 0.49512426, 0.5109058 ,
           0.51969835, 1.9654566 , 0.49512426, 0.51677312, 0.        ,
           0.        , 0.        , 0.        , 0.01402187, 0.        , ...,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.01708285, 0.01874268, 0.        ,
           0.02873746, 0.        , 0.        , 0.        , 0.        ,
           0.06778297, 0.        , 0.05012952, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.01639304,
           0.0584401 , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.01831462,
           0.0225796 , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ])



We just performed search across multiple fields with filtering.  

Let's create a class with everything implemented.

## Putting it all together


```python
class TextSearch:

    def __init__(self, text_fields):
        self.text_fields = text_fields
        self.matrices = {}
        self.vectorizers = {}

    def fit(self, records, vectorizer_params={}):
        self.df = pd.DataFrame(records)

        for f in self.text_fields:
            cv = TfidfVectorizer(**vectorizer_params)
            X = cv.fit_transform(self.df[f])
            self.matrices[f] = X
            self.vectorizers[f] = cv

    def search(self, query, n_results=10, boost={}, filters={}):
        score = np.zeros(len(self.df))

        for f in self.text_fields:
            b = boost.get(f, 1.0)
            q = self.vectorizers[f].transform([query])
            s = cosine_similarity(self.matrices[f], q).flatten()
            score = score + b * s

        for field, value in filters.items():
            mask = (self.df[field] == value).values
            score = score * mask

        idx = np.argsort(-score)[:n_results]
        results = self.df.iloc[idx]
        return results.to_dict(orient='records')
```


```python
index = TextSearch(
    text_fields=['section', 'question', 'text']
)
index.fit(documents)

index.search(
    query='I just signed up. Is it too late to join the course?',
    n_results=5,
    boost={'question': 3.0},
    filters={'course': 'data-engineering-zoomcamp'}
)
```




    [{'text': "Yes, even if you don't register, you're still eligible to submit the homeworks.\nBe aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.",
      'section': 'General course-related questions',
      'question': 'Course - Can I still join the course after the start date?',
      'course': 'data-engineering-zoomcamp'},
     {'text': "The purpose of this document is to capture frequently asked technical questions\nThe exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  “Office Hours'' live.1\nSubscribe to course public Google Calendar (it works from Desktop only).\nRegister before the course starts using this link.\nJoin the course Telegram channel with announcements.\nDon’t forget to register in DataTalks.Club's Slack and join the channel.",
      'section': 'General course-related questions',
      'question': 'Course - When will the course start?',
      'course': 'data-engineering-zoomcamp'},
     {'text': 'Yes, we will keep all the materials after the course finishes, so you can follow the course at your own pace after it finishes.\nYou can also continue looking at the homeworks and continue preparing for the next cohort. I guess you can also start working on your final capstone project.',
      'section': 'General course-related questions',
      'question': 'Course - Can I follow the course after it finishes?',
      'course': 'data-engineering-zoomcamp'},
     {'text': 'No, late submissions are not allowed. But if the form is still not closed and it’s after the due date, you can still submit the homework. confirm your submission by the date-timestamp on the Course page.y\nOlder news:[source1] [source2]',
      'section': 'General course-related questions',
      'question': 'Homework - Are late submissions of homework allowed?',
      'course': 'data-engineering-zoomcamp'},
     {'text': 'You can start by installing and setting up all the dependencies and requirements:\nGoogle cloud account\nGoogle Cloud SDK\nPython 3 (installed with Anaconda)\nTerraform\nGit\nLook over the prerequisites and syllabus to see if you are comfortable with these subjects.',
      'section': 'General course-related questions',
      'question': 'Course - What can I do before the course starts?',
      'course': 'data-engineering-zoomcamp'}]



#### (See minsearch library that Alexey created to implement the above process)


**Note: Limitations on basic text search**

Basic text search looks for exact word matches only.  For example, let's say that a student asks, "I just discovered this course.  Can I still join?"  If a frequently asked question answer contains the words "discovered" and "join", then question and its answer will be returned as a result.  However, if basic text search is implemented without any modifications, the question "I just found out about the course.  Can I still enroll?" will return no results, even though a FAQ with the same meaning ("I just discovered this course.  Can I still join?") exists.  

## Vector Search

Vector search is useful when words don't match exactly.  Here we see a few different types of vector search.  They're both fairly old methods, but they help illustrate the point.  

First is applying singular value decomposition (SVD).  This is a way to reduce dimensionality while preserving as much information as possible.


```python
# Document vector
X
```




    <Compressed Sparse Row sparse matrix of dtype 'float64'
    	with 23808 stored elements and shape (948, 1333)>




```python
from sklearn.decomposition import TruncatedSVD
```


```python
X = matrices['text']
cv = vectorizers['text']
```


```python
svd = TruncatedSVD(n_components=16)
X_emb = svd.fit_transform(X)
```


```python
X_emb.shape
```




    (948, 16)



We've gone from features to 16 features.  


```python
X_emb[0] #First document
```




    array([ 0.09652786, -0.08195697, -0.10265425, -0.08042629,  0.06810401,
           -0.05799825,  0.02837983, -0.13782717,  0.24663788, -0.25501494,
            0.12841484,  0.08486312, -0.12004902,  0.05955784, -0.04039409,
           -0.0282507 ])



Instead of a bunch of zeros in a matrix with hundreds of columns, we've got a dense representation of the document.  This is called embedding.  

When we reduce the dimensionality, SVD tries to capture as much of the original information from the original documents as possible.  In the case of synonyms (such as "enroll" and "sign up") get reduced into the same concepts.  These dense vectors capture the semantic similarities between different words.  

At the end, we get the above representation, where we can think of each score is the extent to which the document (FAQ in this case) relates to a particular topic.

We can do the same thing with the query.


```python
# representing the query 

query = 'I just signed up. Is it too late to join the course?'

Q = cv.transform([query])
Q_emb = svd.transform(Q)
Q_emb[0]
```




    array([ 0.05790215, -0.03845095, -0.05667402, -0.02833906,  0.04039044,
           -0.06036059,  0.01526585, -0.0897822 ,  0.16965456, -0.16647851,
            0.08795804,  0.06600029, -0.07771599,  0.04879043, -0.02845536,
           -0.0155083 ])



We'll next compute the cosine similarity.  The principles are still the same, even though it's a dense matrix versus a sparse matrix.


```python
# compute similarity of the document vector and the query vector
np.dot(X_emb[0], Q_emb[0])
```




    0.15091374232849208




```python
#Compute across all vectors
score = cosine_similarity(X_emb, Q_emb).flatten()    #Flatten turns score into a 1-dimensional array
idx = np.argsort(-score)[:10]
df.loc[idx]
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
      <th>course</th>
      <th>section</th>
      <th>question</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - When will the course start?</td>
      <td>The purpose of this document is to capture fre...</td>
    </tr>
    <tr>
      <th>436</th>
      <td>machine-learning-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Is it going to be live? When?</td>
      <td>The course videos are pre-recorded, you can st...</td>
    </tr>
    <tr>
      <th>449</th>
      <td>machine-learning-zoomcamp</td>
      <td>General course-related questions</td>
      <td>The course has already started. Can I still jo...</td>
      <td>Yes, you can. You won’t be able to submit some...</td>
    </tr>
    <tr>
      <th>764</th>
      <td>machine-learning-zoomcamp</td>
      <td>Projects (Midterm and Capstone)</td>
      <td>What If I submitted only two projects and fail...</td>
      <td>If you have submitted two projects (and peer-r...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - Can I follow the course after it fini...</td>
      <td>Yes, we will keep all the materials after the ...</td>
    </tr>
    <tr>
      <th>451</th>
      <td>machine-learning-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Can I submit the homework after the due date?</td>
      <td>No, it’s not possible. The form is closed afte...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - Can I still join the course after the...</td>
      <td>Yes, even if you don't register, you're still ...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Homework - Are late submissions of homework al...</td>
      <td>No, late submissions are not allowed. But if t...</td>
    </tr>
    <tr>
      <th>814</th>
      <td>mlops-zoomcamp</td>
      <td>+-General course questions</td>
      <td>What if my answer is not exactly the same as t...</td>
      <td>Please choose the closest one to your answer. ...</td>
    </tr>
    <tr>
      <th>440</th>
      <td>machine-learning-zoomcamp</td>
      <td>General course-related questions</td>
      <td>I filled the form, but haven't received a conf...</td>
      <td>The process is automated now, so you should re...</td>
    </tr>
  </tbody>
</table>
</div>




There are some benefits to doing search in this way, including dealing with synonyms.  We also have vectors we can use for other purposes, such as training a ML model.  With the original sparse vector, would be much more difficult to do this.  

Something to consider...even though the X matrix has only positive values, the embedded X matrix has negative values.  It's difficult to interpret negative values.  To address this, we can use a different method, **Non-Negative Matrix Factorization**.  


```python
from sklearn.decomposition import NMF

nmf = NMF(n_components=16)
X_emb = nmf.fit_transform(X)
X_emb[0]
```




    array([0.00600652, 0.00589958, 0.        , 0.        , 0.08560362,
           0.        , 0.00106776, 0.        , 0.00221502, 0.01230238,
           0.00029294, 0.        , 0.        , 0.00764361, 0.00462483,
           0.0090999 ])



It's clearer to see now which documents are most important.  We can think about this as topic modeling:  we have 16 different topics, and we can see that topics 7 and 9 are most relevant to this particular document (first document in the list).  


```python
Q = cv.transform([query])
Q_emb = nmf.transform(Q)
Q_emb[0]
```




    array([6.99823218e-03, 5.91035267e-03, 0.00000000e+00, 3.31317467e-03,
           4.04688939e-02, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
           9.41589947e-05, 8.90109940e-04, 0.00000000e+00, 0.00000000e+00,
           0.00000000e+00, 9.17303511e-03, 1.02426119e-02, 0.00000000e+00])



For the query, topics 2, 9, and 14 are most relevant.

Now, we compute cosine similarity.


```python
score = cosine_similarity(X_emb, Q_emb).flatten()
idx = np.argsort(-score)[:10]
df.loc[idx]
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
      <th>course</th>
      <th>section</th>
      <th>question</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>456</th>
      <td>machine-learning-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Submitting learning in public links</td>
      <td>When you post about what you learned from the ...</td>
    </tr>
    <tr>
      <th>452</th>
      <td>machine-learning-zoomcamp</td>
      <td>General course-related questions</td>
      <td>I just joined. What should I do next? How can ...</td>
      <td>Welcome to the course! Go to the course page (...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - When will the course start?</td>
      <td>The purpose of this document is to capture fre...</td>
    </tr>
    <tr>
      <th>760</th>
      <td>machine-learning-zoomcamp</td>
      <td>Projects (Midterm and Capstone)</td>
      <td>How to conduct peer reviews for projects?</td>
      <td>Answer: Previous cohorts projects page has ins...</td>
    </tr>
    <tr>
      <th>809</th>
      <td>machine-learning-zoomcamp</td>
      <td>Miscellaneous</td>
      <td>Any advice for adding the Machine Learning Zoo...</td>
      <td>I’ve seen LinkedIn users list DataTalksClub as...</td>
    </tr>
    <tr>
      <th>772</th>
      <td>machine-learning-zoomcamp</td>
      <td>Miscellaneous</td>
      <td>Is there a way to serve up a form for users to...</td>
      <td>Yes, you can create a mobile app or interface ...</td>
    </tr>
    <tr>
      <th>758</th>
      <td>machine-learning-zoomcamp</td>
      <td>Projects (Midterm and Capstone)</td>
      <td>What modules, topics, problem-sets should a mi...</td>
      <td>Answer: Ideally midterms up to module-06, caps...</td>
    </tr>
    <tr>
      <th>449</th>
      <td>machine-learning-zoomcamp</td>
      <td>General course-related questions</td>
      <td>The course has already started. Can I still jo...</td>
      <td>Yes, you can. You won’t be able to submit some...</td>
    </tr>
    <tr>
      <th>14</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Homework - What are homework and project deadl...</td>
      <td>You can find the latest and up-to-date deadlin...</td>
    </tr>
    <tr>
      <th>17</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Homework and Leaderboard - what is the system ...</td>
      <td>After you submit your homework it will be grad...</td>
    </tr>
  </tbody>
</table>
</div>



This is the simplest way to creating embeddings.  These methods have been around for a long time (e.g., latent semantic embedding).  The issue with these approaches is that the input is bag of words, so you lose the information about word order.  Sometimes word order is quite important.  For RAG, it might not be that important--especially for this dataset--but some situations, including more advanced RAG, may demand another approach.

Book reference:  _Introduction to Information Retrieval_

Alexey also recommends searching "metrics for ranking" for selecting embedders, how many dimensions to choose, etc.

## BERT

BERT (Bidirectional Encoder Representations from Transformers) is a neural network (transformer) that turns a document into an embedding.  They capture not only semantic similarity but also word order.  It's especially useful for things like "find out more", abbreviations, or longer negations ("don't do that"...).  


For this part, we'll get models from Huggingface.


```python
import torch
from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # Set the model to evaluation mode if not training
```




    BertModel(
      (embeddings): BertEmbeddings(
        (word_embeddings): Embedding(30522, 768, padding_idx=0)
        (position_embeddings): Embedding(512, 768)
        (token_type_embeddings): Embedding(2, 768)
        (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (encoder): BertEncoder(
        (layer): ModuleList(
          (0-11): 12 x BertLayer(
            (attention): BertAttention(
              (self): BertSdpaSelfAttention(
                (query): Linear(in_features=768, out_features=768, bias=True)
                (key): Linear(in_features=768, out_features=768, bias=True)
                (value): Linear(in_features=768, out_features=768, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
              (output): BertSelfOutput(
                (dense): Linear(in_features=768, out_features=768, bias=True)
                (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
                (dropout): Dropout(p=0.1, inplace=False)
              )
            )
            (intermediate): BertIntermediate(
              (dense): Linear(in_features=768, out_features=3072, bias=True)
              (intermediate_act_fn): GELUActivation()
            )
            (output): BertOutput(
              (dense): Linear(in_features=3072, out_features=768, bias=True)
              (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
              (dropout): Dropout(p=0.1, inplace=False)
            )
          )
        )
      )
      (pooler): BertPooler(
        (dense): Linear(in_features=768, out_features=768, bias=True)
        (activation): Tanh()
      )
    )




```python
texts = [
    "Yes, we will keep all the materials after the course finishes.",
    "You can follow the course at your own pace after it finishes"
]
encoded_input = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
encoded_input

```




    {'input_ids': tensor([[  101,  2748,  1010,  2057,  2097,  2562,  2035,  1996,  4475,  2044,
              1996,  2607, 12321,  1012,   102],
            [  101,  2017,  2064,  3582,  1996,  2607,  2012,  2115,  2219,  6393,
              2044,  2009, 12321,   102,     0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]])}




```python
with torch.no_grad():  # Disable gradient calculation for inference
    outputs = model(**encoded_input)
    hidden_states = outputs.last_hidden_state
```


```python
hidden_states.shape
```




    torch.Size([2, 15, 768])




```python
hidden_states[0]
```




    tensor([[ 0.1010,  0.0181,  0.1303,  ..., -0.2932,  0.1863,  0.6615],
            [ 1.0608, -0.1242,  0.1370,  ..., -0.1605,  1.0429,  0.3532],
            [ 0.1802,  0.0776,  0.3941,  ..., -0.1379,  0.5974,  0.1704],
            ...,
            [ 0.4738, -0.0184,  0.2186,  ..., -0.0013, -0.0833, -0.2170],
            [ 0.6516,  0.1216, -0.2494,  ...,  0.1557, -0.5632, -0.4310],
            [ 0.7164,  0.2157, -0.0281,  ...,  0.2281, -0.6725, -0.3245]])




```python
hidden_states[0].shape
```




    torch.Size([15, 768])



hidden_states[0].shape


```python
sentence_embeddings = hidden_states.mean(dim=1)
sentence_embeddings.shape
```




    torch.Size([2, 768])




```python
sentence_embeddings
```




    tensor([[ 0.3600, -0.1607,  0.3545,  ...,  0.0429,  0.0348, -0.0382],
            [ 0.1785, -0.5000,  0.2528,  ..., -0.1141, -0.3361,  0.4110]])



So for the result given above, the first (inner) list item shown is the embedding for document #1, the next inner list item is embedding for document #2, etc.

BERT for this purpose is overkill, but it is much, much more powerful for more complex documents and queries.

Now, we need to compute it for all the fields (section, question, text).  We take our query, tokenize the query, then extract the embedding from it. 


```python
X.shape
```




    (948, 1333)




```python
X_emb = sentence_embeddings.numpy()
```


```python
def make_batches(seq, n):
    result = []
    for i in range(0, len(seq), n):
        batch = seq[i:i+n]
        result.append(batch)
    return result
```


```python
from tqdm.auto import tqdm    #Shows status bars
texts = df['text'].tolist()
text_batches = make_batches(texts, 8)

all_embeddings = []

for batch in tqdm(text_batches):
    encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**encoded_input)
        hidden_states = outputs.last_hidden_state
        
        batch_embeddings = hidden_states.mean(dim=1)
        batch_embeddings_np = batch_embeddings.cpu().numpy()
        all_embeddings.append(batch_embeddings_np)

final_embeddings = np.vstack(all_embeddings)
```


      0%|          | 0/119 [00:00<?, ?it/s]



```python
def compute_embeddings(texts, batch_size=8):
    text_batches = make_batches(texts, 8)
    
    all_embeddings = []
    
    for batch in tqdm(text_batches):
        encoded_input = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
    
        with torch.no_grad():
            outputs = model(**encoded_input)
            hidden_states = outputs.last_hidden_state
            
            batch_embeddings = hidden_states.mean(dim=1)
            batch_embeddings_np = batch_embeddings.cpu().numpy()
            all_embeddings.append(batch_embeddings_np)
    
    final_embeddings = np.vstack(all_embeddings)
    return final_embeddings
```


```python
X_text = compute_embeddings(df['text'].tolist())
```


      0%|          | 0/119 [00:00<?, ?it/s]



```python
X_text.shape
```




    (948, 768)




```python
np.set_printoptions(threshold=40, edgeitems=4)

X_text
```




    array([[-0.00456325, -0.11667515,  0.6274717 ,  0.04425927, ...,
            -0.09477367, -0.03659188,  0.10031687,  0.02927116],
           [-0.14233607, -0.19853897,  0.28455406,  0.37105793, ...,
            -0.08485454, -0.01139052, -0.15399775,  0.0953509 ],
           [ 0.19672215, -0.08461291,  0.28200477,  0.07613111, ...,
            -0.08951467,  0.11395848, -0.06448034, -0.01282625],
           [ 0.34318587, -0.08406589,  0.36584705,  0.09601819, ...,
            -0.2666232 , -0.03561973, -0.0370797 ,  0.22697449],
           ...,
           [-0.15767317, -0.07575713,  0.07039975, -0.05150383, ...,
             0.04530649, -0.19681732, -0.2565146 ,  0.29486308],
           [-0.28217435, -0.3332434 ,  0.29785013, -0.06722228, ...,
            -0.04969649, -0.3504273 ,  0.03266073,  0.09537268],
           [-0.4280712 , -0.39468765,  0.30942   ,  0.08858248, ...,
            -0.03242783, -0.05943285, -0.12965178,  0.0788708 ],
           [-0.16892141, -0.25146288,  0.47843295,  0.07010107, ...,
            -0.11354773, -0.18535407, -0.16108926,  0.27272934]],
          dtype=float32)



The work we did above illustrates the main principles of how relevance search works.  The main idea is that we represent all the documents and the query in a vector space--could be bag-of-words space, BERT space, embeddings from LLM, some sort of vector.  Then, we compute the similarity among all the documents in our index and our query.  

In this example, we had less than 1,000 documents, so we can afford to compute the similarity between the query and all the documents.  But if you're Google and you've indexed the entire internet, you don't want to compute similarity of a query with documents across the entire internet.  

There are two key approaches to address this problem.  For text search, Inverted Index and Posting List are methods/tools used.  Another idea for vector search is LSH: Locality-sensitive hashing.  A good implementation that is pretty fast is Random Projection.  The idea is to take a long embedding and turn it into a bit vector with ones and zeros only.  In this form, we can still compute cosine similarity between the bit vectors, and it is roughly the same as the cosine similarity between the original vectors.  It's a fast way to compute cosine similarity and there are ways to make it even faster. 

Book recommendation:  _Mining of Massive Datasets_
