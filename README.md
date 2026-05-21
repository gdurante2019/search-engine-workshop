# Search Engine Workshop (prep for LLM-Zoomcamp)

Reference notebook:  https://github.com/alexeygrigorev/build-your-own-search-engine \
Video: https://www.youtube.com/watch?v=nMrGK5QgPVE

## Overview
In this workshop, we created a search engine using various methods.  We used the FAQ documents from three DataTalks.Club Zoomcamps:  Data Engineering Zoomcamp, ML Zoomcamp, and MLOps Zoomcamp.    

### Prepare environment

First step is to prepare the environment.  I created an environment on my machine, but the instructor used GitHub Codespaces, which is another great option when you don't want to take the time to set up another environment on your machine.  The libraries installed include ```requests```, ```pandas```, ```scikit-learn```, and ```jupyter```.  The next step is to launch jupyter notebook.

### Text search:  FAQ docs ingestion and exploration

To begin, we need to obtain these documents and then put them into a format that we can use to explore the structure of the content and feed them into models.  Then we will employ different tools of varying levels of sophistication to perform searches.  


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




<style>#sk-container-id-27 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-27 {
  color: var(--sklearn-color-text);
}

#sk-container-id-27 pre {
  padding: 0;
}

#sk-container-id-27 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-27 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-27 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-27 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-27 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-27 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-27 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-27 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-27 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-27 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-27 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-27 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-27 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-27 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-27 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-27 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-27 div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-27 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-27 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-27 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-27 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

#sk-container-id-27 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-27 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-27 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-27 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-27 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-27 div.sk-label label.sk-toggleable__label,
#sk-container-id-27 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-27 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-27 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-27 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-27 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-27 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-27 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-27 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-27 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-27 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-27 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-27 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-27 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.estimator-table summary {
    padding: .5rem;
    font-family: monospace;
    cursor: pointer;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover {
    background-color: #e0e0e0;
}

.estimator-table table td {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left;
}

.user-set td.value pre {
    color:rgb(255, 94, 0) !important;
    background-color: transparent !important;
}

.default td {
    color: black;
    text-align: left;
}

.user-set td i,
.default td i {
    color: black;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}
</style><body><div id="sk-container-id-27" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>CountVectorizer()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-27" type="checkbox" checked><label for="sk-estimator-id-27" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>CountVectorizer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html">?<span>Documentation for CountVectorizer</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted" data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('input',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">input&nbsp;</td>
            <td class="value">&#x27;content&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('encoding',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">encoding&nbsp;</td>
            <td class="value">&#x27;utf-8&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('decode_error',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">decode_error&nbsp;</td>
            <td class="value">&#x27;strict&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strip_accents',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">strip_accents&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('lowercase',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">lowercase&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('preprocessor',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">preprocessor&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('tokenizer',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">tokenizer&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('stop_words',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">stop_words&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('token_pattern',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">token_pattern&nbsp;</td>
            <td class="value">&#x27;(?u)\\b\\w\\w+\\b&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('ngram_range',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">ngram_range&nbsp;</td>
            <td class="value">(1, ...)</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('analyzer',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">analyzer&nbsp;</td>
            <td class="value">&#x27;word&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_df',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_df&nbsp;</td>
            <td class="value">1.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_df',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">min_df&nbsp;</td>
            <td class="value">1</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_features&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('vocabulary',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">vocabulary&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('binary',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">binary&nbsp;</td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('dtype',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">dtype&nbsp;</td>
            <td class="value">&lt;class &#x27;numpy.int64&#x27;&gt;</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div><script>function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.fa-regular.fa-copy').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const paramName = element.parentElement.nextElementSibling.textContent.trim();
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});
</script></body>



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




<style>#sk-container-id-28 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-28 {
  color: var(--sklearn-color-text);
}

#sk-container-id-28 pre {
  padding: 0;
}

#sk-container-id-28 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-28 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-28 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-28 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-28 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-28 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-28 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-28 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-28 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-28 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-28 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-28 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-28 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-28 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-28 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-28 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-28 div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-28 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-28 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-28 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-28 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

#sk-container-id-28 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-28 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-28 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-28 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-28 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-28 div.sk-label label.sk-toggleable__label,
#sk-container-id-28 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-28 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-28 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-28 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-28 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-28 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-28 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-28 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-28 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-28 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-28 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-28 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-28 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.estimator-table summary {
    padding: .5rem;
    font-family: monospace;
    cursor: pointer;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover {
    background-color: #e0e0e0;
}

.estimator-table table td {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left;
}

.user-set td.value pre {
    color:rgb(255, 94, 0) !important;
    background-color: transparent !important;
}

.default td {
    color: black;
    text-align: left;
}

.user-set td i,
.default td i {
    color: black;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}
</style><body><div id="sk-container-id-28" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>CountVectorizer(min_df=5)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-28" type="checkbox" checked><label for="sk-estimator-id-28" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>CountVectorizer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html">?<span>Documentation for CountVectorizer</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted" data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('input',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">input&nbsp;</td>
            <td class="value">&#x27;content&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('encoding',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">encoding&nbsp;</td>
            <td class="value">&#x27;utf-8&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('decode_error',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">decode_error&nbsp;</td>
            <td class="value">&#x27;strict&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strip_accents',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">strip_accents&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('lowercase',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">lowercase&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('preprocessor',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">preprocessor&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('tokenizer',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">tokenizer&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('stop_words',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">stop_words&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('token_pattern',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">token_pattern&nbsp;</td>
            <td class="value">&#x27;(?u)\\b\\w\\w+\\b&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('ngram_range',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">ngram_range&nbsp;</td>
            <td class="value">(1, ...)</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('analyzer',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">analyzer&nbsp;</td>
            <td class="value">&#x27;word&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_df',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_df&nbsp;</td>
            <td class="value">1.0</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_df',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">min_df&nbsp;</td>
            <td class="value">5</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_features&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('vocabulary',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">vocabulary&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('binary',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">binary&nbsp;</td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('dtype',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">dtype&nbsp;</td>
            <td class="value">&lt;class &#x27;numpy.int64&#x27;&gt;</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div><script>function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.fa-regular.fa-copy').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const paramName = element.parentElement.nextElementSibling.textContent.trim();
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});
</script></body>




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




<style>#sk-container-id-29 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-29 {
  color: var(--sklearn-color-text);
}

#sk-container-id-29 pre {
  padding: 0;
}

#sk-container-id-29 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-29 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-29 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-29 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-29 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-29 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-29 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-29 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-29 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-29 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-29 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-29 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-29 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-29 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-29 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-29 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-29 div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-29 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-29 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-29 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-29 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

#sk-container-id-29 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-29 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-29 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-29 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-29 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-29 div.sk-label label.sk-toggleable__label,
#sk-container-id-29 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-29 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-29 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-29 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-29 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-29 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-29 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-29 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-29 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-29 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-29 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-29 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-29 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.estimator-table summary {
    padding: .5rem;
    font-family: monospace;
    cursor: pointer;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover {
    background-color: #e0e0e0;
}

.estimator-table table td {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left;
}

.user-set td.value pre {
    color:rgb(255, 94, 0) !important;
    background-color: transparent !important;
}

.default td {
    color: black;
    text-align: left;
}

.user-set td i,
.default td i {
    color: black;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}
</style><body><div id="sk-container-id-29" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>CountVectorizer()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-29" type="checkbox" checked><label for="sk-estimator-id-29" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>CountVectorizer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html">?<span>Documentation for CountVectorizer</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted" data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('input',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">input&nbsp;</td>
            <td class="value">&#x27;content&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('encoding',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">encoding&nbsp;</td>
            <td class="value">&#x27;utf-8&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('decode_error',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">decode_error&nbsp;</td>
            <td class="value">&#x27;strict&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strip_accents',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">strip_accents&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('lowercase',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">lowercase&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('preprocessor',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">preprocessor&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('tokenizer',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">tokenizer&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('stop_words',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">stop_words&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('token_pattern',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">token_pattern&nbsp;</td>
            <td class="value">&#x27;(?u)\\b\\w\\w+\\b&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('ngram_range',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">ngram_range&nbsp;</td>
            <td class="value">(1, ...)</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('analyzer',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">analyzer&nbsp;</td>
            <td class="value">&#x27;word&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_df',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_df&nbsp;</td>
            <td class="value">1.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_df',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">min_df&nbsp;</td>
            <td class="value">1</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_features&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('vocabulary',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">vocabulary&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('binary',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">binary&nbsp;</td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('dtype',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">dtype&nbsp;</td>
            <td class="value">&lt;class &#x27;numpy.int64&#x27;&gt;</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div><script>function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.fa-regular.fa-copy').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const paramName = element.parentElement.nextElementSibling.textContent.trim();
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});
</script></body>




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




<style>#sk-container-id-30 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: #000;
  --sklearn-color-text-muted: #666;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-30 {
  color: var(--sklearn-color-text);
}

#sk-container-id-30 pre {
  padding: 0;
}

#sk-container-id-30 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-30 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-30 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-30 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-30 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-30 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-30 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-30 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-30 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-30 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-30 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-30 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-30 label.sk-toggleable__label {
  cursor: pointer;
  display: flex;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
  align-items: start;
  justify-content: space-between;
  gap: 0.5em;
}

#sk-container-id-30 label.sk-toggleable__label .caption {
  font-size: 0.6rem;
  font-weight: lighter;
  color: var(--sklearn-color-text-muted);
}

#sk-container-id-30 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-30 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-30 div.sk-toggleable__content {
  display: none;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-30 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-30 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-30 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-30 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  display: block;
  width: 100%;
  overflow: visible;
}

#sk-container-id-30 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-30 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-30 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-30 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-30 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-30 div.sk-label label.sk-toggleable__label,
#sk-container-id-30 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-30 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-30 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-30 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-30 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-30 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-30 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-30 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-30 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 0.5em;
  text-align: center;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-30 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-30 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-30 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-30 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}

.estimator-table summary {
    padding: .5rem;
    font-family: monospace;
    cursor: pointer;
}

.estimator-table details[open] {
    padding-left: 0.1rem;
    padding-right: 0.1rem;
    padding-bottom: 0.3rem;
}

.estimator-table .parameters-table {
    margin-left: auto !important;
    margin-right: auto !important;
}

.estimator-table .parameters-table tr:nth-child(odd) {
    background-color: #fff;
}

.estimator-table .parameters-table tr:nth-child(even) {
    background-color: #f6f6f6;
}

.estimator-table .parameters-table tr:hover {
    background-color: #e0e0e0;
}

.estimator-table table td {
    border: 1px solid rgba(106, 105, 104, 0.232);
}

.user-set td {
    color:rgb(255, 94, 0);
    text-align: left;
}

.user-set td.value pre {
    color:rgb(255, 94, 0) !important;
    background-color: transparent !important;
}

.default td {
    color: black;
    text-align: left;
}

.user-set td i,
.default td i {
    color: black;
}

.copy-paste-icon {
    background-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0NDggNTEyIj48IS0tIUZvbnQgQXdlc29tZSBGcmVlIDYuNy4yIGJ5IEBmb250YXdlc29tZSAtIGh0dHBzOi8vZm9udGF3ZXNvbWUuY29tIExpY2Vuc2UgLSBodHRwczovL2ZvbnRhd2Vzb21lLmNvbS9saWNlbnNlL2ZyZWUgQ29weXJpZ2h0IDIwMjUgRm9udGljb25zLCBJbmMuLS0+PHBhdGggZD0iTTIwOCAwTDMzMi4xIDBjMTIuNyAwIDI0LjkgNS4xIDMzLjkgMTQuMWw2Ny45IDY3LjljOSA5IDE0LjEgMjEuMiAxNC4xIDMzLjlMNDQ4IDMzNmMwIDI2LjUtMjEuNSA0OC00OCA0OGwtMTkyIDBjLTI2LjUgMC00OC0yMS41LTQ4LTQ4bDAtMjg4YzAtMjYuNSAyMS41LTQ4IDQ4LTQ4ek00OCAxMjhsODAgMCAwIDY0LTY0IDAgMCAyNTYgMTkyIDAgMC0zMiA2NCAwIDAgNDhjMCAyNi41LTIxLjUgNDgtNDggNDhMNDggNTEyYy0yNi41IDAtNDgtMjEuNS00OC00OEwwIDE3NmMwLTI2LjUgMjEuNS00OCA0OC00OHoiLz48L3N2Zz4=);
    background-repeat: no-repeat;
    background-size: 14px 14px;
    background-position: 0;
    display: inline-block;
    width: 14px;
    height: 14px;
    cursor: pointer;
}
</style><body><div id="sk-container-id-30" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>CountVectorizer(stop_words=&#x27;english&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-30" type="checkbox" checked><label for="sk-estimator-id-30" class="sk-toggleable__label fitted sk-toggleable__label-arrow"><div><div>CountVectorizer</div></div><div><a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.7/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html">?<span>Documentation for CountVectorizer</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></div></label><div class="sk-toggleable__content fitted" data-param-prefix="">
        <div class="estimator-table">
            <details>
                <summary>Parameters</summary>
                <table class="parameters-table">
                  <tbody>

        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('input',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">input&nbsp;</td>
            <td class="value">&#x27;content&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('encoding',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">encoding&nbsp;</td>
            <td class="value">&#x27;utf-8&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('decode_error',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">decode_error&nbsp;</td>
            <td class="value">&#x27;strict&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('strip_accents',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">strip_accents&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('lowercase',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">lowercase&nbsp;</td>
            <td class="value">True</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('preprocessor',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">preprocessor&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('tokenizer',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">tokenizer&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="user-set">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('stop_words',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">stop_words&nbsp;</td>
            <td class="value">&#x27;english&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('token_pattern',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">token_pattern&nbsp;</td>
            <td class="value">&#x27;(?u)\\b\\w\\w+\\b&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('ngram_range',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">ngram_range&nbsp;</td>
            <td class="value">(1, ...)</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('analyzer',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">analyzer&nbsp;</td>
            <td class="value">&#x27;word&#x27;</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_df',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_df&nbsp;</td>
            <td class="value">1.0</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('min_df',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">min_df&nbsp;</td>
            <td class="value">1</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('max_features',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">max_features&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('vocabulary',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">vocabulary&nbsp;</td>
            <td class="value">None</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('binary',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">binary&nbsp;</td>
            <td class="value">False</td>
        </tr>


        <tr class="default">
            <td><i class="copy-paste-icon"
                 onclick="copyToClipboard('dtype',
                          this.parentElement.nextElementSibling)"
            ></i></td>
            <td class="param">dtype&nbsp;</td>
            <td class="value">&lt;class &#x27;numpy.int64&#x27;&gt;</td>
        </tr>

                  </tbody>
                </table>
            </details>
        </div>
    </div></div></div></div></div><script>function copyToClipboard(text, element) {
    // Get the parameter prefix from the closest toggleable content
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const fullParamName = paramPrefix ? `${paramPrefix}${text}` : text;

    const originalStyle = element.style;
    const computedStyle = window.getComputedStyle(element);
    const originalWidth = computedStyle.width;
    const originalHTML = element.innerHTML.replace('Copied!', '');

    navigator.clipboard.writeText(fullParamName)
        .then(() => {
            element.style.width = originalWidth;
            element.style.color = 'green';
            element.innerHTML = "Copied!";

            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        })
        .catch(err => {
            console.error('Failed to copy:', err);
            element.style.color = 'red';
            element.innerHTML = "Failed!";
            setTimeout(() => {
                element.innerHTML = originalHTML;
                element.style = originalStyle;
            }, 2000);
        });
    return false;
}

document.querySelectorAll('.fa-regular.fa-copy').forEach(function(element) {
    const toggleableContent = element.closest('.sk-toggleable__content');
    const paramPrefix = toggleableContent ? toggleableContent.dataset.paramPrefix : '';
    const paramName = element.parentElement.nextElementSibling.textContent.trim();
    const fullParamName = paramPrefix ? `${paramPrefix}${paramName}` : paramName;

    element.setAttribute('title', fullParamName);
});
</script></body>




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




    array([[0., 0., 0., ..., 0., 0., 0.]])




```python
# Source - https://stackoverflow.com/a/78793583
# Posted by Leolo, modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-20, License - CC BY-SA 4.0

np.set_printoptions(legacy='1.25')

query_dict = dict(zip(names, q.toarray()[0]))
query_dict
```




    {'01': 0.0,
     '02': 0.0,
     '03': 0.0,
     '04': 0.0,
     '05': 0.0,
     '06': 0.0,
     '09': 0.0,
     '10': 0.0,
     '100': 0.0,
     '11': 0.0,
     '12': 0.0,
     '127': 0.0,
     '13': 0.0,
     '14': 0.0,
     '15': 0.0,
     '16': 0.0,
     '17': 0.0,
     '19': 0.0,
     '1st': 0.0,
     '20': 0.0,
     '2019': 0.0,
     '2020': 0.0,
     '2021': 0.0,
     '2022': 0.0,
     '2023': 0.0,
     '2024': 0.0,
     '21': 0.0,
     '22': 0.0,
     '24': 0.0,
     '25': 0.0,
     '2pacx': 0.0,
     '30': 0.0,
     '35': 0.0,
     '403': 0.0,
     '42': 0.0,
     '50': 0.0,
     '5000': 0.0,
     '5431': 0.0,
     '5432': 0.0,
     '60': 0.0,
     '600': 0.0,
     '7077': 0.0,
     '80': 0.0,
     '8080': 0.0,
     '8888': 0.0,
     '9696': 0.0,
     'abhijit': 0.0,
     'able': 0.0,
     'abolade': 0.0,
     'absolute': 0.0,
     'accept': 0.0,
     'access': 0.0,
     'accordingly': 0.0,
     'account': 0.0,
     'accuracy': 0.0,
     'action': 0.0,
     'activate': 0.0,
     'actual': 0.0,
     'actually': 0.0,
     'add': 0.0,
     'added': 0.0,
     'adding': 0.0,
     'addition': 0.0,
     'additional': 0.0,
     'additionally': 0.0,
     'address': 0.0,
     'admin': 0.0,
     'advani': 0.0,
     'ahmed': 0.0,
     'ai': 0.0,
     'airflow': 0.0,
     'alejandro': 0.0,
     'alena': 0.0,
     'alex': 0.0,
     'alexey': 0.0,
     'alexeygrigorev': 0.0,
     'allocate': 0.0,
     'allow': 0.0,
     'allowed': 0.0,
     'allowing': 0.0,
     'allows': 0.0,
     'alpha': 0.0,
     'alternative': 0.0,
     'alternatively': 0.0,
     'amazon': 0.0,
     'amazonaws': 0.0,
     'amd64': 0.0,
     'aminat': 0.0,
     'anaconda': 0.0,
     'anaconda3': 0.0,
     'analytics': 0.0,
     'anand': 0.0,
     'andrii': 0.0,
     'ans': 0.0,
     'answer': 0.0,
     'answered': 0.0,
     'answers': 0.0,
     'apache': 0.0,
     'api': 0.0,
     'aponte': 0.0,
     'app': 0.0,
     'appear': 0.0,
     'appears': 0.0,
     'append': 0.0,
     'application': 0.0,
     'apply': 0.0,
     'appname': 0.0,
     'approach': 0.0,
     'apt': 0.0,
     'architecture': 0.0,
     'archives': 0.0,
     'args': 0.0,
     'argument': 0.0,
     'array': 0.0,
     'arrays': 0.0,
     'article': 0.0,
     'asia': 0.0,
     'ask': 0.0,
     'asked': 0.0,
     'asks': 0.0,
     'assigned': 0.0,
     'astype': 0.0,
     'attempting': 0.0,
     'attribute': 0.0,
     'attributeerror': 0.0,
     'auc': 0.0,
     'auth': 0.0,
     'authenticate': 0.0,
     'authentication': 0.0,
     'auto': 0.0,
     'automatically': 0.0,
     'available': 0.0,
     'average': 0.0,
     'avila': 0.0,
     'avoid': 0.0,
     'aws': 0.0,
     'azure': 0.0,
     'backend': 0.0,
     'bad': 0.0,
     'bar': 0.0,
     'base': 0.0,
     'based': 0.0,
     'bash': 0.0,
     'bashrc': 0.0,
     'basically': 0.0,
     'batch': 0.0,
     'best': 0.0,
     'better': 0.0,
     'bhaskar': 0.0,
     'bhatia': 0.0,
     'big': 0.0,
     'bigquery': 0.0,
     'billing': 0.0,
     'bin': 0.0,
     'binary': 0.0,
     'bind': 0.0,
     'bit': 0.0,
     'blob': 0.0,
     'block': 0.0,
     'blog': 0.0,
     'box': 0.0,
     'bq': 0.0,
     'branch': 0.0,
     'brew': 0.0,
     'brigida': 0.0,
     'browser': 0.0,
     'bucket': 0.0,
     'build': 0.0,
     'builder': 0.0,
     'building': 0.0,
     'built': 0.0,
     'calculate': 0.0,
     'calderin': 0.0,
     'called': 0.0,
     'calling': 0.0,
     'capstone': 0.0,
     'capture': 0.0,
     'care': 0.0,
     'careful': 0.0,
     'case': 0.0,
     'cast': 0.0,
     'cat': 0.0,
     'categorical': 0.0,
     'cause': 0.0,
     'caused': 0.0,
     'causing': 0.0,
     'cd': 0.0,
     'cell': 0.0,
     'certificate': 0.0,
     'chakraborty': 0.0,
     'change': 0.0,
     'changed': 0.0,
     'changes': 0.0,
     'changing': 0.0,
     'channel': 0.0,
     'check': 0.0,
     'checking': 0.0,
     'chizhmak': 0.0,
     'chmod': 0.0,
     'choice': 0.0,
     'choose': 0.0,
     'chown': 0.0,
     'churn': 0.0,
     'ci': 0.0,
     'class': 0.0,
     'classes': 0.0,
     'classification': 0.0,
     'clear': 0.0,
     'cli': 0.0,
     'click': 0.0,
     'client': 0.0,
     'clone': 0.0,
     'close': 0.0,
     'closed': 0.0,
     'closest': 0.0,
     'clothing': 0.0,
     'cloud': 0.0,
     'cloudfront': 0.0,
     'club': 0.0,
     'cluster': 0.0,
     'cmd': 0.0,
     'code': 0.0,
     'codespaces': 0.0,
     'coding': 0.0,
     'cohort': 0.0,
     'cohorts': 0.0,
     'col': 0.0,
     'colab': 0.0,
     'column': 0.0,
     'columns': 0.0,
     'com': 0.0,
     'command': 0.0,
     'commands': 0.0,
     'commit': 0.0,
     'common': 0.0,
     'community': 0.0,
     'compatibility': 0.0,
     'compatible': 0.0,
     'complete': 0.0,
     'completed': 0.0,
     'compose': 0.0,
     'compute': 0.0,
     'computer': 0.0,
     'conda': 0.0,
     'config': 0.0,
     'configuration': 0.0,
     'configure': 0.0,
     'configured': 0.0,
     'confirm': 0.0,
     'connect': 0.0,
     'connecting': 0.0,
     'connection': 0.0,
     'connector': 0.0,
     'consider': 0.0,
     'consistent': 0.0,
     'console': 0.0,
     'contain': 0.0,
     'container': 0.0,
     'containers': 0.0,
     'contains': 0.0,
     'content': 0.0,
     'contents': 0.0,
     'context': 0.0,
     'continue': 0.0,
     'control': 0.0,
     'convert': 0.0,
     'converted': 0.0,
     'copy': 0.0,
     'core': 0.0,
     'correct': 0.0,
     'correctly': 0.0,
     'correlation': 0.0,
     'corresponding': 0.0,
     'couldn': 0.0,
     'course': 0.49695797492447685,
     'cp': 0.0,
     'cpu': 0.0,
     'create': 0.0,
     'create_engine': 0.0,
     'created': 0.0,
     'creates': 0.0,
     'creating': 0.0,
     'creation': 0.0,
     'credentials': 0.0,
     'credit': 0.0,
     'crucial': 0.0,
     'csv': 0.0,
     'ctrl': 0.0,
     'curl': 0.0,
     'current': 0.0,
     'custom': 0.0,
     'd37ci6vzurychx': 0.0,
     'daemon': 0.0,
     'daniel': 0.0,
     'dashboard': 0.0,
     'data': 0.0,
     'database': 0.0,
     'dataframe': 0.0,
     'dataframes': 0.0,
     'dataproc': 0.0,
     'dataset': 0.0,
     'datasets': 0.0,
     'datatalks': 0.0,
     'datatalksclub': 0.0,
     'datatype': 0.0,
     'date': 0.0,
     'datetime': 0.0,
     'david': 0.0,
     'day': 0.0,
     'db': 0.0,
     'dbt': 0.0,
     'dbt_project': 0.0,
     'deadline': 0.0,
     'dealing': 0.0,
     'debian': 0.0,
     'debug': 0.0,
     'deep': 0.0,
     'def': 0.0,
     'default': 0.0,
     'define': 0.0,
     'defined': 0.0,
     'definition': 0.0,
     'delete': 0.0,
     'deleted': 0.0,
     'deleting': 0.0,
     'denied': 0.0,
     'dependencies': 0.0,
     'dependency': 0.0,
     'deploy': 0.0,
     'deploying': 0.0,
     'deployment': 0.0,
     'deprecated': 0.0,
     'description': 0.0,
     'desktop': 0.0,
     'detailed': 0.0,
     'details': 0.0,
     'dev': 0.0,
     'development': 0.0,
     'deviation': 0.0,
     'device': 0.0,
     'df': 0.0,
     'df_train': 0.0,
     'dict': 0.0,
     'dictionary': 0.0,
     'dictvectorizer': 0.0,
     'did': 0.0,
     'didn': 0.0,
     'difference': 0.0,
     'different': 0.0,
     'dino': 0.0,
     'dir': 0.0,
     'directly': 0.0,
     'directory': 0.0,
     'disk': 0.0,
     'distribution': 0.0,
     'doc': 0.0,
     'docker': 0.0,
     'dockerfile': 0.0,
     'docs': 0.0,
     'document': 0.0,
     'documentation': 0.0,
     'does': 0.0,
     'doesn': 0.0,
     'doing': 0.0,
     'dolocationid': 0.0,
     'don': 0.0,
     'dot': 0.0,
     'double': 0.0,
     'downgrade': 0.0,
     'download': 0.0,
     'downloaded': 0.0,
     'downloading': 0.0,
     'downloads': 0.0,
     'dpage': 0.0,
     'dragon': 0.0,
     'drive': 0.0,
     'drop': 0.0,
     'dropoff_datetime': 0.0,
     'dropping': 0.0,
     'dt': 0.0,
     'dtc': 0.0,
     'duration': 0.0,
     'dv': 0.0,
     'easier': 0.0,
     'easiest': 0.0,
     'easily': 0.0,
     'ec2': 0.0,
     'echo': 0.0,
     'ecr': 0.0,
     'edit': 0.0,
     'editing': 0.0,
     'editor': 0.0,
     'email': 0.0,
     'en': 0.0,
     'enable': 0.0,
     'enabled': 0.0,
     'encode': 0.0,
     'encoding': 0.0,
     'encounter': 0.0,
     'encountered': 0.0,
     'end': 0.0,
     'endpoint': 0.0,
     'engine': 0.0,
     'engineering': 0.0,
     'ensure': 0.0,
     'enter': 0.0,
     'entire': 0.0,
     'entries': 0.0,
     'env': 0.0,
     'environ': 0.0,
     'environment': 0.0,
     'environments': 0.0,
     'equal': 0.0,
     'erick': 0.0,
     'error': 0.0,
     'errors': 0.0,
     'escape': 0.0,
     'especially': 0.0,
     'eu': 0.0,
     'europe': 0.0,
     'evaluate': 0.0,
     'evaluated': 0.0,
     'evaluation': 0.0,
     'exactly': 0.0,
     'example': 0.0,
     'exception': 0.0,
     'exe': 0.0,
     'exec': 0.0,
     'executable': 0.0,
     'execute': 0.0,
     'executed': 0.0,
     'executing': 0.0,
     'execution': 0.0,
     'exist': 0.0,
     'existing': 0.0,
     'exists': 0.0,
     'exit': 0.0,
     'expected': 0.0,
     'experiment': 0.0,
     'explanation': 0.0,
     'explicitly': 0.0,
     'export': 0.0,
     'extension': 0.0,
     'external': 0.0,
     'extra': 0.0,
     'extract': 0.0,
     'face': 0.0,
     'faced': 0.0,
     'fact': 0.0,
     'fact_trips': 0.0,
     'fail': 0.0,
     'failed': 0.0,
     'failing': 0.0,
     'fails': 0.0,
     'false': 0.0,
     'faq': 0.0,
     'fatal': 0.0,
     'feature': 0.0,
     'features': 0.0,
     'fetch': 0.0,
     'fhv': 0.0,
     'fhv_tripdata_2019': 0.0,
     'field': 0.0,
     'figure': 0.0,
     'file': 0.0,
     'files': 0.0,
     'filesystem': 0.0,
     'filter': 0.0,
     'final': 0.0,
     'finally': 0.0,
     'findspark': 0.0,
     'fine': 0.0,
     'fit': 0.0,
     'fix': 0.0,
     'fixed': 0.0,
     'flag': 0.0,
     'flask': 0.0,
     'float': 0.0,
     'float64': 0.0,
     'flow': 0.0,
     'folder': 0.0,
     'folders': 0.0,
     'follow': 0.0,
     'followed': 0.0,
     'following': 0.0,
     'follows': 0.0,
     'forbidden': 0.0,
     'force': 0.0,
     'forget': 0.0,
     'forgot': 0.0,
     'form': 0.0,
     'format': 0.0,
     'forms': 0.0,
     'forward': 0.0,
     'fouesnard': 0.0,
     'free': 0.0,
     'fresh': 0.0,
     'fs': 0.0,
     'function': 0.0,
     'functions': 0.0,
     'future': 0.0,
     'gateway': 0.0,
     'gcloud': 0.0,
     'gcp': 0.0,
     'gcs': 0.0,
     'generate': 0.0,
     'generated': 0.0,
     'generator': 0.0,
     'george': 0.0,
     'get_dummies': 0.0,
     'get_feature_names_out': 0.0,
     'getdbt': 0.0,
     'getorcreate': 0.0,
     'gets': 0.0,
     'getting': 0.0,
     'git': 0.0,
     'github': 0.0,
     'githubusercontent': 0.0,
     'gitignore': 0.0,
     'given': 0.0,
     'gives': 0.0,
     'gmail': 0.0,
     'going': 0.0,
     'good': 0.0,
     'google': 0.0,
     'google_application_credentials': 0.0,
     'googleapi': 0.0,
     'googleapis': 0.0,
     'got': 0.0,
     'gpu': 0.0,
     'great': 0.0,
     'green': 0.0,
     'grigorev': 0.0,
     'group': 0.0,
     'gs': 0.0,
     'guide': 0.0,
     'gunicorn': 0.0,
     'gz': 0.0,
     'gzip': 0.0,
     'hadoop': 0.0,
     'hadoop3': 0.0,
     'hand': 0.0,
     'handle': 0.0,
     'happen': 0.0,
     'happens': 0.0,
     'haven': 0.0,
     'having': 0.0,
     'head': 0.0,
     'help': 0.0,
     'helped': 0.0,
     'helps': 0.0,
     'high': 0.0,
     'higher': 0.0,
     'highly': 0.0,
     'home': 0.0,
     'homebrew': 0.0,
     'homework': 0.0,
     'homeworks': 0.0,
     'hope': 0.0,
     'host': 0.0,
     'hostname': 0.0,
     'hot': 0.0,
     'hours': 0.0,
     'housing': 0.0,
     'hrithik': 0.0,
     'html': 0.0,
     'http': 0.0,
     'https': 0.0,
     'humberto': 0.0,
     'hw': 0.0,
     'hws': 0.0,
     'iam': 0.0,
     'icon': 0.0,
     'id': 0.0,
     'ide': 0.0,
     'idea': 0.0,
     'ignore': 0.0,
     'image': 0.0,
     'images': 0.0,
     'immediately': 0.0,
     'import': 0.0,
     'importance': 0.0,
     'important': 0.0,
     'importerror': 0.0,
     'importing': 0.0,
     'include': 0.0,
     'included': 0.0,
     'including': 0.0,
     'increase': 0.0,
     'index': 0.0,
     'info': 0.0,
     'information': 0.0,
     'init': 0.0,
     'initial': 0.0,
     'input': 0.0,
     'inside': 0.0,
     'install': 0.0,
     'installation': 0.0,
     'installed': 0.0,
     'installing': 0.0,
     'instance': 0.0,
     'instances': 0.0,
     'instead': 0.0,
     'instruction': 0.0,
     'instructions': 0.0,
     'insufficient': 0.0,
     'int': 0.0,
     'int64': 0.0,
     'integer': 0.0,
     'interesting': 0.0,
     'interface': 0.0,
     'internal': 0.0,
     'internet': 0.0,
     'interpreter': 0.0,
     'intro': 0.0,
     'invalid': 0.0,
     'io': 0.0,
     'ip': 0.0,
     'ipynb': 0.0,
     'isn': 0.0,
     'issue': 0.0,
     'issues': 0.0,
     'iteration': 0.0,
     'ivan': 0.0,
     'jan': 0.0,
     'jar': 0.0,
     'java': 0.0,
     'java_home': 0.0,
     'job': 0.0,
     'jobs': 0.0,
     'join': 0.7433502942325737,
     'json': 0.0,
     'jupyter': 0.0,
     'just': 0.4477310702009737,
     'kafka': 0.0,
     'kaggle': 0.0,
     'keras': 0.0,
     'kernel': 0.0,
     'key': 0.0,
     'keys': 0.0,
     'kill': 0.0,
     'kind': 0.0,
     'kinesis': 0.0,
     'kniazeva': 0.0,
     'know': 0.0,
     'known': 0.0,
     'konrad': 0.0,
     'krishna': 0.0,
     'kubectl': 0.0,
     'kubernetes': 0.0,
     'kumar': 0.0,
     'kwargs': 0.0,
     'lambda': 0.0,
     'laptop': 0.0,
     'large': 0.0,
     'larkin': 0.0,
     'later': 0.0,
     'latest': 0.0,
     'launch': 0.0,
     'layer': 0.0,
     'layers': 0.0,
     'lead': 0.0,
     'leaderboard': 0.0,
     'learn': 0.0,
     'learning': 0.0,
     'left': 0.0,
     'length': 0.0,
     'lesson': 0.0,
     'lessons': 0.0,
     'let': 0.0,
     'level': 0.0,
     'lib': 0.0,
     'libraries': 0.0,
     'library': 0.0,
     'like': 0.0,
     'likely': 0.0,
     'line': 0.0,
     'linear': 0.0,
     'lines': 0.0,
     'link': 0.0,
     'links': 0.0,
     'linux': 0.0,
     'list': 0.0,
     'listed': 0.0,
     'litvinov': 0.0,
     'live': 0.0,
     'll': 0.0,
     'load': 0.0,
     'loaded': 0.0,
     'loading': 0.0,
     'local': 0.0,
     'localhost': 0.0,
     'locally': 0.0,
     'located': 0.0,
     'location': 0.0,
     'locations': 0.0,
     'lock': 0.0,
     'log': 0.0,
     'logged': 0.0,
     'login': 0.0,
     'logistic': 0.0,
     'logs': 0.0,
     'long': 0.0,
     'longer': 0.0,
     'look': 0.0,
     'looking': 0.0,
     'looks': 0.0,
     'loop': 0.0,
     'lot': 0.0,
     'lower': 0.0,
     'ls': 0.0,
     'lukafiardi': 0.0,
     'mac': 0.0,
     'machine': 0.0,
     'macos': 0.0,
     'mage': 0.0,
     'magic': 0.0,
     'main': 0.0,
     'make': 0.0,
     'makes': 0.0,
     'making': 0.0,
     'manage': 0.0,
     'manually': 0.0,
     'map': 0.0,
     'marcos': 0.0,
     'marcosmjd': 0.0,
     'martin': 0.0,
     'master': 0.0,
     'match': 0.0,
     'matplotlib': 0.0,
     'matrix': 0.0,
     'max': 0.0,
     'maybe': 0.0,
     'md': 0.0,
     'mean': 0.0,
     'mean_squared_error': 0.0,
     'means': 0.0,
     'media': 0.0,
     'median_house_value': 0.0,
     'memoona': 0.0,
     'memory': 0.0,
     'mentioned': 0.0,
     'merge': 0.0,
     'message': 0.0,
     'metadata': 0.0,
     'method': 0.0,
     'metrics': 0.0,
     'microsoft': 0.0,
     'midterm': 0.0,
     'minutes': 0.0,
     'missing': 0.0,
     'mkdir': 0.0,
     'ml': 0.0,
     'mlbookcamp': 0.0,
     'mlflow': 0.0,
     'mlops': 0.0,
     'mode': 0.0,
     'model': 0.0,
     'models': 0.0,
     'modified': 0.0,
     'modify': 0.0,
     'module': 0.0,
     'modulenotfounderror': 0.0,
     'modules': 0.0,
     'month': 0.0,
     'mount': 0.0,
     'moved': 0.0,
     'multiple': 0.0,
     'multiplication': 0.0,
     'mélanie': 0.0,
     'named': 0.0,
     'names': 0.0,
     'nan': 0.0,
     'nano': 0.0,
     'native': 0.0,
     'navigate': 0.0,
     'ndarray': 0.0,
     'necessary': 0.0,
     'need': 0.0,
     'needed': 0.0,
     'needs': 0.0,
     'negative': 0.0,
     'net': 0.0,
     'network': 0.0,
     'networks': 0.0,
     'new': 0.0,
     'newly': 0.0,
     'non': 0.0,
     'normal': 0.0,
     'note': 0.0,
     'notebook': 0.0,
     'notebooks': 0.0,
     'notes': 0.0,
     'notice': 0.0,
     'np': 0.0,
     'nukta': 0.0,
     'null': 0.0,
     'number': 0.0,
     'numeric': 0.0,
     'numerical': 0.0,
     'numpy': 0.0,
     'ny': 0.0,
     'ny_taxi': 0.0,
     'ny_taxi_postgres_data': 0.0,
     'nyc': 0.0,
     'object': 0.0,
     'observations': 0.0,
     'occur': 0.0,
     'occurred': 0.0,
     'occurs': 0.0,
     'odimegwu': 0.0,
     'office': 0.0,
     'official': 0.0,
     'ok': 0.0,
     'old': 0.0,
     'older': 0.0,
     'olga': 0.0,
     'onehotencoder': 0.0,
     'ones': 0.0,
     'open': 0.0,
     'operation': 0.0,
     'opt': 0.0,
     'option': 0.0,
     'optional': 0.0,
     'options': 0.0,
     'order': 0.0,
     'org': 0.0,
     'origin': 0.0,
     'original': 0.0,
     'os': 0.0,
     'output': 0.0,
     'outputs': 0.0,
     'overfitting': 0.0,
     'owner': 0.0,
     'package': 0.0,
     'packages': 0.0,
     'page': 0.0,
     'pandas': 0.0,
     'parameter': 0.0,
     'parameters': 0.0,
     'params': 0.0,
     'parquet': 0.0,
     'parse': 0.0,
     'particular': 0.0,
     'particularly': 0.0,
     'pass': 0.0,
     'passed': 0.0,
     'passing': 0.0,
     'password': 0.0,
     'paste': 0.0,
     'pastor': 0.0,
     'path': 0.0,
     'paths': 0.0,
     'pay': 0.0,
     'payment_type': 0.0,
     'pc': 0.0,
     'pd': 0.0,
     'peer': 0.0,
     'perform': 0.0,
     'performance': 0.0,
     'period': 0.0,
     'permission': 0.0,
     'permissions': 0.0,
     'pg': 0.0,
     'pgadmin': 0.0,
     'pgadmin4': 0.0,
     'pgadmin_default_email': 0.0,
     'pgadmin_default_password': 0.0,
     'pgcli': 0.0,
     'pgdatabase': 0.0,
     'pick': 0.0,
     'pickup_datetime': 0.0,
     'pip': 0.0,
     'pipeline': 0.0,
     'pipenv': 0.0,
     'pipfile': 0.0,
     'pl3mmuxubc_hihxl5ji8t4o6lpaophaclr': 0.0,
     'place': 0.0,
     'platform': 0.0,
     'platforms': 0.0,
     'plt': 0.0,
     'point': 0.0,
     'points': 0.0,
     'port': 0.0,
     'ports': 0.0,
     'position': 0.0,
     'possible': 0.0,
     'post': 0.0,
     'postgres': 0.0,
     'postgres_db': 0.0,
     'postgres_password': 0.0,
     'postgres_user': 0.0,
     'postgresql': 0.0,
     'powershell': 0.0,
     'pq': 0.0,
     'practice': 0.0,
     'pre': 0.0,
     'precision': 0.0,
     'predict': 0.0,
     'prediction': 0.0,
     'predictions': 0.0,
     'prefect': 0.0,
     'prefer': 0.0,
     'present': 0.0,
     'press': 0.0,
     'pretty': 0.0,
     'prevent': 0.0,
     'previous': 0.0,
     'previously': 0.0,
     'print': 0.0,
     'private': 0.0,
     'probably': 0.0,
     'problem': 0.0,
     'problems': 0.0,
     'proceed': 0.0,
     'process': 0.0,
     'processes': 0.0,
     'production': 0.0,
     'profile': 0.0,
     'program': 0.0,
     'programs': 0.0,
     'progress': 0.0,
     'project': 0.0,
     'project_id': 0.0,
     'projects': 0.0,
     'prompt': 0.0,
     'properly': 0.0,
     'protobuf': 0.0,
     'provide': 0.0,
     'provided': 0.0,
     'provides': 0.0,
     'ps': 0.0,
     'psql': 0.0,
     'psycopg2': 0.0,
     'pubhtml': 0.0,
     'public': 0.0,
     'pull': 0.0,
     'pulocationid': 0.0,
     'purpose': 0.0,
     'push': 0.0,
     'pwd': 0.0,
     'py': 0.0,
     'py4j': 0.0,
     'pyarrow': 0.0,
     'pyspark': 0.0,
     'python': 0.0,
     'python3': 0.0,
     'pythonpath': 0.0,
     'queries': 0.0,
     'query': 0.0,
     'question': 0.0,
     'questions': 0.0,
     'quinn': 0.0,
     'quite': 0.0,
     'quotes': 0.0,
     'radikal': 0.0,
     'raised': 0.0,
     'ram': 0.0,
     'ran': 0.0,
     'random': 0.0,
     'random_state': 0.0,
     'range': 0.0,
     'raw': 0.0,
     'read': 0.0,
     'read_csv': 0.0,
     'read_parquet': 0.0,
     'reading': 0.0,
     'ready': 0.0,
     'real': 0.0,
     'really': 0.0,
     'reason': 0.0,
     'recall': 0.0,
     'receive': 0.0,
     'recent': 0.0,
     'recognized': 0.0,
     'recommend': 0.0,
     'recommended': 0.0,
     'record': 0.0,
     'recorded': 0.0,
     'records': 0.0,
     'recreate': 0.0,
     'refer': 0.0,
     'reference': 0.0,
     'regarding': 0.0,
     'region': 0.0,
     'register': 0.0,
     'registry': 0.0,
     'regression': 0.0,
     'regularization': 0.0,
     'reinstall': 0.0,
     'related': 0.0,
     'releases': 0.0,
     'remember': 0.0,
     'remote': 0.0,
     'remove': 0.0,
     'removed': 0.0,
     'removing': 0.0,
     'rename': 0.0,
     'replace': 0.0,
     'replaced': 0.0,
     ...}



We see most of the term values are 0.0, since those words don't appear in the query.  The word 'course' does appear in the query, so it has a value ```'course': 0.38148200594064524```.


```python
doc_dict = dict(zip(names, X.toarray()[0]))
doc_dict
```




    {'01': 0.0,
     '02': 0.0,
     '03': 0.0,
     '04': 0.0,
     '05': 0.0,
     '06': 0.0,
     '09': 0.0,
     '10': 0.0,
     '100': 0.0,
     '11': 0.0,
     '12': 0.0,
     '127': 0.0,
     '13': 0.0,
     '14': 0.0,
     '15': 0.0,
     '16': 0.0,
     '17': 0.0,
     '19': 0.0,
     '1st': 0.0,
     '20': 0.0,
     '2019': 0.0,
     '2020': 0.0,
     '2021': 0.0,
     '2022': 0.0,
     '2023': 0.0,
     '2024': 0.1414957090150672,
     '21': 0.0,
     '22': 0.0,
     '24': 0.0,
     '25': 0.0,
     '2pacx': 0.0,
     '30': 0.0,
     '35': 0.0,
     '403': 0.0,
     '42': 0.0,
     '50': 0.0,
     '5000': 0.0,
     '5431': 0.0,
     '5432': 0.0,
     '60': 0.0,
     '600': 0.0,
     '7077': 0.0,
     '80': 0.0,
     '8080': 0.0,
     '8888': 0.0,
     '9696': 0.0,
     'abhijit': 0.0,
     'able': 0.0,
     'abolade': 0.0,
     'absolute': 0.0,
     'accept': 0.0,
     'access': 0.0,
     'accordingly': 0.0,
     'account': 0.0,
     'accuracy': 0.0,
     'action': 0.0,
     'activate': 0.0,
     'actual': 0.0,
     'actually': 0.0,
     'add': 0.0,
     'added': 0.0,
     'adding': 0.0,
     'addition': 0.0,
     'additional': 0.0,
     'additionally': 0.0,
     'address': 0.0,
     'admin': 0.0,
     'advani': 0.0,
     'ahmed': 0.0,
     'ai': 0.0,
     'airflow': 0.0,
     'alejandro': 0.0,
     'alena': 0.0,
     'alex': 0.0,
     'alexey': 0.0,
     'alexeygrigorev': 0.0,
     'allocate': 0.0,
     'allow': 0.0,
     'allowed': 0.0,
     'allowing': 0.0,
     'allows': 0.0,
     'alpha': 0.0,
     'alternative': 0.0,
     'alternatively': 0.0,
     'amazon': 0.0,
     'amazonaws': 0.0,
     'amd64': 0.0,
     'aminat': 0.0,
     'anaconda': 0.0,
     'anaconda3': 0.0,
     'analytics': 0.0,
     'anand': 0.0,
     'andrii': 0.0,
     'ans': 0.0,
     'answer': 0.0,
     'answered': 0.0,
     'answers': 0.0,
     'apache': 0.0,
     'api': 0.0,
     'aponte': 0.0,
     'app': 0.0,
     'appear': 0.0,
     'appears': 0.0,
     'append': 0.0,
     'application': 0.0,
     'apply': 0.0,
     'appname': 0.0,
     'approach': 0.0,
     'apt': 0.0,
     'architecture': 0.0,
     'archives': 0.0,
     'args': 0.0,
     'argument': 0.0,
     'array': 0.0,
     'arrays': 0.0,
     'article': 0.0,
     'asia': 0.0,
     'ask': 0.0,
     'asked': 0.13972161000211597,
     'asks': 0.0,
     'assigned': 0.0,
     'astype': 0.0,
     'attempting': 0.0,
     'attribute': 0.0,
     'attributeerror': 0.0,
     'auc': 0.0,
     'auth': 0.0,
     'authenticate': 0.0,
     'authentication': 0.0,
     'auto': 0.0,
     'automatically': 0.0,
     'available': 0.0,
     'average': 0.0,
     'avila': 0.0,
     'avoid': 0.0,
     'aws': 0.0,
     'azure': 0.0,
     'backend': 0.0,
     'bad': 0.0,
     'bar': 0.0,
     'base': 0.0,
     'based': 0.0,
     'bash': 0.0,
     'bashrc': 0.0,
     'basically': 0.0,
     'batch': 0.0,
     'best': 0.0,
     'better': 0.0,
     'bhaskar': 0.0,
     'bhatia': 0.0,
     'big': 0.0,
     'bigquery': 0.0,
     'billing': 0.0,
     'bin': 0.0,
     'binary': 0.0,
     'bind': 0.0,
     'bit': 0.0,
     'blob': 0.0,
     'block': 0.0,
     'blog': 0.0,
     'box': 0.0,
     'bq': 0.0,
     'branch': 0.0,
     'brew': 0.0,
     'brigida': 0.0,
     'browser': 0.0,
     'bucket': 0.0,
     'build': 0.0,
     'builder': 0.0,
     'building': 0.0,
     'built': 0.0,
     'calculate': 0.0,
     'calderin': 0.0,
     'called': 0.0,
     'calling': 0.0,
     'capstone': 0.0,
     'capture': 0.1666836193411991,
     'care': 0.0,
     'careful': 0.0,
     'case': 0.0,
     'cast': 0.0,
     'cat': 0.0,
     'categorical': 0.0,
     'cause': 0.0,
     'caused': 0.0,
     'causing': 0.0,
     'cd': 0.0,
     'cell': 0.0,
     'certificate': 0.0,
     'chakraborty': 0.0,
     'change': 0.0,
     'changed': 0.0,
     'changes': 0.0,
     'changing': 0.0,
     'channel': 0.27944322000423194,
     'check': 0.0,
     'checking': 0.0,
     'chizhmak': 0.0,
     'chmod': 0.0,
     'choice': 0.0,
     'choose': 0.0,
     'chown': 0.0,
     'churn': 0.0,
     'ci': 0.0,
     'class': 0.0,
     'classes': 0.0,
     'classification': 0.0,
     'clear': 0.0,
     'cli': 0.0,
     'click': 0.0,
     'client': 0.0,
     'clone': 0.0,
     'close': 0.0,
     'closed': 0.0,
     'closest': 0.0,
     'clothing': 0.0,
     'cloud': 0.0,
     'cloudfront': 0.0,
     'club': 0.1555377914634961,
     'cluster': 0.0,
     'cmd': 0.0,
     'code': 0.0,
     'codespaces': 0.0,
     'coding': 0.0,
     'cohort': 0.0,
     'cohorts': 0.0,
     'col': 0.0,
     'colab': 0.0,
     'column': 0.0,
     'columns': 0.0,
     'com': 0.0,
     'command': 0.0,
     'commands': 0.0,
     'commit': 0.0,
     'common': 0.0,
     'community': 0.0,
     'compatibility': 0.0,
     'compatible': 0.0,
     'complete': 0.0,
     'completed': 0.0,
     'compose': 0.0,
     'compute': 0.0,
     'computer': 0.0,
     'conda': 0.0,
     'config': 0.0,
     'configuration': 0.0,
     'configure': 0.0,
     'configured': 0.0,
     'confirm': 0.0,
     'connect': 0.0,
     'connecting': 0.0,
     'connection': 0.0,
     'connector': 0.0,
     'consider': 0.0,
     'consistent': 0.0,
     'console': 0.0,
     'contain': 0.0,
     'container': 0.0,
     'containers': 0.0,
     'contains': 0.0,
     'content': 0.0,
     'contents': 0.0,
     'context': 0.0,
     'continue': 0.0,
     'control': 0.0,
     'convert': 0.0,
     'converted': 0.0,
     'copy': 0.0,
     'core': 0.0,
     'correct': 0.0,
     'correctly': 0.0,
     'correlation': 0.0,
     'corresponding': 0.0,
     'couldn': 0.0,
     'course': 0.5102333963686945,
     'cp': 0.0,
     'cpu': 0.0,
     'create': 0.0,
     'create_engine': 0.0,
     'created': 0.0,
     'creates': 0.0,
     'creating': 0.0,
     'creation': 0.0,
     'credentials': 0.0,
     'credit': 0.0,
     'crucial': 0.0,
     'csv': 0.0,
     'ctrl': 0.0,
     'curl': 0.0,
     'current': 0.0,
     'custom': 0.0,
     'd37ci6vzurychx': 0.0,
     'daemon': 0.0,
     'daniel': 0.0,
     'dashboard': 0.0,
     'data': 0.0,
     'database': 0.0,
     'dataframe': 0.0,
     'dataframes': 0.0,
     'dataproc': 0.0,
     'dataset': 0.0,
     'datasets': 0.0,
     'datatalks': 0.1555377914634961,
     'datatalksclub': 0.0,
     'datatype': 0.0,
     'date': 0.0,
     'datetime': 0.0,
     'david': 0.0,
     'day': 0.1555377914634961,
     'db': 0.0,
     'dbt': 0.0,
     'dbt_project': 0.0,
     'deadline': 0.0,
     'dealing': 0.0,
     'debian': 0.0,
     'debug': 0.0,
     'deep': 0.0,
     'def': 0.0,
     'default': 0.0,
     'define': 0.0,
     'defined': 0.0,
     'definition': 0.0,
     'delete': 0.0,
     'deleted': 0.0,
     'deleting': 0.0,
     'denied': 0.0,
     'dependencies': 0.0,
     'dependency': 0.0,
     'deploy': 0.0,
     'deploying': 0.0,
     'deployment': 0.0,
     'deprecated': 0.0,
     'description': 0.0,
     'desktop': 0.1349976193401806,
     'detailed': 0.0,
     'details': 0.0,
     'dev': 0.0,
     'development': 0.0,
     'deviation': 0.0,
     'device': 0.0,
     'df': 0.0,
     'df_train': 0.0,
     'dict': 0.0,
     'dictionary': 0.0,
     'dictvectorizer': 0.0,
     'did': 0.0,
     'didn': 0.0,
     'difference': 0.0,
     'different': 0.0,
     'dino': 0.0,
     'dir': 0.0,
     'directly': 0.0,
     'directory': 0.0,
     'disk': 0.0,
     'distribution': 0.0,
     'doc': 0.0,
     'docker': 0.0,
     'dockerfile': 0.0,
     'docs': 0.0,
     'document': 0.13972161000211597,
     'documentation': 0.0,
     'does': 0.0,
     'doesn': 0.0,
     'doing': 0.0,
     'dolocationid': 0.0,
     'don': 0.11069178449656153,
     'dot': 0.0,
     'double': 0.0,
     'downgrade': 0.0,
     'download': 0.0,
     'downloaded': 0.0,
     'downloading': 0.0,
     'downloads': 0.0,
     'dpage': 0.0,
     'dragon': 0.0,
     'drive': 0.0,
     'drop': 0.0,
     'dropoff_datetime': 0.0,
     'dropping': 0.0,
     'dt': 0.0,
     'dtc': 0.0,
     'duration': 0.0,
     'dv': 0.0,
     'easier': 0.0,
     'easiest': 0.0,
     'easily': 0.0,
     'ec2': 0.0,
     'echo': 0.0,
     'ecr': 0.0,
     'edit': 0.0,
     'editing': 0.0,
     'editor': 0.0,
     'email': 0.0,
     'en': 0.0,
     'enable': 0.0,
     'enabled': 0.0,
     'encode': 0.0,
     'encoding': 0.0,
     'encounter': 0.0,
     'encountered': 0.0,
     'end': 0.0,
     'endpoint': 0.0,
     'engine': 0.0,
     'engineering': 0.0,
     'ensure': 0.0,
     'enter': 0.0,
     'entire': 0.0,
     'entries': 0.0,
     'env': 0.0,
     'environ': 0.0,
     'environment': 0.0,
     'environments': 0.0,
     'equal': 0.0,
     'erick': 0.0,
     'error': 0.0,
     'errors': 0.0,
     'escape': 0.0,
     'especially': 0.0,
     'eu': 0.0,
     'europe': 0.0,
     'evaluate': 0.0,
     'evaluated': 0.0,
     'evaluation': 0.0,
     'exactly': 0.0,
     'example': 0.0,
     'exception': 0.0,
     'exe': 0.0,
     'exec': 0.0,
     'executable': 0.0,
     'execute': 0.0,
     'executed': 0.0,
     'executing': 0.0,
     'execution': 0.0,
     'exist': 0.0,
     'existing': 0.0,
     'exists': 0.0,
     'exit': 0.0,
     'expected': 0.0,
     'experiment': 0.0,
     'explanation': 0.0,
     'explicitly': 0.0,
     'export': 0.0,
     'extension': 0.0,
     'external': 0.0,
     'extra': 0.0,
     'extract': 0.0,
     'face': 0.0,
     'faced': 0.0,
     'fact': 0.0,
     'fact_trips': 0.0,
     'fail': 0.0,
     'failed': 0.0,
     'failing': 0.0,
     'fails': 0.0,
     'false': 0.0,
     'faq': 0.0,
     'fatal': 0.0,
     'feature': 0.0,
     'features': 0.0,
     'fetch': 0.0,
     'fhv': 0.0,
     'fhv_tripdata_2019': 0.0,
     'field': 0.0,
     'figure': 0.0,
     'file': 0.0,
     'files': 0.0,
     'filesystem': 0.0,
     'filter': 0.0,
     'final': 0.0,
     'finally': 0.0,
     'findspark': 0.0,
     'fine': 0.0,
     'fit': 0.0,
     'fix': 0.0,
     'fixed': 0.0,
     'flag': 0.0,
     'flask': 0.0,
     'float': 0.0,
     'float64': 0.0,
     'flow': 0.0,
     'folder': 0.0,
     'folders': 0.0,
     'follow': 0.0,
     'followed': 0.0,
     'following': 0.0,
     'follows': 0.0,
     'forbidden': 0.0,
     'force': 0.0,
     'forget': 0.15264153689277019,
     'forgot': 0.0,
     'form': 0.0,
     'format': 0.0,
     'forms': 0.0,
     'forward': 0.0,
     'fouesnard': 0.0,
     'free': 0.0,
     'fresh': 0.0,
     'fs': 0.0,
     'function': 0.0,
     'functions': 0.0,
     'future': 0.0,
     'gateway': 0.0,
     'gcloud': 0.0,
     'gcp': 0.0,
     'gcs': 0.0,
     'generate': 0.0,
     'generated': 0.0,
     'generator': 0.0,
     'george': 0.0,
     'get_dummies': 0.0,
     'get_feature_names_out': 0.0,
     'getdbt': 0.0,
     'getorcreate': 0.0,
     'gets': 0.0,
     'getting': 0.0,
     'git': 0.0,
     'github': 0.0,
     'githubusercontent': 0.0,
     'gitignore': 0.0,
     'given': 0.0,
     'gives': 0.0,
     'gmail': 0.0,
     'going': 0.0,
     'good': 0.0,
     'google': 0.09876050335844634,
     'google_application_credentials': 0.0,
     'googleapi': 0.0,
     'googleapis': 0.0,
     'got': 0.0,
     'gpu': 0.0,
     'great': 0.0,
     'green': 0.0,
     'grigorev': 0.0,
     'group': 0.0,
     'gs': 0.0,
     'guide': 0.0,
     'gunicorn': 0.0,
     'gz': 0.0,
     'gzip': 0.0,
     'hadoop': 0.0,
     'hadoop3': 0.0,
     'hand': 0.0,
     'handle': 0.0,
     'happen': 0.0,
     'happens': 0.0,
     'haven': 0.0,
     'having': 0.0,
     'head': 0.0,
     'help': 0.0,
     'helped': 0.0,
     'helps': 0.0,
     'high': 0.0,
     'higher': 0.0,
     'highly': 0.0,
     'home': 0.0,
     'homebrew': 0.0,
     'homework': 0.0,
     'homeworks': 0.0,
     'hope': 0.0,
     'host': 0.0,
     'hostname': 0.0,
     'hot': 0.0,
     'hours': 0.1414957090150672,
     'housing': 0.0,
     'hrithik': 0.0,
     'html': 0.0,
     'http': 0.0,
     'https': 0.0,
     'humberto': 0.0,
     'hw': 0.0,
     'hws': 0.0,
     'iam': 0.0,
     'icon': 0.0,
     'id': 0.0,
     'ide': 0.0,
     'idea': 0.0,
     'ignore': 0.0,
     'image': 0.0,
     'images': 0.0,
     'immediately': 0.0,
     'import': 0.0,
     'importance': 0.0,
     'important': 0.0,
     'importerror': 0.0,
     'importing': 0.0,
     'include': 0.0,
     'included': 0.0,
     'including': 0.0,
     'increase': 0.0,
     'index': 0.0,
     'info': 0.0,
     'information': 0.0,
     'init': 0.0,
     'initial': 0.0,
     'input': 0.0,
     'inside': 0.0,
     'install': 0.0,
     'installation': 0.0,
     'installed': 0.0,
     'installing': 0.0,
     'instance': 0.0,
     'instances': 0.0,
     'instead': 0.0,
     'instruction': 0.0,
     'instructions': 0.0,
     'insufficient': 0.0,
     'int': 0.0,
     'int64': 0.0,
     'integer': 0.0,
     'interesting': 0.0,
     'interface': 0.0,
     'internal': 0.0,
     'internet': 0.0,
     'interpreter': 0.0,
     'intro': 0.0,
     'invalid': 0.0,
     'io': 0.0,
     'ip': 0.0,
     'ipynb': 0.0,
     'isn': 0.0,
     'issue': 0.0,
     'issues': 0.0,
     'iteration': 0.0,
     'ivan': 0.0,
     'jan': 0.158775528610509,
     'jar': 0.0,
     'java': 0.0,
     'java_home': 0.0,
     'job': 0.0,
     'jobs': 0.0,
     'join': 0.30528307378554037,
     'json': 0.0,
     'jupyter': 0.0,
     'just': 0.0,
     'kafka': 0.0,
     'kaggle': 0.0,
     'keras': 0.0,
     'kernel': 0.0,
     'key': 0.0,
     'keys': 0.0,
     'kill': 0.0,
     'kind': 0.0,
     'kinesis': 0.0,
     'kniazeva': 0.0,
     'know': 0.0,
     'known': 0.0,
     'konrad': 0.0,
     'krishna': 0.0,
     'kubectl': 0.0,
     'kubernetes': 0.0,
     'kumar': 0.0,
     'kwargs': 0.0,
     'lambda': 0.0,
     'laptop': 0.0,
     'large': 0.0,
     'larkin': 0.0,
     'later': 0.0,
     'latest': 0.0,
     'launch': 0.0,
     'layer': 0.0,
     'layers': 0.0,
     'lead': 0.0,
     'leaderboard': 0.0,
     'learn': 0.0,
     'learning': 0.0,
     'left': 0.0,
     'length': 0.0,
     'lesson': 0.0,
     'lessons': 0.0,
     'let': 0.0,
     'level': 0.0,
     'lib': 0.0,
     'libraries': 0.0,
     'library': 0.0,
     'like': 0.0,
     'likely': 0.0,
     'line': 0.0,
     'linear': 0.0,
     'lines': 0.0,
     'link': 0.1047978728540846,
     'links': 0.0,
     'linux': 0.0,
     'list': 0.0,
     'listed': 0.0,
     'litvinov': 0.0,
     'live': 0.15264153689277019,
     'll': 0.0,
     'load': 0.0,
     'loaded': 0.0,
     'loading': 0.0,
     'local': 0.0,
     'localhost': 0.0,
     'locally': 0.0,
     'located': 0.0,
     'location': 0.0,
     'locations': 0.0,
     'lock': 0.0,
     'log': 0.0,
     'logged': 0.0,
     'login': 0.0,
     'logistic': 0.0,
     'logs': 0.0,
     'long': 0.0,
     'longer': 0.0,
     'look': 0.0,
     'looking': 0.0,
     'looks': 0.0,
     'loop': 0.0,
     'lot': 0.0,
     'lower': 0.0,
     'ls': 0.0,
     'lukafiardi': 0.0,
     'mac': 0.0,
     'machine': 0.0,
     'macos': 0.0,
     'mage': 0.0,
     'magic': 0.0,
     'main': 0.0,
     'make': 0.0,
     'makes': 0.0,
     'making': 0.0,
     'manage': 0.0,
     'manually': 0.0,
     'map': 0.0,
     'marcos': 0.0,
     'marcosmjd': 0.0,
     'martin': 0.0,
     'master': 0.0,
     'match': 0.0,
     'matplotlib': 0.0,
     'matrix': 0.0,
     'max': 0.0,
     'maybe': 0.0,
     'md': 0.0,
     'mean': 0.0,
     'mean_squared_error': 0.0,
     'means': 0.0,
     'media': 0.0,
     'median_house_value': 0.0,
     'memoona': 0.0,
     'memory': 0.0,
     'mentioned': 0.0,
     'merge': 0.0,
     'message': 0.0,
     'metadata': 0.0,
     'method': 0.0,
     'metrics': 0.0,
     'microsoft': 0.0,
     'midterm': 0.0,
     'minutes': 0.0,
     'missing': 0.0,
     'mkdir': 0.0,
     'ml': 0.0,
     'mlbookcamp': 0.0,
     'mlflow': 0.0,
     'mlops': 0.0,
     'mode': 0.0,
     'model': 0.0,
     'models': 0.0,
     'modified': 0.0,
     'modify': 0.0,
     'module': 0.0,
     'modulenotfounderror': 0.0,
     'modules': 0.0,
     'month': 0.0,
     'mount': 0.0,
     'moved': 0.0,
     'multiple': 0.0,
     'multiplication': 0.0,
     'mélanie': 0.0,
     'named': 0.0,
     'names': 0.0,
     'nan': 0.0,
     'nano': 0.0,
     'native': 0.0,
     'navigate': 0.0,
     'ndarray': 0.0,
     'necessary': 0.0,
     'need': 0.0,
     'needed': 0.0,
     'needs': 0.0,
     'negative': 0.0,
     'net': 0.0,
     'network': 0.0,
     'networks': 0.0,
     'new': 0.0,
     'newly': 0.0,
     'non': 0.0,
     'normal': 0.0,
     'note': 0.0,
     'notebook': 0.0,
     'notebooks': 0.0,
     'notes': 0.0,
     'notice': 0.0,
     'np': 0.0,
     'nukta': 0.0,
     'null': 0.0,
     'number': 0.0,
     'numeric': 0.0,
     'numerical': 0.0,
     'numpy': 0.0,
     'ny': 0.0,
     'ny_taxi': 0.0,
     'ny_taxi_postgres_data': 0.0,
     'nyc': 0.0,
     'object': 0.0,
     'observations': 0.0,
     'occur': 0.0,
     'occurred': 0.0,
     'occurs': 0.0,
     'odimegwu': 0.0,
     'office': 0.1555377914634961,
     'official': 0.0,
     'ok': 0.0,
     'old': 0.0,
     'older': 0.0,
     'olga': 0.0,
     'onehotencoder': 0.0,
     'ones': 0.0,
     'open': 0.0,
     'operation': 0.0,
     'opt': 0.0,
     'option': 0.0,
     'optional': 0.0,
     'options': 0.0,
     'order': 0.0,
     'org': 0.0,
     'origin': 0.0,
     'original': 0.0,
     'os': 0.0,
     'output': 0.0,
     'outputs': 0.0,
     'overfitting': 0.0,
     'owner': 0.0,
     'package': 0.0,
     'packages': 0.0,
     'page': 0.0,
     'pandas': 0.0,
     'parameter': 0.0,
     'parameters': 0.0,
     'params': 0.0,
     'parquet': 0.0,
     'parse': 0.0,
     'particular': 0.0,
     'particularly': 0.0,
     'pass': 0.0,
     'passed': 0.0,
     'passing': 0.0,
     'password': 0.0,
     'paste': 0.0,
     'pastor': 0.0,
     'path': 0.0,
     'paths': 0.0,
     'pay': 0.0,
     'payment_type': 0.0,
     'pc': 0.0,
     'pd': 0.0,
     'peer': 0.0,
     'perform': 0.0,
     'performance': 0.0,
     'period': 0.0,
     'permission': 0.0,
     'permissions': 0.0,
     'pg': 0.0,
     'pgadmin': 0.0,
     'pgadmin4': 0.0,
     'pgadmin_default_email': 0.0,
     'pgadmin_default_password': 0.0,
     'pgcli': 0.0,
     'pgdatabase': 0.0,
     'pick': 0.0,
     'pickup_datetime': 0.0,
     'pip': 0.0,
     'pipeline': 0.0,
     'pipenv': 0.0,
     'pipfile': 0.0,
     'pl3mmuxubc_hihxl5ji8t4o6lpaophaclr': 0.0,
     'place': 0.0,
     'platform': 0.0,
     'platforms': 0.0,
     'plt': 0.0,
     'point': 0.0,
     'points': 0.0,
     'port': 0.0,
     'ports': 0.0,
     'position': 0.0,
     'possible': 0.0,
     'post': 0.0,
     'postgres': 0.0,
     'postgres_db': 0.0,
     'postgres_password': 0.0,
     'postgres_user': 0.0,
     'postgresql': 0.0,
     'powershell': 0.0,
     'pq': 0.0,
     'practice': 0.0,
     'pre': 0.0,
     'precision': 0.0,
     'predict': 0.0,
     'prediction': 0.0,
     'predictions': 0.0,
     'prefect': 0.0,
     'prefer': 0.0,
     'present': 0.0,
     'press': 0.0,
     'pretty': 0.0,
     'prevent': 0.0,
     'previous': 0.0,
     'previously': 0.0,
     'print': 0.0,
     'private': 0.0,
     'probably': 0.0,
     'problem': 0.0,
     'problems': 0.0,
     'proceed': 0.0,
     'process': 0.0,
     'processes': 0.0,
     'production': 0.0,
     'profile': 0.0,
     'program': 0.0,
     'programs': 0.0,
     'progress': 0.0,
     'project': 0.0,
     'project_id': 0.0,
     'projects': 0.0,
     'prompt': 0.0,
     'properly': 0.0,
     'protobuf': 0.0,
     'provide': 0.0,
     'provided': 0.0,
     'provides': 0.0,
     'ps': 0.0,
     'psql': 0.0,
     'psycopg2': 0.0,
     'pubhtml': 0.0,
     'public': 0.13096763733447864,
     'pull': 0.0,
     'pulocationid': 0.0,
     'purpose': 0.1666836193411991,
     'push': 0.0,
     'pwd': 0.0,
     'py': 0.0,
     'py4j': 0.0,
     'pyarrow': 0.0,
     'pyspark': 0.0,
     'python': 0.0,
     'python3': 0.0,
     'pythonpath': 0.0,
     'queries': 0.0,
     'query': 0.0,
     'question': 0.0,
     'questions': 0.11453369967598406,
     'quinn': 0.0,
     'quite': 0.0,
     'quotes': 0.0,
     'radikal': 0.0,
     'raised': 0.0,
     'ram': 0.0,
     'ran': 0.0,
     'random': 0.0,
     'random_state': 0.0,
     'range': 0.0,
     'raw': 0.0,
     'read': 0.0,
     'read_csv': 0.0,
     'read_parquet': 0.0,
     'reading': 0.0,
     'ready': 0.0,
     'real': 0.0,
     'really': 0.0,
     'reason': 0.0,
     'recall': 0.0,
     'receive': 0.0,
     'recent': 0.0,
     'recognized': 0.0,
     'recommend': 0.0,
     'recommended': 0.0,
     'record': 0.0,
     'recorded': 0.0,
     'records': 0.0,
     'recreate': 0.0,
     'refer': 0.0,
     'reference': 0.0,
     'regarding': 0.0,
     'region': 0.0,
     'register': 0.3333672386823982,
     'registry': 0.0,
     'regression': 0.0,
     'regularization': 0.0,
     'reinstall': 0.0,
     'related': 0.0,
     'releases': 0.0,
     'remember': 0.0,
     'remote': 0.0,
     'remove': 0.0,
     'removed': 0.0,
     'removing': 0.0,
     'rename': 0.0,
     'replace': 0.0,
     'replaced': 0.0,
     ...}



So what we're looking for is where the document and the query contain the same words, then the document (FAQ) is relevant to the query.  Because each of the terms is weighted (TF-IDF weight), we can now see which documents are more important to this query.  Now we can rank the documents according to this criteria.  To do this, we multiply the weight of the term in the query with the weight of the term in a document, and sum across all of the matching terms (dot product in linear algebra).  This gives us a measure of similarity.  If we do this for all the documents, we can score all the documents and then rank by the relevant score.


```python
# Computing similarity between query and document (cosine similarity)

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
            [0.        ],
            [0.15870689],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.09680922],
            [0.        ],
            [0.        ],
            [0.07529201],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.29986763],
            [0.10520675],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.27447476],
            [0.12828407],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.05163407],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.03156309],
            [0.04914818],
            [0.07138962],
            [0.        ],
            [0.04329773],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.02804374],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.06739038],
            [0.        ],
            [0.00980845],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.05820102],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.05020173],
            [0.        ],
            [0.        ],
            [0.0605701 ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.05101782],
            [0.        ],
            [0.        ],
            [0.04327901],
            [0.        ],
            [0.03089898],
            [0.        ],
            [0.        ],
            [0.05106564],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.08232505],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.02129978],
            [0.        ],
            [0.0308046 ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.16499001],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.03561953],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.07733187],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.12621053],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.01502752],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.05398514],
            [0.        ],
            [0.05115414],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.03869534],
            [0.04112908],
            [0.03728758],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.09064943],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.04233182],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.07784175],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.02591152],
            [0.        ],
            [0.03451997],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.04143129],
            [0.        ],
            [0.05407899],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.22471032],
            [0.04931254],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.03820044],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.02533066],
            [0.04085302],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.07593909],
            [0.0723982 ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.01328652],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.10202387],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.038628  ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.12084703],
            [0.10694586],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.08997023],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.05839441],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.12589674],
            [0.196371  ],
            [0.        ],
            [0.        ],
            [0.06058604],
            [0.35656584],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.16267179],
            [0.        ],
            [0.        ],
            [0.30485152],
            [0.33773583],
            [0.1231099 ],
            [0.13606474],
            [0.26771296],
            [0.        ],
            [0.10835741],
            [0.13944604],
            [0.03865125],
            [0.18083089],
            [0.03179529],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.05501329],
            [0.03482028],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.06558528],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.03200722],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.05236864],
            [0.0581809 ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.10647026],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.03710508],
            [0.        ],
            [0.        ],
            [0.04720167],
            [0.        ],
            [0.        ],
            [0.04364401],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.03714607],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.04808267],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.14400322],
            [0.        ],
            [0.        ],
            [0.03667548],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.06555259],
            [0.13437698],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.05902352],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.01885244],
            [0.03485812],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.05249485],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.06591061],
            [0.        ],
            [0.        ],
            [0.05969678],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.0284637 ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.06244092],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.02243907],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.15333692],
            [0.        ],
            [0.03506481],
            [0.18859024],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.03423508],
            [0.        ],
            [0.        ],
            [0.01588394],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.04924389],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.05098356],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.03294308],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.03026252],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.02958017],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.07023608],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.04109976],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.04644028],
            [0.        ],
            [0.        ],
            [0.03833017],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.0364746 ],
            [0.08415092],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.02442063],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.12810532],
            [0.05637395],
            [0.04826177],
            [0.        ],
            [0.04906345],
            [0.        ],
            [0.26727441],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.11135001],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.06894246],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.0685693 ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.04106944],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.06551955],
            [0.        ],
            [0.07186382],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.06333884],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.12685281],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.07782842],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.02660461],
            [0.06140912],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.05395122],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.02053581],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.02983282],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.03416571],
            [0.03748537],
            [0.        ],
            [0.05747492],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.13556595],
            [0.        ],
            [0.10025904],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.03278609],
            [0.11688021],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.03662924],
            [0.0451592 ],
            [0.        ],
            [0.        ],
            [0.        ],
            [0.        ],
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
           [0.        ],
           [0.15870689],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.09680922],
           [0.        ],
           [0.        ],
           [0.07529201],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.29986763],
           [0.10520675],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.27447476],
           [0.12828407],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.05163407],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.03156309],
           [0.04914818],
           [0.07138962],
           [0.        ],
           [0.04329773],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.02804374],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.06739038],
           [0.        ],
           [0.00980845],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.05820102],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.05020173],
           [0.        ],
           [0.        ],
           [0.0605701 ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.05101782],
           [0.        ],
           [0.        ],
           [0.04327901],
           [0.        ],
           [0.03089898],
           [0.        ],
           [0.        ],
           [0.05106564],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.08232505],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.02129978],
           [0.        ],
           [0.0308046 ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.16499001],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.03561953],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.07733187],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.12621053],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.01502752],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.05398514],
           [0.        ],
           [0.05115414],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.03869534],
           [0.04112908],
           [0.03728758],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.09064943],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.04233182],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.07784175],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.02591152],
           [0.        ],
           [0.03451997],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.04143129],
           [0.        ],
           [0.05407899],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.22471032],
           [0.04931254],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.03820044],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.02533066],
           [0.04085302],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.07593909],
           [0.0723982 ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.01328652],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.10202387],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.038628  ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.12084703],
           [0.10694586],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.08997023],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.05839441],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.12589674],
           [0.196371  ],
           [0.        ],
           [0.        ],
           [0.06058604],
           [0.35656584],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.16267179],
           [0.        ],
           [0.        ],
           [0.30485152],
           [0.33773583],
           [0.1231099 ],
           [0.13606474],
           [0.26771296],
           [0.        ],
           [0.10835741],
           [0.13944604],
           [0.03865125],
           [0.18083089],
           [0.03179529],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.05501329],
           [0.03482028],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.06558528],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.03200722],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.05236864],
           [0.0581809 ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.10647026],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.03710508],
           [0.        ],
           [0.        ],
           [0.04720167],
           [0.        ],
           [0.        ],
           [0.04364401],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.03714607],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.04808267],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.14400322],
           [0.        ],
           [0.        ],
           [0.03667548],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.06555259],
           [0.13437698],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.05902352],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.01885244],
           [0.03485812],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.05249485],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.06591061],
           [0.        ],
           [0.        ],
           [0.05969678],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.0284637 ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.06244092],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.02243907],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.15333692],
           [0.        ],
           [0.03506481],
           [0.18859024],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.03423508],
           [0.        ],
           [0.        ],
           [0.01588394],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.04924389],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.05098356],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.03294308],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.03026252],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.02958017],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.07023608],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.04109976],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.04644028],
           [0.        ],
           [0.        ],
           [0.03833017],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.0364746 ],
           [0.08415092],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.02442063],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.12810532],
           [0.05637395],
           [0.04826177],
           [0.        ],
           [0.04906345],
           [0.        ],
           [0.26727441],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.11135001],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.06894246],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.0685693 ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.04106944],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.06551955],
           [0.        ],
           [0.07186382],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.06333884],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.12685281],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.07782842],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.02660461],
           [0.06140912],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.05395122],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.02053581],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.02983282],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.03416571],
           [0.03748537],
           [0.        ],
           [0.05747492],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.13556595],
           [0.        ],
           [0.10025904],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.03278609],
           [0.11688021],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.03662924],
           [0.0451592 ],
           [0.        ],
           [0.        ],
           [0.        ],
           [0.        ],
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
score
```




    array([0.48049682, 0.        , 0.        , 0.2083882 , 0.        ,
           0.        , 0.        , 0.17557272, 0.        , 0.        ,
           0.        , 0.15870689, 0.        , 0.        , 0.        ,
           0.09680922, 0.        , 0.        , 0.07529201, 0.        ,
           0.        , 0.        , 0.29986763, 0.10520675, 0.        ,
           0.        , 0.        , 0.27447476, 0.12828407, 0.        ,
           0.        , 0.        , 0.        , 0.05163407, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.03156309,
           0.04914818, 0.07138962, 0.        , 0.04329773, 0.        ,
           0.        , 0.        , 0.        , 0.02804374, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.06739038, 0.        , 0.00980845,
           0.        , 0.        , 0.        , 0.        , 0.05820102,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.05020173, 0.        , 0.        , 0.0605701 , 0.        ,
           0.        , 0.        , 0.        , 0.05101782, 0.        ,
           0.        , 0.04327901, 0.        , 0.03089898, 0.        ,
           0.        , 0.05106564, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.08232505, 0.        , 0.        , 0.        , 0.02129978,
           0.        , 0.0308046 , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.16499001, 0.        ,
           0.        , 0.        , 0.03561953, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.07733187, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.12621053, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.01502752, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.05398514, 0.        , 0.05115414, 0.        , 0.        ,
           0.        , 0.        , 0.03869534, 0.04112908, 0.03728758,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.09064943, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.04233182,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.07784175, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.02591152, 0.        , 0.03451997, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.04143129, 0.        , 0.05407899, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.22471032, 0.04931254, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.03820044, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.02533066, 0.04085302, 0.        , 0.        ,
           0.        , 0.        , 0.07593909, 0.0723982 , 0.        ,
           0.        , 0.        , 0.        , 0.01328652, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.10202387, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.038628  , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.12084703, 0.10694586, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.08997023, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.05839441, 0.        , 0.        , 0.        , 0.        ,
           0.12589674, 0.196371  , 0.        , 0.        , 0.06058604,
           0.35656584, 0.        , 0.        , 0.        , 0.        ,
           0.16267179, 0.        , 0.        , 0.30485152, 0.33773583,
           0.1231099 , 0.13606474, 0.26771296, 0.        , 0.10835741,
           0.13944604, 0.03865125, 0.18083089, 0.03179529, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.05501329,
           0.03482028, 0.        , 0.        , 0.        , 0.        ,
           0.06558528, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.03200722, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.05236864, 0.0581809 , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.10647026, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.03710508, 0.        , 0.        ,
           0.04720167, 0.        , 0.        , 0.04364401, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.03714607, 0.        , 0.        ,
           0.        , 0.04808267, 0.        , 0.        , 0.        ,
           0.14400322, 0.        , 0.        , 0.03667548, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.06555259, 0.13437698, 0.        ,
           0.        , 0.        , 0.        , 0.05902352, 0.        ,
           0.        , 0.        , 0.        , 0.01885244, 0.03485812,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.05249485,
           0.        , 0.        , 0.        , 0.06591061, 0.        ,
           0.        , 0.05969678, 0.        , 0.        , 0.        ,
           0.0284637 , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.06244092,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.02243907, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.15333692,
           0.        , 0.03506481, 0.18859024, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.03423508,
           0.        , 0.        , 0.01588394, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.04924389, 0.        , 0.        , 0.        ,
           0.05098356, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.03294308, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.03026252, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.02958017, 0.        , 0.        , 0.        , 0.07023608,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.04109976, 0.        , 0.        , 0.        , 0.04644028,
           0.        , 0.        , 0.03833017, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.0364746 , 0.08415092, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.02442063,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.12810532, 0.05637395,
           0.04826177, 0.        , 0.04906345, 0.        , 0.26727441,
           0.        , 0.        , 0.        , 0.11135001, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.06894246,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.0685693 , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.04106944, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.06551955,
           0.        , 0.07186382, 0.        , 0.        , 0.        ,
           0.06333884, 0.        , 0.        , 0.        , 0.12685281,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.07782842, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.02660461, 0.06140912, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.05395122, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.02053581, 0.        , 0.        , 0.        ,
           0.        , 0.02983282, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.03416571, 0.03748537, 0.        , 0.05747492, 0.        ,
           0.        , 0.        , 0.        , 0.13556595, 0.        ,
           0.10025904, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.03278609, 0.11688021, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.03662924, 0.0451592 , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        ])



To find out the ranking of the documents by importance, we can use numpy's ```argsort()```.  This returns the *index* of the documents, from most importance to least.


```python
import numpy as np
```


```python
# This gives us the the indices of the documents, rather than the scores themselves, and sorts them from lowest to highest.  
# Interested in the 5 highest similarity scores, so the last 5 indices.
np.argsort(score)
```




    array([473, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613,
           614, 615, 616, 617, 618, 620, 621, 622, 624, 625, 627, 601, 628,
           600, 596, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580,
           581, 582, 583, 584, 585, 586, 589, 590, 591, 592, 594, 595, 597,
           569, 629, 632, 665, 666, 667, 668, 670, 671, 673, 674, 675, 676,
           677, 678, 679, 680, 682, 683, 684, 686, 687, 688, 689, 690, 691,
           664, 631, 663, 661, 633, 634, 635, 636, 637, 638, 640, 641, 642,
           643, 644, 645, 646, 647, 649, 650, 651, 652, 653, 655, 658, 659,
           660, 662, 692, 567, 564, 478, 479, 480, 481, 482, 483, 484, 486,
           487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499,
           500, 501, 477, 504, 476, 474, 433, 434, 437, 438, 441, 442, 443,
           444, 446, 447, 453, 459, 460, 461, 462, 463, 466, 467, 468, 469,
           471, 472, 946, 475, 566, 505, 507, 539, 540, 541, 542, 543, 544,
           545, 546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 558,
           559, 560, 562, 563, 538, 506, 537, 535, 508, 509, 510, 512, 513,
           514, 515, 516, 517, 518, 519, 520, 521, 523, 524, 526, 527, 529,
           530, 531, 532, 533, 534, 536, 432, 694, 696, 855, 856, 859, 860,
           861, 862, 863, 864, 865, 866, 868, 869, 870, 871, 872, 873, 874,
           875, 876, 877, 878, 879, 880, 854, 881, 853, 851, 828, 829, 830,
           831, 832, 833, 834, 835, 836, 837, 838, 839, 840, 841, 842, 843,
           844, 845, 846, 847, 848, 849, 850, 852, 827, 882, 884, 919, 920,
           921, 924, 925, 926, 927, 928, 929, 930, 931, 934, 935, 936, 937,
           938, 939, 940, 941, 942, 943, 944, 945, 918, 883, 917, 914, 885,
           886, 887, 888, 889, 890, 892, 893, 894, 895, 897, 898, 899, 900,
           901, 902, 903, 904, 907, 909, 910, 911, 912, 916, 695, 825, 823,
           728, 730, 731, 733, 734, 735, 736, 737, 738, 739, 742, 743, 744,
           745, 746, 747, 748, 750, 751, 752, 753, 754, 755, 727, 756, 726,
           723, 697, 698, 699, 701, 702, 703, 704, 705, 706, 707, 708, 709,
           710, 711, 712, 713, 714, 716, 717, 718, 720, 721, 722, 724, 824,
           757, 763, 796, 797, 798, 799, 800, 801, 802, 803, 805, 807, 808,
           809, 811, 812, 813, 815, 816, 817, 818, 819, 820, 821, 822, 795,
           761, 794, 792, 765, 766, 767, 769, 770, 771, 772, 773, 775, 776,
           777, 778, 779, 781, 782, 783, 784, 785, 786, 788, 789, 790, 791,
           793, 431, 947, 429, 169, 170, 171, 175, 176, 177, 178, 179, 180,
           181, 168, 182, 184, 185, 186, 187, 188, 189, 190, 191, 192, 194,
           183, 166, 164, 163, 137, 138, 139, 140, 141, 143, 144, 145, 146,
           147, 149, 150, 151, 152, 153, 154, 155, 156, 157, 159, 160, 161,
           162, 195, 196, 197, 198, 229, 230, 231, 232, 233, 234, 235, 236,
           237, 238, 239, 240, 242, 244, 245, 246, 247, 248, 249, 250, 251,
           252, 253, 228, 136, 227, 224, 199, 200, 201, 202, 203, 204, 205,
           206, 207, 208, 210, 211, 212, 213, 214, 215, 216, 217, 218, 220,
           221, 222, 223, 225, 254, 135, 133,  38,  42,  44,  45,  46,  47,
            49,  50,  51,  52,  37,  53,  55,  56,  57,  58,  59,  60,  61,
            63,  65,  66,  54,  36,  35,  34,   1,   2,   4,   5,   6,   8,
             9,  10,  12,  13,  14,  16,  17,  19,  20,  21,  24,  25,  26,
            29,  30,  31,  32,  67,  68,  70,  71, 108, 109, 110, 111, 112,
           114, 115, 116, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
           128, 129, 130, 131, 132, 107, 134, 105, 102,  72,  73,  74,  76,
            77,  79,  80,  81,  82,  84,  85,  87,  89,  90,  92,  93,  94,
            95,  96,  97,  98,  99, 101, 103, 255, 219, 257, 354, 355, 356,
           357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367, 368, 369,
           370, 371, 372, 373, 374, 353, 375, 256, 350, 325, 326, 329, 330,
           331, 332, 334, 335, 336, 337, 338, 339, 340, 341, 343, 344, 345,
           346, 347, 348, 349, 351, 324, 376, 378, 407, 408, 409, 410, 412,
           413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425,
           426, 427, 428, 406, 377, 405, 403, 379, 380, 381, 382, 383, 384,
           385, 387, 388, 389, 390, 391, 392, 393, 394, 397, 398, 399, 400,
           401, 402, 404, 323, 352, 303, 296, 297, 298, 271, 299, 270, 269,
           272, 268, 302, 279, 304, 305, 306, 307, 308, 301, 309, 295, 293,
           278, 281, 282, 283, 284, 277, 285, 294, 286, 275, 274, 289, 290,
           291, 292, 273, 276, 310, 280, 266, 318, 317, 316, 263, 261, 315,
           314, 262, 260, 264, 320, 313, 259, 312, 258, 311, 319,  64, 333,
           158, 672, 598, 891, 104, 648, 749, 321, 241, 857,  48, 630, 715,
           896, 700, 106,  88,  39, 458, 485, 922, 693, 905, 669, 243, 465,
           599, 656, 117, 740, 932, 568, 522, 557, 174, 906, 300, 732, 386,
           456, 172, 322, 787, 725, 173, 265, 209,  86,  43, 528, 933, 729,
           525, 561, 760, 762,  40, 681, 288,  75, 685,  83,  91, 167,  33,
           502, 619, 867, 165, 267, 464, 759, 908, 503,  69, 430, 593, 626,
            78, 439, 858, 639, 810, 804, 587, 470, 623,  62, 780, 774, 719,
            41, 806, 328,  18, 327, 142, 826, 226, 100, 741, 411, 193,  15,
           915, 342,  23, 511, 396, 454, 768, 923, 395, 450, 435, 148, 814,
           758,  28, 588, 913, 451, 455, 565, 654,  11, 445, 113,   7, 457,
           657, 436,   3, 287, 764, 452,  27,  22, 448, 449, 440,   0])




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
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0])




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
           0.        , 0.        , 0.        , 0.02804374, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.06739038, 0.        , 0.00980845,
           0.        , 0.        , 0.        , 0.        , 0.05820102,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.05020173, 0.        , 0.        , 0.0605701 , 0.        ,
           0.        , 0.        , 0.        , 0.05101782, 0.        ,
           0.        , 0.04327901, 0.        , 0.03089898, 0.        ,
           0.        , 0.05106564, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.08232505, 0.        , 0.        , 0.        , 0.02129978,
           0.        , 0.0308046 , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.16499001, 0.        ,
           0.        , 0.        , 0.03561953, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.07733187, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.12621053, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.01502752, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.05398514, 0.        , 0.05115414, 0.        , 0.        ,
           0.        , 0.        , 0.03869534, 0.04112908, 0.03728758,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.09064943, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.04233182,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.07784175, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.02591152, 0.        , 0.03451997, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.04143129, 0.        , 0.05407899, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.22471032, 0.04931254, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.03820044, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.02533066, 0.04085302, 0.        , 0.        ,
           0.        , 0.        , 0.07593909, 0.0723982 , 0.        ,
           0.        , 0.        , 0.        , 0.01328652, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.10202387, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.038628  , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.12084703, 0.10694586, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           1.        , 1.08997023, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.05839441, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        ])




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
           0.        , 0.        , 0.        , 0.01402187, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.03369519, 0.        , 0.00490423,
           0.        , 0.        , 0.        , 0.        , 0.02910051,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.02510086, 0.        , 0.        , 0.03028505, 0.        ,
           0.        , 0.        , 0.        , 0.02550891, 0.        ,
           0.        , 0.02163951, 0.        , 0.01544949, 0.        ,
           0.        , 0.02553282, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.04116253, 0.        , 0.        , 0.        , 0.01064989,
           0.        , 0.0154023 , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.082495  , 0.        ,
           0.        , 0.        , 0.01780977, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.03866593, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.06310526, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.00751376, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.02699257, 0.        , 0.02557707, 0.        , 0.        ,
           0.        , 0.        , 0.01934767, 0.02056454, 0.01864379,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.04532471, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.02116591,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.03892087, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.01295576, 0.        , 0.01725998, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.02071565, 0.        , 0.02703949, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.11235516, 0.02465627, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.01910022, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.01266533, 0.02042651, 0.        , 0.        ,
           0.        , 0.        , 0.03796955, 0.0361991 , 0.        ,
           0.        , 0.        , 0.        , 0.00664326, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.05101194, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.019314  , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.06042352, 0.05347293, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           1.        , 3.04498512, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.02919721, 0.        , 0.        , 0.        , 0.        ,
           0.55807263, 0.59330976, 0.49512426, 0.49512426, 2.30621464,
           0.67340718, 2.24620544, 1.96106955, 0.49512426, 0.49512426,
           0.57646015, 2.5933807 , 0.49512426, 3.64755002, 2.51639729,
           0.55667921, 0.56315663, 2.58192257, 3.49512426, 1.71474243,
           1.44939975, 0.51444988, 0.5855397 , 0.01589764, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.02750665,
           0.01741014, 0.        , 0.        , 0.        , 0.        ,
           0.03279264, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.01600361, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.02618432, 0.02909045, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.05323513, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.01855254, 0.        , 0.        ,
           0.02360084, 0.        , 0.        , 0.02182201, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.01857304, 0.        , 0.        ,
           0.        , 0.02404134, 0.        , 0.        , 0.        ,
           0.07200161, 0.        , 0.        , 0.01833774, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.03277629, 0.06718849, 0.        ,
           0.        , 0.        , 0.        , 0.02951176, 0.        ,
           0.        , 0.        , 0.        , 0.00942622, 0.01742906,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.02624743,
           0.        , 0.        , 0.        , 0.0329553 , 0.        ,
           0.        , 0.02984839, 0.        , 0.        , 0.        ,
           0.01423185, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.03122046,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.01121954, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.07666846,
           0.        , 0.01753241, 0.09429512, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.01711754,
           0.        , 0.        , 0.00794197, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.02462194, 0.        , 0.        , 0.        ,
           0.02549178, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.01647154, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.01513126, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.01479008, 0.        , 0.        , 0.        , 0.03511804,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.02054988, 0.        , 0.        , 0.        , 0.02322014,
           0.        , 0.        , 0.01916508, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.0182373 , 0.04207546, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.01221032,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.06405266, 0.02818698,
           0.02413088, 0.        , 0.02453172, 0.        , 0.1336372 ,
           0.        , 0.        , 0.        , 0.05567501, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.03447123,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.03428465, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.02053472, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.03275977,
           0.        , 1.86676623, 0.        , 0.        , 0.        ,
           0.60757601, 2.67416303, 1.96102581, 0.57590658, 0.63933299,
           0.57590658, 0.57590658, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.03891421, 1.50388561, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.0133023 , 0.03070456, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.02697561, 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.01026791, 0.        , 0.        , 0.        ,
           0.        , 0.01491641, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.01708285, 0.01874268, 0.        , 0.02873746, 0.        ,
           0.        , 0.        , 0.        , 0.06778297, 0.        ,
           0.05012952, 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.01639304, 0.0584401 , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.01831462, 0.0225796 , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        ])



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




    array([ 0.0965305 , -0.08218878, -0.099804  , -0.0760841 ,  0.06764978,
           -0.06791426,  0.01472062, -0.1688537 ,  0.198959  ,  0.2983358 ,
            0.02450391,  0.05243722, -0.10675918, -0.04795719,  0.01833377,
           -0.02042856])



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




    array([ 0.05790191, -0.0385412 , -0.05504264, -0.02616081,  0.03963291,
           -0.0673403 ,  0.00506382, -0.11123317,  0.13453799,  0.19988642,
            0.01938119,  0.0539686 , -0.07937754, -0.03665409,  0.02974943,
           -0.02680328])



We'll next compute the cosine similarity.  The principles are still the same, even though it's a dense matrix versus a sparse matrix.


```python
# compute similarity of the document vector and the query vector
np.dot(X_emb[0], Q_emb[0])
```




    0.14338282987813689




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
      <th>764</th>
      <td>machine-learning-zoomcamp</td>
      <td>Projects (Midterm and Capstone)</td>
      <td>What If I submitted only two projects and fail...</td>
      <td>If you have submitted two projects (and peer-r...</td>
    </tr>
    <tr>
      <th>449</th>
      <td>machine-learning-zoomcamp</td>
      <td>General course-related questions</td>
      <td>The course has already started. Can I still jo...</td>
      <td>Yes, you can. You won’t be able to submit some...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - When will the course start?</td>
      <td>The purpose of this document is to capture fre...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - Can I still join the course after the...</td>
      <td>Yes, even if you don't register, you're still ...</td>
    </tr>
    <tr>
      <th>436</th>
      <td>machine-learning-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Is it going to be live? When?</td>
      <td>The course videos are pre-recorded, you can st...</td>
    </tr>
    <tr>
      <th>451</th>
      <td>machine-learning-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Can I submit the homework after the due date?</td>
      <td>No, it’s not possible. The form is closed afte...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Certificate - Can I follow the course in a sel...</td>
      <td>No, you can only get a certificate if you fini...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - Can I follow the course after it fini...</td>
      <td>Yes, we will keep all the materials after the ...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Office Hours - I can’t attend the “Office hour...</td>
      <td>Yes! Every “Office Hours” will be recorded and...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Homework - Are late submissions of homework al...</td>
      <td>No, late submissions are not allowed. But if t...</td>
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

    /opt/miniconda3/envs/dtc/lib/python3.13/site-packages/sklearn/decomposition/_nmf.py:1728: ConvergenceWarning: Maximum number of iterations 200 reached. Increase it to improve convergence.
      warnings.warn(





    array([0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        , 0.00050536, 0.        , 0.27060147, 0.        ,
           0.        , 0.        , 0.        , 0.        , 0.        ,
           0.        ])



It's clearer to see now which documents are most important.  We can think about this as topic modeling:  we have 16 different topics, and we can see that topics 7 and 9 are most relevant to this particular document (first document in the list).  


```python
Q = cv.transform([query])
Q_emb = nmf.transform(Q)
Q_emb[0]
```




    array([0.        , 0.00238097, 0.        , 0.        , 0.        ,
           0.        , 0.        , 0.        , 0.18001189, 0.        ,
           0.        , 0.        , 0.        , 0.00241403, 0.        ,
           0.        ])



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
      <th>814</th>
      <td>mlops-zoomcamp</td>
      <td>+-General course questions</td>
      <td>What if my answer is not exactly the same as t...</td>
      <td>Please choose the closest one to your answer. ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - Can I still join the course after the...</td>
      <td>Yes, even if you don't register, you're still ...</td>
    </tr>
    <tr>
      <th>449</th>
      <td>machine-learning-zoomcamp</td>
      <td>General course-related questions</td>
      <td>The course has already started. Can I still jo...</td>
      <td>Yes, you can. You won’t be able to submit some...</td>
    </tr>
    <tr>
      <th>11</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Certificate - Can I follow the course in a sel...</td>
      <td>No, you can only get a certificate if you fini...</td>
    </tr>
    <tr>
      <th>0</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - When will the course start?</td>
      <td>The purpose of this document is to capture fre...</td>
    </tr>
    <tr>
      <th>7</th>
      <td>data-engineering-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Course - Can I follow the course after it fini...</td>
      <td>Yes, we will keep all the materials after the ...</td>
    </tr>
    <tr>
      <th>764</th>
      <td>machine-learning-zoomcamp</td>
      <td>Projects (Midterm and Capstone)</td>
      <td>What If I submitted only two projects and fail...</td>
      <td>If you have submitted two projects (and peer-r...</td>
    </tr>
    <tr>
      <th>451</th>
      <td>machine-learning-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Can I submit the homework after the due date?</td>
      <td>No, it’s not possible. The form is closed afte...</td>
    </tr>
    <tr>
      <th>436</th>
      <td>machine-learning-zoomcamp</td>
      <td>General course-related questions</td>
      <td>Is it going to be live? When?</td>
      <td>The course videos are pre-recorded, you can st...</td>
    </tr>
    <tr>
      <th>395</th>
      <td>data-engineering-zoomcamp</td>
      <td>Project</td>
      <td>How is my capstone project going to be evaluated?</td>
      <td>Each submitted project will be evaluated by 3 ...</td>
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


    tokenizer_config.json:   0%|          | 0.00/48.0 [00:00<?, ?B/s]



    vocab.txt: 0.00B [00:00, ?B/s]



    tokenizer.json: 0.00B [00:00, ?B/s]



    config.json:   0%|          | 0.00/570 [00:00<?, ?B/s]



    model.safetensors:   0%|          | 0.00/440M [00:00<?, ?B/s]





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
from tqdm.auto import tqdm
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
X_text
```




    array([[-0.00456325, -0.11667515,  0.6274717 , ..., -0.03659188,
             0.10031687,  0.02927116],
           [-0.14233607, -0.19853897,  0.28455406, ..., -0.01139052,
            -0.15399775,  0.0953509 ],
           [ 0.19672215, -0.08461291,  0.28200477, ...,  0.11395848,
            -0.06448034, -0.01282625],
           ...,
           [-0.28217435, -0.3332434 ,  0.29785013, ..., -0.3504273 ,
             0.03266073,  0.09537268],
           [-0.4280712 , -0.39468765,  0.30942   , ..., -0.05943285,
            -0.12965178,  0.0788708 ],
           [-0.16892141, -0.25146288,  0.47843295, ..., -0.18535407,
            -0.16108926,  0.27272934]], dtype=float32)



What we did today is to illustrate the main principles of how relevance search works.  The main idea is that we represent all the documents and the query in a vector space--could be bag-of-words space, BERT space, embeddings from LLM, some sort of vector.  Then, we compute the similarity among all the documents in our index and our query.  

In this example, we had less than 1,000 documents, so we can afford to compute the similarity between the query and all the documents.  But if you're Google and you've indexed the entire internet, you can't go compute similarity betwen your query and documents across the entire internet.  

There are two key concepts to address this problem.  For text search, Inverted Index and Posting List are methods/tools used.  Another idea for vector search is LSH: Locality-sensitive hashing.  A good implementation that is pretty fast is Random Projection.  The idea is to take a long embedding and turn it into a bit vector with ones and zeros only.  In this form, we can still compute cosine similarity between the bit vectors, and it is roughly the same as the cosine similarity between the original vectors.  It's a fast way to compute cosine similarity and there are ways to make it even faster. 

Book recommendation:  _Mining of Massive Datasets_
