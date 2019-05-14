<img src="https://travis-ci.com/chippermist/cs273-sentiment-analysis.svg?branch=master" />

# UC Irvine Drug Sentiment Analysis
###### CS273 - Data & Knowledge Bases

## Description

As medical advancement continues, drugs have become a very important part of human life to help prolong as well as increase the quality of life for a human being. Every year there are hundreds of new medicines being produced to treat multiple conditions and often end up producing an array of side effects. This causes a lot of confusion as even though some drug might have really good reviews from people suffering from a particular condition, someone suffering from another condition taking the same medication might not have a similar experience. 

Using sentiment analysis, we can figure out based on someone's review and condition whether they have positive feedback from the drug hereby determining the effectiveness of the said drug. This can be used by pharmaceutical companies as well as users to optimize their strategy of which drugs will have positive feedback. 


## Dataset 

[UCI ML Drug Review dataset](https://www.kaggle.com/jessicali9530/kuc-hackathon-winter-2018)


## Setup

Since the code is written in Python 3.x, we have set up a virtualenv to make sure all the required modules and settings are preserved.

If you already have `virtualenv` you can skip the first step:
```
pip install virtualenv
```

To set up and run the files use:
```
cd cs273-sentiment-analysis/
virtualenv -p python3 sentiment-analysis
source sentiment-analysis/bin/activate
pip install -r requirements.txt
python -m nltk.downloader all
```
Now you will be in a `virtualenv` running Python 3.x. To deactivate and return back into standard mode, run `deactivate`.


Note:
If you're running on windows replace `source sentiment-analysis/bin/activate` with `venv\Scripts\activate`.
On MacOS if you're having trouble with `import torch` then run `brew install libomp`.

## Team Members

* Chinmay Garg

* Aneesha Mathur
