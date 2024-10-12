# 520 project - A news summarizer

News summarization is a significant problem because it helps users efficiently digest vast amounts of information while keeping them informed about current events. With the increasing volume of news articles, a reliable summarization method can improve accessibility and understanding for diverse audiences. The T5 model is an excellent choice for this task due to its versatility and strong performance across various natural language processing tasks, allowing it to generate coherent and contextually relevant summaries that maintain the essence of the original articles. Additionally, its ability to handle different input formats and tasks makes it adaptable for evolving news summarization needs.

## Highlits

1. Python 3.10.15
2. pre-trained model: T5-small, gpt2 (archived)
3. Fine-tuning dataset source: https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail/data
4. Generate summary for news texts input length up to 500 words.

## Build Model

### Data preprocessing
Original raw dataset has 287k rows with two major columns: article and highlights
I processed the dataset first then randomly picked 80k rows for training.

The following data processing action are performed:
1. Remove duplicates
2. Remove all rows that have article longer than 500 words. -> 93k rows left
3. Remove leading/trailing/multiple whitespaces
4. Remove all unwanted chars
   1. Extra punctuation
   2. Special chars e.g. | ~ ^
5. Standardize text, e.g. US’s to US has
6. Randomly pick 80k for training, 4k for evaluation


### Training

In order to train the model locally, make sure using python3.10
and install all the dependencies from the requirements with

```
python -m pip install -r requirenments.txt
```

Use train.py or news_summary_t5_train.ipynb to train the model.
Make sure download and unzip the dataset from the above the into 
the working dir ```./dataset```

## Running
### cloud
The application is built based on streamlit and currently hosted at 
```https://share.streamlit.io/```.

The user needs to prepare some news text. The app allows the user 
to input the following parameters: max_length, top_k and top_p.

### Local 
After the training is done, there are two ways using the model

1. (Highly recommend) Start the streamlit app using
```
streamlit run server.py
```

2. Use ```summarizer.py ```. Make sure replace the input text
in the main function. The project is mainly designed as a streamlit
app. The summarizer script does not take inputs. The user has to
modify the code to be able to tune the parameters.

