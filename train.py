import numpy as np
import pandas as pd
import os
import re
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2TokenizerFast, Trainer, TrainingArguments, default_data_collator
from datasets import Dataset, load_metric
import torch
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import contractions

# DataSet Source
# https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail/data

def load_input_dataset_files():
    files = {}
    for dirname, _, filenames in os.walk('./dataset'):
        for filename in filenames:
            fullpath = os.path.join(dirname, filename)
            if filename.split(".")[-1] == "csv":
                files[''.join(filename.split(".")[:-1])] = pd.read_csv(fullpath)
                print(f"Loaded file: {filename}")
    return files

def short_text(df):
    # remove all the rows having text longer than 900 words
    df['article_length'] = df['article'].apply(lambda x: len(x.split()))
    df['summary_length'] = df['highlights'].apply(lambda x: len(x.split()))

    df['article_length'] = pd.to_numeric(df['article_length'], errors='coerce')
    df['summary_length'] = pd.to_numeric(df['summary_length'], errors='coerce')

    return df[(df['article_length'] <= 900) & (df['summary_length'] <= 900)]



def normalize_text(text):
    # Remove leading/trailing whitespace
    text = text.strip()
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    return text

def remove_unwanted_characters(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove excess punctuation (e.g., !!!, ???)
    text = re.sub(r'([!?.])\1+', r'\1', text)
    # Remove other unwanted symbols (customize as needed)
    text = re.sub(r'[|~^]', '', text)
    return text

def expand_contractions_text(text):
    return contractions.fix(text)

def standardize_possessives(text):
    # Example: Convert "US's" to "US is" if appropriate
    # Note: This is context-dependent and should be used carefully
    text = re.sub(r"(\b\w+)'s\b", r"\1 is", text)
    return text

def remove_stopwords(text, stop_words):
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return ' '.join(tokens)

def format_all(df_list):
    for df in df_list:

        df.drop_duplicates(subset=['article', 'highlights'], inplace=True)

        # change it's to it is etc
        df['article'] = df['article'].apply(expand_contractions_text)
        df['highlights'] = df['highlights'].apply(expand_contractions_text)

        df['article'] = df['article'].apply(standardize_possessives)
        df['highlights'] = df['highlights'].apply(standardize_possessives)

        df['article'] = df['article'].apply(normalize_text)
        df['highlights'] = df['highlights'].apply(normalize_text)

        df['article'] = df['article'].apply(remove_unwanted_characters)
        df['highlights'] = df['highlights'].apply(remove_unwanted_characters)

        df['article'] = df['article'].str.lower()
        df['highlights'] = df['highlights'].str.lower()

def add_special_tokens():
    """ Returns GPT2 tokenizer after adding separator and padding tokens """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    special_tokens = {'pad_token':'<|pad|>','sep_token':'<|sep|>'}
    num_add_toks = tokenizer.add_special_tokens(special_tokens)
    return tokenizer

def prepare_input_text(article, summary):
    return f"{article} </s> {summary}"

def generate_input_text_for_all(df_list):
    for df in df_list:
        df['input_text'] = df.apply(lambda x: prepare_input_text(x['article'], x['highlights']), axis=1)

def tokenize_function(df, tokenizer, ignore_idx):
    encodings = tokenizer(df['input_text'],
                          padding='max_length',
                          truncation=True,
                          max_length=800
                          )
    # Create labels, setting padding tokens to -100 for loss calculation
    encodings['labels'] = [[-100 if token == ignore_idx else token for token in input_ids] for input_ids in encodings['input_ids']]
    return encodings

def compute_metrics(eval_pred, tokenizer):
    rouge = load_metric("rouge")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    batch_size = 10  # Adjust based on memory capacity
    rouge_results = []

    # Process ROUGE calculation in smaller batches
    for i in range(0, len(decoded_preds), batch_size):
        batch_preds = decoded_preds[i:i + batch_size]
        batch_labels = decoded_labels[i:i + batch_size]
        result = rouge.compute(predictions=batch_preds, references=batch_labels, use_stemmer=True)
        rouge_results.append(result)

    # Aggregate results
    final_result = {"rouge2": {"fmeasure": np.mean([r["rouge2"].mid.fmeasure for r in rouge_results])}}
    return final_result

def build_train_args(learning_rate,
                     per_device_train_batch_size,
                     per_device_eval_batch_size,
                     num_train_epochs,
                     weight_decay,
                     fp16=True):
    return TrainingArguments(output_dir='./results',
                                evaluation_strategy="steps",
                                eval_steps=1000,
                                learning_rate=learning_rate,
                                per_device_train_batch_size=per_device_train_batch_size,
                                per_device_eval_batch_size=per_device_eval_batch_size,
                                num_train_epochs=num_train_epochs,
                                weight_decay=weight_decay,
                                fp16=fp16,
                                # load_best_model_at_end=True,
                                # metric_for_best_model="rouge2",
                                # greater_is_better=True,
                                remove_unused_columns=False
                            )


def main():
    input_files = load_input_dataset_files()
    raw_train_all_df = input_files['train']
    raw_validation_all_df = input_files['validation']

    nltk.download('stopwords')
    nltk.download('punkt_tab')
    stop_words = set(stopwords.words('english'))

    train_all_df = short_text(raw_train_all_df)
    validation_all_df = short_text(raw_validation_all_df)

    train_df = train_all_df.sample(n=40000, random_state=42)
    validation_df = validation_all_df.sample(n=2000, random_state=42)

    train_df_copy = train_df.copy()
    validation_df_copy = validation_df.copy()

    format_all([train_df_copy, validation_df_copy])

    # Load the GPT-2 tokenizer and model
    # Move the model to GPU if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = add_special_tokens()
    model.resize_token_embeddings(len(tokenizer))
    ignore_idx = tokenizer.pad_token_id
    model.to(device)

    generate_input_text_for_all([train_df_copy, validation_df_copy])

    train_dataset = Dataset.from_pandas(train_df_copy[['input_text']])
    validation_dataset = Dataset.from_pandas(validation_df_copy[['input_text']])

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    validation_dataset = validation_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    validation_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=build_train_args(learning_rate=2e-5,
                              per_device_train_batch_size=6,
                              per_device_eval_batch_size=6,
                              num_train_epochs=3,
                              weight_decay=0.01),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        # compute_metrics=compute_metrics
    )

    trainer.train()

    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')


if __name__ == '__main__':
    main()
