import numpy as np
import pandas as pd
import os
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, default_data_collator, AutoTokenizer
from datasets import Dataset, load_metric
import torch
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

    return df[(df['article_length'] <= 500) & (df['summary_length'] <= 500)]

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
    # Example: Convert "US's" to "US has" if appropriate
    text = re.sub(r"(\b\w+)'s\b", r"\1 is", text)
    return text

def format_all(df_list):
    for df in df_list:
        df.drop_duplicates(subset=['article', 'highlights'], inplace=True)

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


def prepare_input_text(article, summary):
    return f"{article} </s> {summary}"


def tokenize_function(df, tokenizer):
    inputs = df['article']
    model_inputs = tokenizer(inputs, max_length=512, padding='max_length', truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(df['highlights'], max_length=150, padding='max_length', truncation=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

def compute_metrics(eval_pred,tokenizer):
    rouge = load_metric("rouge")
    predictions, labels = eval_pred
    # Decode the predicted token IDs to strings
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Decode the label token IDs to strings
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE scores
    results = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract ROUGE-2 scores
    rouge2 = results["rouge2"]
    return {"rouge2": round(rouge2.fmeasure * 100, 2)}  # Return F1 score as percentage

def build_train_args(learning_rate,
                     per_device_train_batch_size,
                     per_device_eval_batch_size,
                     num_train_epochs,
                     weight_decay,
                     fp16=True,
                     metric="loss"):
    return  TrainingArguments(output_dir='./results',
                                evaluation_strategy="steps",
                                eval_steps=500,
                                learning_rate=learning_rate,
                                per_device_train_batch_size=per_device_train_batch_size,
                                per_device_eval_batch_size=per_device_eval_batch_size,
                                num_train_epochs=num_train_epochs,
                                weight_decay=weight_decay,
                                fp16=fp16,
                                metric_for_best_model=metric,
                                # greater_is_better=True,
                                remove_unused_columns=True,
                                load_best_model_at_end=True
                            )


def main():
    input_files = load_input_dataset_files()
    raw_train_all_df = input_files['train']
    raw_validation_all_df = input_files['validation']

    train_all_df = short_text(raw_train_all_df)
    validation_all_df = short_text(raw_validation_all_df)

    train_df = train_all_df.sample(n=40000, random_state=42)
    validation_df = validation_all_df.sample(n=2000, random_state=42)

    train_df_copy = train_df.copy()
    validation_df_copy = validation_df.copy()

    format_all([train_df_copy, validation_df_copy])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.to(device)

    train_dataset = Dataset.from_pandas(train_df_copy)
    validation_dataset = Dataset.from_pandas(validation_df_copy)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    validation_dataset = validation_dataset.map(tokenize_function, batched=True)

    trainer = Trainer(
        model=model,
        args=build_train_args(learning_rate=5e-6,
                              per_device_train_batch_size=32,
                              per_device_eval_batch_size=32,
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
