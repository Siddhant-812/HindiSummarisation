# CS-626 (2024) IITB Project
## Hindi Abstractive Text Summarisation
This repository contains the project for the course CS-626 "Speech and Natural Language Processing and the Web" (2024).
The projects implements the paper "SimCLS: A Simple Framework for Contrastive Learning of Abstractive Summarization" published in ACL-2021 for Hindi Abstractie Text Summarisation

## How to run?
data_preprocess.py - script for generating candidate summaries and arranging them based on scores and storing in a pickle file.
model.py - script for defining the model and loss function
main.py - script for training

run the folowing command to train the model
python main.py --data_path ./ranked_summary/train_ranked_summaries.pkl --model_type ai4bharat/indic-bert --batch_size 4 --epochs 3 --output_dir ./output

## Dataset
The Dataset used is the XL-Sum dataset which contains news article and summary pairs for different languages.
We have used the Hindi split of the dataset which contains aroung 88k article-summary pairs.
link: https://huggingface.co/datasets/csebuetnlp/xlsum

## Text Summarisation Model
We have used the pre-trained mT5 model for generating candidate summaries.
We have generated 4 candidate summaries for each article using diverse beam search and arranged them in decending order of the Rouge scores with the reference summaries.
The link to the model is : https://huggingface.co/csebuetnlp/mT5_multilingual_XLSum

## Re-Ranker Model
We fine-tuned the Indic-Bert model (https://huggingface.co/ai4bharat/indic-bert) using LoRA for ranking the candidate summaries based on their scores with the document.
For calculating the candidate score with the article we have trained the model with two different scoring metrics:
i) Cosine Similarity
ii) Euclidean Distance
The model using cosine similarity as a scoring function performed slightly better than the model using euclidean distance as a score metric.

## Training Details
Due to GPU constraints we fine-tuned the model using only a subset of the training data around (25k samples).
The model was trained for 3 epochs with a batch-size of 4 for 6 hours on NVIDIA RTX 2080 GPU.

## Results
![image](https://github.com/user-attachments/assets/c5c7d305-aed3-44b3-ace7-b1d246eb1c62)

## Streamlit App
To demonstrate the working of the model we built a streamlit app that generated the top ranked summaries of the input text based on the similarity score.
![image](https://github.com/user-attachments/assets/3b3047a6-4190-4e95-bc59-2cabef1e128e)
![image](https://github.com/user-attachments/assets/06e0b1bf-418a-42be-a30e-1bf95d2aab28)



