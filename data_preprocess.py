from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from indicnlp.tokenize.sentence_tokenize import sentence_split
import re, os
import torch
import warnings
from rouge_score import rouge_scorer
from tqdm import tqdm
import pickle
from indicnlp.tokenize import indic_tokenize

warnings.filterwarnings("ignore")
data = load_dataset("csebuetnlp/xlsum","hindi")

model_name = "csebuetnlp/mT5_multilingual_XLSum"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True, lang="hindi")

WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

def diverse_beam_search(text, model, tokenizer, diversity_penalty=1.0, num_beams = 4, num_summary = 4):

    clean_text = []

    for idx, doc in enumerate(text):
        clean_text.append(WHITESPACE_HANDLER(doc))
    
    encoding = tokenizer(
    clean_text,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    output_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_length=84,
    num_beams=num_beams,
    num_beam_groups=4, 
    no_repeat_ngram_size=2,
    num_return_sequences=num_summary, 
    diversity_penalty=diversity_penalty,  
    early_stopping=True,
    )

    batch_size = len(clean_text)
    summaries = []
    for i in range(batch_size):
        batch_output_id = output_ids[i * num_summary : (i + 1) * num_summary]
        summary_texts = [
            tokenizer.decode(batch_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for batch_output in batch_output_id
        ]
        summaries.append(summary_texts)
    print("Done")
    return summaries


def hindi_sentence_tokenize(text):
    return sentence_split(text, lang='hi')

def rank_candidates_by_rogue(article, ref_summary, candidate_summaries):

    ref_summary = "\n".join([sent.lower() for sent in hindi_sentence_tokenize(ref_summary)])
    candidates = [
        "\n".join([sent.lower() for sent in hindi_sentence_tokenize(cand)]) for cand in candidate_summaries
    ]

    scores = []

    for cand in candidates:
        rouge = rouge_scorer.score(ref_summary, cand)
        avg_score = (rouge["rouge1"].fmeasure + rouge["rouge2"].fmeasure + rouge["rougeLsum"].fmeasure) / 3
        scores.append(avg_score)

    ranked_candidates = sorted(zip(candidates, scores), key= lambda x: x[1], reverse=True)
    return ranked_candidates


def process_and_save_summaries(dataset_split, split_name, output_dir, batch_size=8):
    os.makedirs(output_dir, exist_ok=True)
    processed_data = []

    for i in tqdm(range(0, len(dataset_split), batch_size)):
        # Create batch indices
        batch_indices = list(range(i, min(i + batch_size, len(dataset_split))))
        batch = dataset_split.select(batch_indices)
        articles = batch["text"]
        ref_summaries = batch["summary"]

        # Ensure that articles and summaries are lists
        articles = list(articles)
        ref_summaries = list(ref_summaries)

        # Generate candidate summaries for the batch
        candidate_summaries_batch = diverse_beam_search(articles, model, tokenizer)

        # Process each example in the batch
        for j in range(len(batch)):
            example = batch[j]
            ranked_candidates = rank_candidates_by_rogue(
                articles[j], ref_summaries[j], candidate_summaries_batch[j]
            )
            processed_data.append({
                "id": example['id'],
                "article": articles[j],
                "reference_summary": ref_summaries[j],
                "candidates": [cand for cand, _ in ranked_candidates],
                "scores": [score for _, score in ranked_candidates]
            })

    output_file = os.path.join(output_dir, f"{split_name}_ranked_summaries.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(processed_data, f)

    print(f"[INFO] Saved {split_name} data to {output_file}")

output_directory = "ranked_summary/"
train_subset = data['train'].select(range(25000))
process_and_save_summaries(train_subset, "train", output_directory)
process_and_save_summaries(data['validation'], "validation", output_directory)
process_and_save_summaries(data['test'], "test", output_directory)
