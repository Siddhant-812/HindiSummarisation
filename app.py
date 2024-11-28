import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from model import ReRanker
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# Load models and tokenizers during app initialization
@st.cache_resource
def load_models():
    # Load the candidate summary generation model (mT5)
    mt5_model_name = "csebuetnlp/mT5_multilingual_XLSum"
    mt5_tokenizer = AutoTokenizer.from_pretrained(mt5_model_name)
    mt5_model = AutoModelForSeq2SeqLM.from_pretrained(mt5_model_name).to("cuda" if torch.cuda.is_available() else "cpu")

    # Load the ranking model (IndicBERT + LoRA)
    ranking_model_name = "ai4bharat/indic-bert"
    ranking_tokenizer = AutoTokenizer.from_pretrained(ranking_model_name)
    ranking_model = ReRanker(ranking_model_name, pad_token_id=ranking_tokenizer.pad_token_id).to("cuda" if torch.cuda.is_available() else "cpu")
    ranking_model.encoder.load_adapter("./output", "default")  # Load the trained LoRA adapter

    return mt5_model, mt5_tokenizer, ranking_model, ranking_tokenizer


def generate_candidate_summaries(input_text, model, tokenizer, num_candidates=4):
    """
    Generate candidate summaries using mT5 model.
    """
    encoded_input = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to("cuda" if torch.cuda.is_available() else "cpu")
    outputs = model.generate(
        input_ids=encoded_input["input_ids"],
        attention_mask=encoded_input["attention_mask"],
        num_return_sequences=num_candidates,
        num_beams=num_candidates,
        num_beam_groups=4,
        no_repeat_ngram_size=2,
        max_length=80,
        diversity_penalty=1.0,
        early_stopping=True,
    )
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_outputs


def rank_summaries(input_text, candidate_summaries, model, tokenizer, device="cuda"):
    """
    Rank candidate summaries using the trained IndicBERT + LoRA ReRanker model.
    """
    encoded_input = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)
    encoded_candidates = tokenizer(candidate_summaries, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)

    with torch.no_grad():
        outputs = model(
            text_id=encoded_input["input_ids"],
            candidate_id=encoded_candidates["input_ids"],
            require_gold=False,
        )
        scores = outputs["score"].squeeze(0)  # Shape: (num_candidates,)
        ranked_indices = torch.argsort(scores, descending=True).tolist()

    ranked_results = [{"summary": candidate_summaries[i], "score": scores[i].item()} for i in ranked_indices]
    return ranked_results


# Streamlit App
def main():
    st.title("Text Summarization and Ranking App")
    st.write("This app generates and ranks summaries based on their similarity scores with the input document.")

    # Input text box
    input_text = st.text_area("Enter the text for summarization:", height=300)

    # Generate and rank summaries when the button is clicked
    if st.button("Generate and Rank Summaries"):
        if not input_text.strip():
            st.warning("Please enter text for summarization!")
            return

        st.write("### Generating candidate summaries...")
        with st.spinner("Generating summaries..."):
            mt5_model, mt5_tokenizer, ranking_model, ranking_tokenizer = load_models()
            candidate_summaries = generate_candidate_summaries(input_text, mt5_model, mt5_tokenizer)

        st.write("### Ranking summaries...")
        with st.spinner("Ranking summaries..."):
            ranked_results = rank_summaries(input_text, candidate_summaries, ranking_model, ranking_tokenizer)

        # Display results
        st.write("### Ranked Summaries with Similarity Scores")
        for idx, result in enumerate(ranked_results, 1):
            st.write(f"**Rank {idx}:**")
            st.write(f"- **Summary:** {result['summary']}")
            st.write(f"- **Score:** {result['score']:.4f}")
            st.markdown("---")


if __name__ == "__main__":
    main()
