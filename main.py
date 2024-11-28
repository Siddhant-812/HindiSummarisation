import torch
import pickle
import argparse
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from model import ReRanker, RankingLoss
import warnings
from tqdm import tqdm 
warnings.filterwarnings("ignore")


class HindiDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512, max_num=4):
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_num = max_num

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['article']
        candidates = item['candidates'][:self.max_num]
        gold = item['reference_summary']

        text_enc = self.tokenizer(text, max_length=self.max_len, truncation=True, padding='max_length', return_tensors="pt")
        candidates_enc = self.tokenizer(candidates, max_length=self.max_len, truncation=True, padding='max_length', return_tensors="pt")
        gold_enc = self.tokenizer(gold, max_length=self.max_len, truncation=True, padding='max_length', return_tensors="pt")

        return {
            "text_id": text_enc["input_ids"].squeeze(0),
            "candidate_ids": candidates_enc["input_ids"],
            "summary_id": gold_enc["input_ids"].squeeze(0)
        }


def train(args):
    # Load tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    dataset = HindiDataset(args.data_path, tokenizer, max_len=args.max_len, max_num=args.max_num)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Load model
    model = ReRanker(args.model_type, tokenizer.pad_token_id).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch + 1}/{args.epochs}") as pbar:
            for batch in dataloader:
                text_id = batch["text_id"].to(args.device)
                candidate_id = batch["candidate_ids"].to(args.device)
                summary_id = batch["summary_id"].to(args.device)

                outputs = model(text_id, candidate_id, summary_id)
                loss = RankingLoss(outputs["score"], outputs["summary_score"], margin=args.margin, gold_margin=args.gold_margin)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"Loss": total_loss / (pbar.n + 1)})  # Update progress bar with current loss
                pbar.update(1)

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

    # Save the LoRA fine-tuned model
    model.encoder.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to the pickle file containing the dataset.")
    parser.add_argument("--model_type", type=str, default="ai4bharat/indic-bert", help="Pretrained model type.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--max_len", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--max_num", type=int, default=4, help="Maximum number of candidate summaries.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--margin", type=float, default=0.01, help="Margin for candidate loss.")
    parser.add_argument("--gold_margin", type=float, default=0.01, help="Margin for gold summary loss.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (e.g., 'cuda' or 'cpu').")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save the fine-tuned model.")
    args = parser.parse_args()
    train(args)
