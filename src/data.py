"""Dataset loading and preprocessing for MicroMixer-1 training."""

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import torch


class ConversationDataset(Dataset):
    """Dataset for conversational language modeling."""

    def __init__(self, texts: list[str], tokenizer, max_length: int = 128):
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
        self.sequences = []
        for text in texts:
            if not text or not text.strip():
                continue
            token_ids = tokenizer.encode(text)
            if len(token_ids) > max_length:
                token_ids = token_ids[:max_length]
            if len(token_ids) >= 2:
                self.sequences.append(token_ids)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx) -> dict:
        seq = self.sequences[idx]
        seq_len = len(seq)
        pad_len = self.max_length - seq_len
        padded = seq + [self.pad_token_id] * pad_len
        tensor = torch.tensor(padded, dtype=torch.long)
        x = tensor[:-1]
        y = tensor[1:]
        mask = torch.zeros(self.max_length - 1, dtype=torch.long)
        mask[:seq_len - 1] = 1
        return {'input_ids': x, 'labels': y, 'attention_mask': mask}


def load_daily_dialog(max_samples: int = None) -> list[str]:
    try:
        dataset = load_dataset("daily_dialog", split="train", streaming=True)
        texts = []
        for sample in dataset:
            dialog = sample["dialog"]
            joined = " <turn> ".join(dialog)
            texts.append(joined)
            if max_samples is not None and len(texts) >= max_samples:
                break
        return texts
    except Exception as e:
        print(f"Warning: Could not load daily_dialog: {e}")
        return []


def load_tiny_stories(max_samples: int = None) -> list[str]:
    try:
        dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        texts = []
        for sample in dataset:
            text = sample["text"]
            if text and text.strip():
                texts.append(text)
            if max_samples is not None and len(texts) >= max_samples:
                break
        return texts
    except Exception as e:
        print(f"Warning: Could not load TinyStories: {e}")
        return []


def create_dataloader(
    texts: list[str],
    tokenizer,
    max_length: int = 128,
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader:
    dataset = ConversationDataset(texts, tokenizer, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=0,
    )


def load_combined_dataset(
    max_daily_dialog: int = None,
    max_tiny_stories: int = None,
) -> list[str]:
    texts = []
    
    daily = load_daily_dialog(max_daily_dialog)
    if daily:
        texts.extend(daily)
        print(f"  Loaded {len(daily)} DailyDialog samples")
    
    stories = load_tiny_stories(max_tiny_stories)
    if stories:
        texts.extend(stories)
        print(f"  Loaded {len(stories)} TinyStories samples")
    
    if not texts:
        print("  Warning: No datasets loaded, using synthetic data")
        texts = [
            "Hello, how are you today?",
            "I am doing well, thank you for asking!",
            "What is your name?",
            "My name is MicroMixer, nice to meet you!",
            "The weather is beautiful outside.",
            "I love programming in Python.",
            "Machine learning is fascinating.",
            "Let us build something cool together!",
            "This is a simple conversation.",
            "Learning new things is always exciting.",
        ] * 100
    
    return texts
