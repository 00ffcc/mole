from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset

class PretrainDataset(IterableDataset):
    def __init__(self, tokenizer, max_length=2049, dataset_name_or_path='pietrolesci/pile-deduped-pythia-preshuffled'):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer.pad_token = tokenizer.eos_token
        self.max_length = max_length
        self.ds = load_dataset(dataset_name_or_path, "detokenized", split='train', streaming=True)
    def __iter__(self):
        for sample in self.ds:
            # 构建输入文本
            text = f"{self.tokenizer.bos_token}{str(sample['text'])}{self.tokenizer.eos_token}"
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding.input_ids.squeeze()
            loss_mask = encoding.attention_mask.squeeze()

            yield input_ids[:-1], input_ids[1:], loss_mask[1:]
    
if __name__ == '__main__':
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m-deduped')
    print(tokenizer.vocab_size)
    ds = PretrainDataset(tokenizer=tokenizer)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    for batch in dl:
        print(batch)
        break