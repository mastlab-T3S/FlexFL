import numpy
import torch
from datasets import Dataset


class SST_2_Dataset(Dataset):
    def __init__(self, dataset, elem, tokenizer, max_length):
        self.elem = elem
        self.tokenizer = tokenizer
        self.max_length = max_length
        if self.elem == "train":
            self.dataz = numpy.array(dataset["train"]["text"])
            self.label = torch.Tensor(dataset["train"]["label"])
        elif self.elem == "test":
            self.dataz = numpy.array(dataset["test"]["text"])
            self.label = torch.Tensor(dataset["test"]["label"])
        else:
            self.dataz = numpy.array(dataset["validation"]["text"])
            self.label = torch.Tensor(dataset["validation"]["label"])
        inputs = self.tokenizer(
            self.dataz.tolist(),
            truncation=True,
            padding='max_length',  # 按最大长度填充
            max_length=self.max_length,
            return_tensors='pt'  # 返回 PyTorch 张量
        )
        self.dataz = inputs["input_ids"]
        self.attention_mask = inputs["attention_mask"]
        self.label = self.label.to(torch.long)

    def __len__(self):
        return len(self.dataz)

    def __getitem__(self, item):
        return {"input_ids": self.dataz[item], "labels": self.label[item], "attention_mask": self.attention_mask[item]}