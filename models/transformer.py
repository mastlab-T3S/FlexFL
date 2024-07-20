# 手写transformer

import math

import numpy
from datasets import Dataset, load_dataset
from torch.nn.modules.transformer import _get_clones, _get_activation_fn
import numpy as np

from torch.nn import TransformerEncoder, Linear
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_model, max_len, device):
        """
        constructor of sinusoid encoding class

        :param d_model: dimension of model
        :param max_len: max sequence length
        :param device: hardware device setting
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]


class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix
    """

    def __init__(self, vocab_size, d_model):
        """
        class for token embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, out, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, out)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super(Pooler, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, sequence_output):
        # Assuming sequence_output has shape [batch_size, seq_length, hidden_size]

        # # Extract the hidden state of the first token
        # pooled_output,_ = torch.max(sequence_output, dim=1)
        #
        # # Apply linear transformation and activation function
        # pooled_output = self.activation(self.linear(pooled_output))
        sorted_values, indices = torch.topk(sequence_output, k=3, dim=1, largest=True, sorted=False)

        # 提取第一和第二大值的索引
        top1_index = indices[:, 0]
        top2_index = indices[:, 1]
        top3_index = indices[:, 2]

        # 通过索引获取第一和第二大的值
        top1_value = torch.gather(sequence_output, 1, top1_index.unsqueeze(1))
        top2_value = torch.gather(sequence_output, 1, top2_index.unsqueeze(1))
        top3_value = torch.gather(sequence_output, 1, top3_index.unsqueeze(1))

        # 将第一和第二大的值连接起来
        pooled_output = torch.cat([top1_value, top2_value, top3_value], dim=1)

        # 应用线性层进行映射
        pooled_output = self.activation(self.linear(pooled_output))

        return pooled_output


####################################################################################

class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding (sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, out, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn1 = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, out=d_model, drop_prob=drop_prob)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.linear = Linear(d_model, out)

    def forward(self, x, src_mask):
        # 1. compute self attention
        _x = x
        if src_mask is not None:
            x = self.attention(q=x, k=x, v=x, mask=src_mask.unsqueeze(1).unsqueeze(-1))
        else:
            x = self.attention(q=x, k=x, v=x)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        _x = x
        x = self.ffn1(x)

        # 4. add and norm
        x = self.dropout2(x)
        x = self.norm2(x + _x)

        x = self.linear(x)
        return x


class Transformer(nn.Module):
    """
    vocab_size: 词典大小
    d_model: 词向量维度
    nhead: 多头注意力的头数
    num_encoder_layers: encoder层数
    max_len: 句子最大长度
    slim_idx: 保留几层
    scale: 每层的宽度比例
    """

    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_encoder_layers: int = 8,
                 max_len: int = 256, slim_idx: int = 1, scale: float = 1,
                 dropout=0.1):
        super(Transformer, self).__init__()
        assert slim_idx <= num_encoder_layers, "slim_idx must be less than or equal to num_encoder_layers"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scale = scale
        self.slim_idx = slim_idx
        self.num_encoder_layers = num_encoder_layers
        self.embedding = TransformerEmbedding(vocab_size, d_model, max_len, dropout, self.device)
        self.encoder_layers = nn.ModuleList()

        for i in range(1, num_encoder_layers + 1):
            if i < slim_idx:
                self.encoder_layers.append(
                    EncoderLayer(d_model, 4 * d_model, d_model, nhead, dropout)
                )
            elif i == slim_idx:
                self.encoder_layers.append(
                    EncoderLayer(d_model, 4 * d_model, int(d_model * scale), nhead, dropout)
                )
            elif i > slim_idx:
                self.encoder_layers.append(
                    EncoderLayer(int(d_model * scale), 4 * int(d_model * scale), int(d_model * scale), nhead, dropout)
                )
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(3 * int(d_model * scale), d_model)
        self.linear2 = nn.Linear(d_model, 2)
        self.pool = Pooler(int(d_model * scale))

    def forward(self, input, mask=None):
        src = self.embedding(input)
        for i in range(self.num_encoder_layers):
            if i <= self.slim_idx:
                src = self.encoder_layers[i](src, mask)
            else:
                src = self.encoder_layers[i](src, None)
        src = self.pool(src)
        out = src.view(src.size()[0], -1)
        out = self.linear1(out)
        out = self.dropout(out)
        out = self.linear2(out)

        return out


if __name__ == "__main__":
    batch_size = 32


    class Dataset_java_huggingface(Dataset):
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
            return {"input_ids": self.dataz[item], "labels": self.label[item],
                    "attention_mask": self.attention_mask[item]}


    # r"/home/mastlab/.cache/huggingface/datasets/glue/sst2"
    dataset = load_dataset(r"C:\Users\DoubleZeroWater\.cache\huggingface\datasets\glue\sst2")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = Dataset_java_huggingface(dataset, "train", tokenizer, 256)
    test_dataset = Dataset_java_huggingface(dataset, "test", tokenizer, 256)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    num_epochs = 1000
    hidden_size = 256
    num_layers = 3
    lr = 0.0001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(vocab_size=30522, d_model=hidden_size, nhead=8, num_encoder_layers=16, max_len=256,
                        slim_idx=2,
                        scale=0.40625, dropout=0.1)
    model.to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()


    def train(model, iterator, optimizer, criterion):
        model.train()
        total_loss = 0
        cnt = 0
        for batch in iterator:
            optimizer.zero_grad()
            text = torch.Tensor(batch["input_ids"]).to(torch.long).to(device)
            target = torch.Tensor(batch["labels"]).to(torch.long).to(device)
            mask = torch.Tensor(batch["attention_mask"]).to(torch.long).to(device)
            output = model(text, mask)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            cnt += 1

        return total_loss / cnt


    # Evaluation Loop
    def evaluate(model, iterator):
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in iterator:
                text = torch.Tensor(batch["input_ids"]).to(torch.long).to(device)
                target = torch.Tensor(batch["labels"]).to(torch.long).to(device)
                mask = torch.Tensor(batch["attention_mask"]).to(torch.long).to(device)
                output = model(text, mask)
                # 计算预测结果，通常使用 argmax 函数
                predictions = torch.argmax(output, dim=1)
                # 统计预测正确的数量
                correct += (predictions == target).sum().item()
                total += target.size(0)

        accuracy = correct / total
        return accuracy


    # Training and Evaluation
    N_EPOCHS = 1000
    best_valid_loss = float('inf')
    total = sum([param.nelement() for param in model.parameters()])
    print(total / 1e6)
    for epoch in range(N_EPOCHS):
        train_loss = train(model, train_loader, optimizer, criterion)
        test_loss = evaluate(model, test_loader)
        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'Test Acc: {test_loss:.3f}')

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')
