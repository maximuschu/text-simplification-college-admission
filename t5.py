import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import cuda


class T5Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text
        self.ctext = self.data.ctext

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus(
            [ctext], max_length=self.source_len, pad_to_max_length=True, return_tensors='pt')
        target = self.tokenizer.batch_encode_plus(
            [text], max_length=self.summ_len, pad_to_max_length=True, return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for _, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        labels = y[:, 1:].clone().detach()
        labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask,
                        decoder_input_ids=y_ids, labels=labels)
        loss = outputs[0]

        if _ % 10 == 0:
            # wandb.log({"Training Loss": loss.item()})
            print(f'Training Loss: {loss.item()}')

        if _ % 500 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test(epoch, tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True
            )
            preds = [tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(
                t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _ % 100 == 0:
                print(f'Completed {_}')

            predictions.extend(preds)
            actuals.extend(target)
    return predictions, actuals


# Train a model using the text files as the direct input.
def train_model_direct_files():
    filter_out_info = True
    train_ratio = 0.7
    file_numbers = [4, 10, 18, 20, 22, 23, 24, 25, 26, 30, 31, 32, 34, 37, 38, 45, 46, 48, 49, 50, 51, 52, 53, 56, 58, 59, 61, 65, 71, 73, 76, 77, 85, 88, 91, 94, 97, 98, 100, 101, 102, 103, 105, 106, 107, 108, 109, 110, 114, 115, 119, 121, 123, 127, 129, 130, 140, 142, 143, 149, 156, 157, 163, 164, 166, 170, 171, 172, 174, 175, 180, 184, 191, 192, 199, 200, 211, 213, 215, 216, 217, 218, 221, 223, 224, 227, 229, 232, 235, 237, 241, 243, 246, 247, 250, 253, 258, 259, 272, 273, 278, 279, 286,
                    291, 301, 308, 309, 310, 311, 317, 325, 330]
    PATH = 'wikilarge_epoch-5.pt'
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    device = torch.device('cpu')
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters())
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

    inputs = []
    labels = []
    for file_number in file_numbers:
        file = open(
            f"Output/Sentences/{file_number}.txt", "r", encoding='utf-8')
        for index, line in enumerate(file):
            line = line.strip().replace("\n", "")
            if(index % 5 == 1):
                if('<INFO>' in line):
                    if(filter_out_info == False):
                        line = line.replace('<INFO>', '')
                        inputs.append(line)
                    else:
                        pass
                else:
                    inputs.append(line)
            if(index % 5 == 2):
                if('<INFO>' in line):
                    if(filter_out_info == False):
                        line = line.replace('<INFO>', '')
                        labels.append(line)
                    else:
                        pass
                else:
                    labels.append(line)
    for i in range(len(inputs) - 1, -1, -1):
        if(len(inputs[i]) == 0):
            del inputs[i]
            del labels[i]

    random.seed(18)
    temp = list(zip(inputs, labels))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    inputs, labels = list(res1), list(res2)
    inputs = ["summarize: " + current for current in inputs]

    train_split_amount = round(len(inputs)*train_ratio)
    train_inputs = inputs[0:train_split_amount]
    train_labels = labels[0:train_split_amount]
    test_inputs = inputs[train_split_amount:]
    test_labels = labels[train_split_amount:]

    d = {'ctext': train_inputs, 'text': train_labels}
    df = pd.DataFrame(data=d)
    print(df.head())
    train_set = T5Dataset(df, tokenizer, 512, 150)
    train_params = {
        'batch_size': 2,
        'shuffle': True,
        'num_workers': 0
    }
    train_loader = DataLoader(train_set, **train_params)
    for epoch in range(1):
        train(epoch, tokenizer, model, device, train_loader, optimizer)

    d = {'ctext': test_inputs, 'text': test_labels}
    df = pd.DataFrame(data=d)
    test_set = T5Dataset(df, tokenizer, 512, 150)
    test_params = {
        'batch_size': 2,
        'shuffle': False,
        'num_workers': 0
    }
    test_loader = DataLoader(test_set, **test_params)
    predictions, actuals = test(1, tokenizer, model, device, test_loader)
    final_df = pd.DataFrame(
        {'Generated Text': predictions, 'Actual Text': actuals})
    final_df.to_csv('predictions.csv')


# Train a model first, then test the performance of the model.
def train_and_test_model():
    filter_out_info = True
    file_numbers = [4, 10, 18, 20, 22, 23, 24, 25, 26, 30, 31, 32, 34, 37, 38, 45, 46, 48, 49, 50, 51, 52, 53, 56, 58, 59, 61, 65, 71, 73, 76, 77, 85, 88, 91, 94, 97, 98, 100, 101, 102, 103, 105, 106, 107, 108, 109, 110, 114, 115, 119, 121, 123, 127, 129, 130, 140, 142, 143, 149, 156, 157, 163, 164, 166, 170, 171, 172, 174, 175, 180, 184, 191, 192, 199, 200, 211, 213, 215, 216, 217, 218, 221, 223, 224, 227, 229, 232, 235, 237, 241, 243, 246, 247, 250, 253, 258, 259, 272, 273, 278, 279, 286,
                    291, 301, 308, 309, 310, 311, 317, 325, 330]
    PATH = 'newsela_epoch-5.pt'
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    device = torch.device('cpu')
    device = 'cuda' if cuda.is_available() else 'cpu'
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters())
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    df = pd.read_csv("train_alignments.csv")
    df = df.drop('Unnamed: 0', 1)
    df = df.rename(columns={"Original": "ctext", "Simplified": "text"})
    temp_inputs = df['ctext'].tolist()
    temp_inputs = ["summarize: " + current for current in temp_inputs]
    df['ctext'] = temp_inputs
    train_set = T5Dataset(df, tokenizer, 512, 150)
    train_params = {
        'batch_size': 2,
        'shuffle': True,
        'num_workers': 0
    }

    train_loader = DataLoader(train_set, **train_params)
    for epoch in range(3):
        train(epoch, tokenizer, model, device, train_loader, optimizer)

    # Save Model
    torch.save(model.state_dict(), "newsela/newsela_final.pt")

    inputs = []
    labels = []
    random.seed(18)
    random.shuffle(file_numbers)
    file_numbers = file_numbers[int(
        len(file_numbers)*0.5):int(len(file_numbers)*0.8)]
    for file_number in file_numbers:
        file = open(
            f"Output/Final/{file_number}.txt", "r", encoding='utf-8')
        for index, line in enumerate(file):
            line = line.strip().replace("\n", "")
            if(index % 5 == 1):
                if('<INFO>' in line):
                    if(filter_out_info == False):
                        line = line.replace('<INFO>', '')
                        inputs.append(line)
                    else:
                        pass
                else:
                    inputs.append(line)
            if(index % 5 == 2):
                if('<INFO>' in line):
                    if(filter_out_info == False):
                        line = line.replace('<INFO>', '')
                        labels.append(line)
                    else:
                        pass
                else:
                    labels.append(line)
    for i in range(len(inputs) - 1, -1, -1):
        if(len(inputs[i]) == 0):
            del inputs[i]
            del labels[i]
    for i in range(len(labels) - 1, -1, -1):
        if(len(labels[i]) == 0):
            del inputs[i]
            del labels[i]

    random.seed(18)
    temp = list(zip(inputs, labels))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    inputs, labels = list(res1), list(res2)
    format_inputs = ["summarize: " + current for current in inputs]

    d = {'ctext': format_inputs, 'text': labels}
    df = pd.DataFrame(data=d)
    test_set = T5Dataset(df, tokenizer, 512, 150)
    test_params = {
        'batch_size': 2,
        'shuffle': False,
        'num_workers': 0
    }
    test_loader = DataLoader(test_set, **test_params)
    predictions, actuals = test(1, tokenizer, model, device, test_loader)
    final_df = pd.DataFrame(
        {'Original Text': inputs, 'Generated Text': predictions, 'Actual Text': actuals})
    final_df.to_csv('newsela_final.csv')
    print(final_df.values)


# Directly test the performance of a model.
def test_model():
    filter_out_info = True
    file_numbers = [4, 10, 18, 20, 22, 23, 24, 25, 26, 30, 31, 32, 34, 37, 38, 45, 46, 48, 49, 50, 51, 52, 53, 56, 58, 59, 61, 65, 71, 73, 76, 77, 85, 88, 91, 94, 97, 98, 100, 101, 102, 103, 105, 106, 107, 108, 109, 110, 114, 115, 119, 121, 123, 127, 129, 130, 140, 142, 143, 149, 156, 157, 163, 164, 166, 170, 171, 172, 174, 175, 180, 184, 191, 192, 199, 200, 211, 213, 215, 216, 217, 218, 221, 223, 224, 227, 229, 232, 235, 237, 241, 243, 246, 247, 250, 253, 258, 259, 272, 273, 278, 279, 286,
                    291, 301, 308, 309, 310, 311, 317, 325, 330]
    PATH = 'wikilarge_epoch-5.pt'
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    device = torch.device('cpu')
    device = 'cuda' if cuda.is_available() else 'cpu'
    model = model.to(device)
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    df = pd.read_csv("train_alignments.csv")
    df = df.drop('Unnamed: 0', 1)
    df = df.rename(columns={"Original": "ctext", "Simplified": "text"})
    temp_inputs = df['ctext'].tolist()
    temp_inputs = ["summarize: " + current for current in temp_inputs]
    df['ctext'] = temp_inputs
    print(df.head())
    train_set = T5Dataset(df, tokenizer, 512, 150)
    train_params = {
        'batch_size': 2,
        'shuffle': True,
        'num_workers': 0
    }

    inputs = []
    labels = []
    random.seed(18)
    random.shuffle(file_numbers)
    file_numbers = file_numbers[int(
        len(file_numbers)*0.5):int(len(file_numbers)*0.8)]
    for file_number in file_numbers:
        file = open(
            f"Output/Final/{file_number}.txt", "r", encoding='utf-8')
        for index, line in enumerate(file):
            line = line.strip().replace("\n", "")
            if(index % 5 == 1):
                if('<INFO>' in line):
                    if(filter_out_info == False):
                        line = line.replace('<INFO>', '')
                        inputs.append(line)
                    else:
                        pass
                else:
                    inputs.append(line)
            if(index % 5 == 2):
                if('<INFO>' in line):
                    if(filter_out_info == False):
                        line = line.replace('<INFO>', '')
                        labels.append(line)
                    else:
                        pass
                else:
                    labels.append(line)
    for i in range(len(inputs) - 1, -1, -1):
        if(len(inputs[i]) == 0):
            del inputs[i]
            del labels[i]
    for i in range(len(labels) - 1, -1, -1):
        if(len(labels[i]) == 0):
            del inputs[i]
            del labels[i]

    random.seed(18)
    temp = list(zip(inputs, labels))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    inputs, labels = list(res1), list(res2)
    format_inputs = ["summarize: " + current for current in inputs]

    d = {'ctext': format_inputs, 'text': labels}
    df = pd.DataFrame(data=d)
    test_set = T5Dataset(df, tokenizer, 512, 150)
    test_params = {
        'batch_size': 2,
        'shuffle': False,
        'num_workers': 0
    }
    test_loader = DataLoader(test_set, **test_params)
    predictions, actuals = test(1, tokenizer, model, device, test_loader)
    final_df = pd.DataFrame(
        {'Original Text': inputs, 'Generated Text': predictions, 'Actual Text': actuals})
    final_df.to_csv('wikilarge_untrained_final.csv')
    print(final_df.values)


def main():
    # train_model_direct_files()
    # train_and_test_model()
    test_model()


if __name__ == '__main__':
    main()
