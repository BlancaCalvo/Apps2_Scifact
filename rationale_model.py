import argparse
import torch
import jsonlines
import os

from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from typing import List
from sklearn.metrics import f1_score, precision_score, recall_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')

class SciFactRationaleSelectionDataset(Dataset):

    # This function loads all the claims in the train/dev set, it extracts the sentences in the annotated abstracts,
    # and then checks if these are the annotated rationales (sentences), which stores as is_evidence:True/False,
    # the dict is pairs of sentences and claims with the is_evidence relation between them
    # cited_doc_ids in train set are the cited articles after the claim was stated, so are likely to be the doc
    # containing the evidence sentences, so that's were it looks for it, but also to other ones: that's the evidence key
    # in the train dataset
    # the instances that contain an emtpy dict in evidence are the NOTENOUGHINFO claims, our goal is that our model
    # does not retrieve any evidence for these instances

    def __init__(self, corpus: str, claims: str):
        self.samples = []
        corpus = {doc['doc_id']: doc for doc in jsonlines.open(corpus)}
        for claim in jsonlines.open(claims):
            for doc_id, evidence in claim['evidence'].items():
                doc = corpus[int(doc_id)]
                evidence_sentence_idx = {s for es in evidence for s in es['sentences']}
                for i, sentence in enumerate(doc['abstract']):
                    self.samples.append({
                        'claim': claim['claim'],
                        'sentence': sentence,
                        'is_evidence': i in evidence_sentence_idx
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def encode(claims: List[str], sentences: List[str], tokenizer):

    # This function encodes to ids the batch to a max sequence length of 512, and pads if input is shorter

    encoded_dict = tokenizer.batch_encode_plus(zip(sentences, claims), pad_to_max_length=True, return_tensors='pt')
    if encoded_dict['input_ids'].size(1) > 512:
        # Too long for the model. Truncate it
        encoded_dict = tokenizer.batch_encode_plus( zip(sentences, claims), max_length=512,
                                                    truncation_strategy='only_first', pad_to_max_length=True,
                                                    return_tensors='pt')
    encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
    return encoded_dict


def evaluate(model, dataset, batch_size, tokenizer):

    # this function trains de model, for each batch in the DataLoader it encodes each claim and sentence together
    # separated by [SEP], it then produces logits that indicate if the sentence is relevant or not

    model.eval()
    targets = []
    outputs = []
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size):
            encoded_dict = encode(batch['claim'], batch['sentence'], tokenizer)
            logits = model(**encoded_dict)[0] # get binary logits: either True or False
            targets.extend(batch['is_evidence'].float().tolist())
            outputs.extend(logits.argmax(dim=1).tolist())
            #print(outputs) # at the beginning always predicts 0, give it more epochs
    return f1_score(targets, outputs, zero_division=0),\
           precision_score(targets, outputs, zero_division=0),\
           recall_score(targets, outputs, zero_division=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', type=str, default = 'data/corpus.jsonl', required=False)
    parser.add_argument('--train', type=str, default = 'data/claims_train.jsonl', required=False)
    parser.add_argument('--dev', type=str, default = 'data/claims_dev.jsonl', required=False)
    parser.add_argument('--dest', type=str, required=False, default = 'models/distil/', help='Folder to save the model')
    parser.add_argument('--model', type=str, default='distilbert-base-uncased')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    trainset = SciFactRationaleSelectionDataset(args.corpus, args.train)#[0:100]
    devset = SciFactRationaleSelectionDataset(args.corpus, args.dev)#[0:25]

    print(len(trainset))
    print(trainset[1]) # TRUE EXAMPLE
    print(trainset[4]) # FALSE EXAMPLE

    batch_size = 8
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device)
    optimizer = torch.optim.Adam([ # ask why does the optimizer change, is this a common practice?
        {'params': model.distilbert.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ])
    scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 20)

    for e in range(args.epochs): # for each epoch
        model.train() # train
        t = tqdm(DataLoader(trainset, batch_size=batch_size, shuffle=True)) # shuffle and create the batches
        for i, batch in enumerate(t): # for each batch
            #print(i)
            encoded_dict = encode(batch['claim'], batch['sentence'], tokenizer) # encode like sentence [SEP] claim
            loss, logits = model(**encoded_dict, labels=batch['is_evidence'].long().to(device)) # run the model
            loss.backward() # backpropagate the loss
            if (i + 1) % (256 // batch_size) == 0: # gradient accumulation
                optimizer.step()
                optimizer.zero_grad()
                t.set_description(f'Epoch {e}, iter {i}, loss: {round(loss.item(), 4)}')
        scheduler.step()
        train_score = evaluate(model, trainset, batch_size, tokenizer)
        print(f'Epoch {e}, train f1: %.4f, precision: %.4f, recall: %.4f' % train_score)
        #print(f'Epoch {e}, train loss: %.4f' % loss)
        dev_score = evaluate(model, devset, batch_size, tokenizer)
        print(f'Epoch {e}, dev f1: %.4f, precision: %.4f, recall: %.4f' % dev_score)
        # Save
        save_path = os.path.join(args.dest, f'epoch-{e}-f1-{int(dev_score[0] * 1e4)}')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)

main()