import argparse
import jsonlines

import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--corpus', type=str, default='data/corpus.jsonl', required=False)
parser.add_argument('--dataset', type=str, default='data/claims_dev.jsonl', required=False)
parser.add_argument('--model', type=str, default='models/distil/epoch-19-f1-6445/', required=False)
parser.add_argument('--threshold', type=float, default=0.5, required=False)
parser.add_argument('--output', type=str, default='predictions/predicted_rationale_dev.jsonl', required=False)
parser.add_argument('--include-nei', action='store_true')
parser.add_argument('--abstract_store', type=str, default = 'predictions/abstract_dev.jsonl', required=False)
parser.add_argument('--k_rationales', type=int, default=None, required=False)
args = parser.parse_args()

dataset = jsonlines.open(args.dataset)

# TAKE GOLD ABSTRACTS AND SAVE THEM IN A FILE
abstracts = jsonlines.open(args.abstract_store, 'w')

for data in dataset:
    doc_ids = list(map(int, data['evidence'].keys()))
    if not doc_ids and args.include_nei:
        doc_ids = [data['cited_doc_ids'][0]]

    abstracts.write({
        'claim_id': data['id'],
        'doc_ids': doc_ids
    })

# SELECT THE RATIONAL
corpus = {doc['doc_id']: doc for doc in jsonlines.open(args.corpus)}
abstract_retrieval = jsonlines.open(args.abstract_store)
dataset = jsonlines.open(args.dataset)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device "{device}"')

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()
results = []

with torch.no_grad():
    for data, retrieval in tqdm(list(zip(dataset, abstract_retrieval))):
        assert data['id'] == retrieval['claim_id']
        claim = data['claim']

        evidence_scores = {}
        for doc_id in retrieval['doc_ids']:
            doc = corpus[doc_id]
            sentences = doc['abstract']

            encoded_dict = tokenizer.batch_encode_plus(
                zip(sentences, [claim] * len(sentences)), # if not args.only_rationale else sentences,
                #sentences,
                pad_to_max_length=True,
                return_tensors='pt'
            )
            encoded_dict = {key: tensor.to(device) for key, tensor in encoded_dict.items()}
            sentence_scores = torch.softmax(model(**encoded_dict)[0], dim=1)[:, 1].detach().cpu().numpy()
            evidence_scores[doc_id] = sentence_scores
        results.append({
            'claim_id': retrieval['claim_id'],
            'evidence_scores': evidence_scores
        })


def output_k(output_path, k=None):
    output = jsonlines.open(output_path, 'w')
    for result in results:
        if k:
            evidence = {doc_id: list(sorted(sentence_scores.argsort()[-k:][::-1].tolist()))
                        for doc_id, sentence_scores in result['evidence_scores'].items()}
        else:
            evidence = {doc_id: (sentence_scores >= args.threshold).nonzero()[0].tolist()
                        for doc_id, sentence_scores in result['evidence_scores'].items()}
        output.write({
            'claim_id': result['claim_id'],
            'evidence': evidence
        })


output_k(args.output, args.k_rationales)

