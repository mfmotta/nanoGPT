

import os
import re
import numpy as np
import tiktoken
from pathlib import Path
from urllib.request import urlretrieve
import bibtexparser
from tqdm import tqdm
from pdfminer.high_level import extract_text
from datasets import Dataset, load_dataset


def download_paper(arxiv_id, dir_path):
    pdf_url = f"https://export.arxiv.org/pdf/{arxiv_id}"
    pdf_local_path = os.path.join(dir_path, f"{arxiv_id}.pdf")
    if pdf_url is not None:
        pdf_path, _ = urlretrieve(pdf_url, pdf_local_path)

    return pdf_path


def download_bibtex(arxiv_id, dir_path):
    arxiv_id_bib = str(arxiv_id).replace('.', '-')
    bibtex_url = f"https://dblp.uni-trier.de/rec/journals/corr/abs-{arxiv_id_bib}.bib?param=1"
    bibtex_local_path = os.path.join(dir_path, f"{arxiv_id_bib}.bib")
    if bibtex_url is not None:
        bib_path, _ = urlretrieve(bibtex_url, bibtex_local_path)

    return bib_path

def extract_metadata(arxiv_id, dir_path):
    bib_path = download_bibtex(arxiv_id, dir_path)
    library = bibtexparser.parse_file(bib_path) 
    authors = library.entries[0].fields_dict['author']._value.splitlines()
    authors = [re.sub("\s{2,}"," ",elem.replace('and ','')) for elem in authors] 
    title = library.entries[0].fields_dict['title']._value
    title = re.sub("\s{2,}"," ",title)
    return title, authors


def list_pdfs_in_dir(dir_path):
     pathlist = Path(dir_path).glob('**/*.pdf')
     titles = []
     for path in pathlist:
          titles.append(str(path))

     return titles

def extract_text_from_pdf(file_path):
     return extract_text(file_path)

def clean_pdf_lines(pdf_path, str_len):
    # remove  lines with string length < str_len
    # remove lines before theabstract
    # return lines
    abstract_idx = intro_idx = None
    lines = extract_text_from_pdf(pdf_path).splitlines()
    lines = [line for line in lines if len(line) >= str_len]
    for i in range(len(lines)):
        if lines[i].lower() == 'abstract':
            abstract_idx = i + 1
            break
    for j in range(i, len(lines)):
        if 'introduction' in lines[j].lower():
            intro_idx = j
            abstract = " ".join(lines[abstract_idx:intro_idx])
            text =  " ".join(lines[intro_idx:])
            break
    if abstract_idx is None or intro_idx is None:
        raise ValueError("Did not correctly parse paper, check if it contains 'abstract' and 'introduction'")
    return abstract, text


def create_corpus(papers_path, arxiv_ids, out_path = 'data', min_line_length = 4):

    #add logic to append if files already exist
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    text_file = out_path+"/papers_text.txt"
    abstract_file = out_path+"/papers_abstract.txt"

    #split_index = int(num_papers*split)
    with open(text_file, "a", encoding="utf-8") as file, \
        open(abstract_file, "a", encoding="utf-8") as abs_file:
          for idx, arxiv_id in enumerate(arxiv_ids): 
            path = download_paper(arxiv_id, papers_path)
            abstract, text = clean_pdf_lines(path, str_len = min_line_length) 
            title, author = extract_metadata(arxiv_id, papers_path)
            file.write(title+" : "+text + "\n")
            abs_file.write(title+" : "+abstract + "\n")
            
    print('\n Created {} and {}'.format(text_file, abstract_file))
    return text_file, abstract_file


def create_dataset(text_file_path, abstract_file_path, test_size = 0.2):

    dataset = load_dataset("text", data_files=text_file_path)

    with open(abstract_file_path, "r", encoding="utf-8") as abstract_file:
        abstract_data = abstract_file.read().splitlines()

    papers_dataset = Dataset.from_dict({
        "text": dataset['train']["text"],
        "abstract": abstract_data
    })

    split_dataset = papers_dataset.train_test_split(test_size=test_size, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    return split_dataset


def tokenize(dataset):
     
    enc = tiktoken.get_encoding("gpt2")
    prefix = "summarize as abstract: "
    def process(examples):
        #based both on karpathy's prepare.py and  https://huggingface.co/docs/transformers/tasks/summarization
        inputs = [prefix + doc for doc in examples["text"]]
        ids = [enc.encode_ordinary(elem)+[enc.eot_token] for elem in inputs] # encode_ordinary ignores any special tokens
        # added the end of text token, e.g. 50256 for gpt2 bpe # note: check if eot should be prepended not appended 
        label = enc.encode_ordinary_batch(examples["abstract"]) #TODO use encode_batch ?
        #out = {'ids': ids, 'len': len(ids)}
        lengths = [len(id) for id in ids]
        return {'ids': ids, 'label': label, 'len': lengths }

    tokenized = dataset.map( #https://github.com/huggingface/datasets/blob/src/datasets/dataset_dict.py
            process,
            remove_columns=['text', 'abstract'],
            desc="tokenizing the splits",
            batched=True #in contrast to Karpathy's several examples
        )
    return tokenized

def create_sharded_dataset(tokenized_dataset, out_dir, total_batches):
    #total_batches = num_shards #TODO perhaps rename?

    for split, dset in tokenized_dataset.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(out_dir, f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        #total_batches = 2 #1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()




