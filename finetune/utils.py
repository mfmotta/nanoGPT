

import os
from pathlib import Path
import warnings
from urllib.request import urlretrieve
import bibtexparser
import re
from pdfminer.high_level import extract_text
from datasets import Dataset


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


def create_dataset(papers_path, arxiv_ids, split = 0.8, out_path = 'data'):

    train_corpus_file, val_corpus_file = create_corpus(papers_path, arxiv_ids, split = 0.8)
    num_papers = len(arxiv_ids)
    split_index = int(num_papers*split)

    train_dataset = {
        "text": [os.path.abspath(train_corpus_file) for _ in range(split_index)],
    }

    val_dataset = {
        "text": [os.path.abspath(val_corpus_file) for _ in range(split_index, num_papers)],
    }

    train_dataset = Dataset.from_dict(train_dataset, split = 'train')
    val_dataset = Dataset.from_dict(val_dataset, split = 'val')
    print('\n Created train and val splits')
    return train_dataset, val_dataset


