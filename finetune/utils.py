

from urllib.request import urlretrieve
import bibtexparser
import re
import os


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

def extract_metadata(bib_path):
    library = bibtexparser.parse_file(bib_path) 
    authors = library.entries[0].fields_dict['author']._value.splitlines()
    authors = [re.sub("\s{2,}"," ",elem.replace('and ','')) for elem in authors] 
    title = library.entries[0].fields_dict['title']._value
    title = re.sub("\s{2,}"," ",title)
    return title, authors


