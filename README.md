<h1 align="center">RAP: A Metric for Balancing Repetition and Performance in Open-Source Large Language Models</h1>

<p align="center">
  <a>Donghao HUANG</a> ·
  <a>Thanh-Son NGUYEN</a> ·
  <a>Fiona LIAUSVIA</a> ·
  <a>Zhaoxia WANG</a>
</p>

**Repetition-Aware Performance (RAP)** is a novel evaluation
metric that quantifies and integrates repetition
penalty into the assessment of model perfor-
mance, enabling tuning of repetition penalty
parameter (RPP).

## Abstract
Large Language Models (LLMs) have significantly
advanced natural language processing,
but content repetition in open-source LLMs
remains a critical challenge that adversely affects
user experience. The repetition penalty
parameter (RPP) aims to mitigate this issue by
preventing repeated content generation, but excessive
use of RPP can compromise the overall
quality. In this paper, we propose Repetition-
Aware Performance (RAP), a novel evaluation
metric that quantifies and integrates repetition
penalty into the assessment of model performance,
enabling tuning of RPP. We evaluate
our approach using twelve open-source LLMs,
ranging from 2 billion to 70 billion parameters,
tested on question answering and machine
translation tasks across three datasets with varying
prompting techniques. Experimental results
show that RAP effectively tunes RPP,
helping to identify a trade-off value that significantly
reduces repetition while minimizing
performance loss. The code and the dataset
of generated text can be accessed at https:
//github.com/inflaton/rap.

## Installation
```bash
# Clone the repository
git clone https://github.com/inflaton/rap.git
cd rap

# Create a virtual environment (e.g., using conda or venv)
conda create -n rap python=3.12
conda activate rap

# Install dependencies
pip install -r requirements.txt
```

## Datasets

- **QA**: under folder `data/datasets`
- **MT**: under folder `src-mt/datasets`

## Citation
If you use this code, please cite:
```bibtex
@inproceedings{huang2025rap,
    title = "RAP: A Metric for Balancing Repetition and Performance in Open-Source Large Language Models",
    author = "Huang, Donghao  and
      Nguyen, Thanh-Son  and
      Zhang, Ruiyi  and
      Liausvia, Fiona",
    booktitle = "Findings of the Nations of the Americas Chapter of the Association for Computational Linguistics: NAACL 2025",
    month = apr,
    year = "2025",
    address = "Albuquerque, USA",
    publisher = "Association for Computational Linguistics",
}
```

## License
MIT License