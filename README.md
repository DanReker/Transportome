![Transportome](https://user-images.githubusercontent.com/127516906/229308328-865a34e9-c2e0-457d-8171-ec1723ee07b0.jpg)

Repository for datasets and code to reproduce drug transporter study currently under review.

The following files are currently stored in this repository
| Filename  | Content |
| ------------- | ------------- |
| code.py | Python code to run retrospective evaluations, model fitting, and predictions  |
| database.tsv  | The main database used for initial model fitting  |
| database_drugbank5_only.tsv  | Only the DrugBank fraction of the main database  |
| additional_X.tsv  | Additional (non-)substrates extracted from active learning literature search   |
| investigational_compounds.tsv  | Investigational drugs currently undergoing (pre) clinical evaluation   |
| investigational_library.tsv  | Investigational drugs that were experimentally characterized by us  |
| drugbank5_approved.smiles  | DrugBank 5 approved small molecules  |

## Requirements
The code was implemented using
- [Python 3.7.9](https://www.python.org/)
- [RDKit 2020.09.1](https://www.rdkit.org/docs/Install.html)
- [Scikit-learn 1.0.2](https://scikit-learn.org/stable/)
- [Imblearn 0.9.0](https://imbalanced-learn.org/stable/)
