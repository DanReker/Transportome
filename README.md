# Transportome

Repository for datasets and code to reproduce drug transporter dataset currently under review.

The following files are currently stored in this repository
| Filename  | Content |
| ------------- | ------------- |
| Code.py | Python code to run retrospective evaluations, model fitting, and predictions  |
| Database.tsv  | The main database used for initial model fitting  |
| Database_drugbank5_only.tsv  | Only the DrugBank fraction of the main database  |
| additional_X.tsv  | additional (non-)substrates extracted from active learning literature search   |
| investigational_compounds.tsv  | Investigational drugs currently undergoing (pre) clinical evaluation   |
| investigational_library.tsv  | Investigational drugs that were experimentally characterized by us  |
| drugbank5_approved.smiles  | DrugBank 5 approved small molecules  |

## Requirements
The code was implemented using
- Python 3.7.9
- RDKit 2020.09.1
- Scikit-learn 1.0.2
- Imblearn 0.9.0
