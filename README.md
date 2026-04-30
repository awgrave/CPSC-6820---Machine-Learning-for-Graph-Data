# Patient similarity graphs for clinical risk prediction
## Overview
This project predicts in-hospital mortality (binary label) from the MIMIC-III clinical database by representing patients as nodes in a graph.
Edges connect each patient to their k nearest neighbors in standardized feature space (patient similarity graph).   

Models compared:

Logistic regression (baseline)

2-layer GCN

2-layer GraphSAGE

## Dependencies
Use **Python 3.10+** (venv, conda, or Google Colab).   

Required packages:

torch  
torch-geometric  
numpy  
pandas  
scikit-learn  
matplotlib  
seaborn  

## Data setup
This project does not include datasets due to PhysioNet's Credentialed Health Data Use Agreement 1.5.0

Access: If you would like to access the data, go to [MIMIC-iii](https://physionet.org/content/mimiciii/1.4/) 

Follow the required steps to get access.

Paths: Unpack CSVs into mimic-iii-clinical-database-1.4 in the project directory 

Resources: Full runs read large tables; adjust N_PATIENTS or use a subsample/Colab-friendly subset if needed.

## License

[Apache 2.0](https://choosealicense/com/licenses/apache-2.0/)
