# Code 'Multi-Hazard risk to global port infrastructure'

The codes allows one to produce the analysis presented in the paper 'Multi-hazard risk to global port infrastructure and resulting trade and logistics losses', published in Communications Earth & Environment. 

The code is split into two folders, the 'Operational_thresholds' folder and the 'Port_specific_risk' folder.

The 'Operational_thresholds' folder contains the code to reproduce the analysis of the operational downtime as a result of climatic extremes (waves, temperature, wind, overtopping). The input data is too large to be uploaded to Github, please contact the corresponding author in case you want to access this. 

The 'Port_specific_risk' folder contains the code to reproduce the analysis of the port-specific risk and trade risk estimates given natural hazard impact (flooding, TC wind, earthquakes) and operational thresholds. 
The 'Risk_code_probabilistic.py' file runs the analysis in a probabilistic manner, given the probabilistic sample of input parameters, in this case ~10,000 samples. One could also run this in an expected value mode, where only 1 run is performed. The 'Output_post_process.py' file takes all the output files of the probabilistic run and finds the median and confidence interval per output considered. 

The final model results, as presented in the paper, are available from Mendeley Data, which includes also the port-level asset database needed to run the analysis. Other input data can be requested from the Corresponding author. 
Mendeley data: https://data.mendeley.com/datasets/kdyt24tsh5

Corresponding Author:
Jasper Verschuur, jasper.verschuur@keble.ox.ac.uk
