# DNA Sequence Classification With Kernel Methods
Machine Learning with Kernel Methods (MVA 2021) - DNA sequence classification from scratch using kernel methods and ML algorithm

Transcription factors (TFs) are regulatory proteins that bind specific sequence motifs in the genome to activate or repress transcription of target genes.
Genome-wide protein-DNA binding maps can be profiled using some experimental techniques and thus all genomics can be classified into two classes for a TF of interest: bound or unbound.
In this challenge, we will work with three datasets corresponding to three different TFs, and predict whether or not a sequence bound with a TF or not.

To generate a submission file, please start the script: `svm_generate_predictions.py`. Feel free to try different parameters by modifying the "parameters" section in the script. It is provided here with the parameters that gave our best submission. 

Note that it may take a long time to compute the sum kernel of our best submission (several hours). If you want to quickly have correct baselines, you can change to a mismatch kernel with k=10 and m=1 (very fast), or a mismatch kernel with k=12 and m=2 (around 15 minutes to compute the kernels).
