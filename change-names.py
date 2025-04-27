import os

for icycle in range(20):
    # data_dir = "Results/predictions-v2016"
    # file1 = f"{data_dir}/BA-PSR_ES_consensus_CS_using_curves_concat_rates_then_consen_tf-c16.txt"
    # file2 = f"{data_dir}/BA-PSR_ES_consensus_CS_using_curves_concat_rates_then_consen_tf-c16.txt"
    data_dir = "Results/predictions-metalloprotein-ligand"
    file1 = f"{data_dir}/BA-PSR-comb_CS-act_type_P-L_M-L_P-M-c{icycle}_facet-curves_rates-max_dim1-r1-r12-ns23-treenum20000.csv"
    file2 = f"{data_dir}/BA-PSR-comb_CS-act_type_P-L_M-L_P-M-c{icycle}_facet-curves_rates.csv"
    os.system(f"mv {file1} {file2}")
