import pandas as pd
import glob

def sbs_matrix(frequency_table_folder_path="Frequency_Table", cancer_type="Ovary-AdenoCA", output_matrix_path="ovary.SBS96.all"):
    df = pd.DataFrame(index=pd.read_csv("SBS-96_mutation_type.csv", header=None)[0])
    for x in list(glob.glob(frequency_table_folder_path + "/" + cancer_type + "*sbs_freq_table.csv")):
        sample = pd.read_csv(x)
        sample['filter'] = [x[0] for x in sample['SubType']]
        sample.sort_values(["filter", "Type"], inplace=True)
        unique_id = int(x.split("_")[-4])
        df[unique_id] = sample['Frequency'].tolist()

    df = df.reindex(sorted(df.columns), axis=1)
    df.to_csv(output_matrix_path, sep="\t")
    return df

sbs_matrix(frequency_table_folder_path = "Frequency_Table/",
           cancer_type = "Ovary-AdenoCA",
           output_matrix_path = "ovary.SBS96.all")