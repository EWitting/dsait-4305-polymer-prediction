import dill
import warnings

from tqdm import tqdm

class OligomerPreprocessor:
    def __init__(self, oligomer_len: int = 1):
        self.oligomer_len = oligomer_len

    def fit(self, X, y=None):
        return self

    def transform(self, smiles_list: list[str]):
        graphs = []
        no_oligomer_count = 0
        pbar = tqdm(smiles_list, desc=f"Processing SMILES string to oligomer of length {self.oligomer_len}")
        for smiles in pbar:
            processed_smiles = smiles

            if self.oligomer_len > 1 and smiles.count("*") == 2:
                try:
                    # apply polymerization to create a longer SMILES string.
                    processed_smiles = smiles_to_oligomer(smiles, self.oligomer_len)
                except Exception as e:
                    no_oligomer_count += 1
                    pbar.set_postfix({"Unsuccessful": no_oligomer_count})

            # Convert the final SMILES string to a graph Data object using RDKit
            try:
                graph = from_smiles(processed_smiles)
                graph.x = graph.x.float()
                graphs.append(graph)
            except Exception as e:
                warnings.warn(
                    f"Could not create graph for SMILES '{processed_smiles}': {e}. Skipping."
                )
        if no_oligomer_count > 0:
            warnings.warn(f"Could not generate oligomer for {no_oligomer_count}/{len(smiles_list)} SMILES strings.")

        return graphs

    def fit_transform(self, smiles_list: list[str], y=None):
        self.fit(smiles_list, y)
        return self.transform(smiles_list)