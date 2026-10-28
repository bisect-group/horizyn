import random, pandas as pd
rng = random.Random(0)
rxns = ["CC(=O)O.OCC>>CC(=O)OCC.O","CCO>>CC=O","OC(=O)CC(=O)C(O)=O>>CC(=O)C(O)=O.O=C=O","CC(=O)C>>CC(O)C",
"OCC(O)CO>>OCC(=O)CO","CCCO>>CCC=O","CC(N)C(=O)O>>CC(=O)C(=O)O.N","OC(=O)C=CC(=O)O>>OC(=O)CC(=O)C(=O)O",
"CCCCO>>CCCC=O","CC(=O)OC>>CC(=O)O.CO","OCCO>>O=CCO","CC(O)C(=O)O>>CC(=O)C(=O)O",
"CCC(=O)C>>CCC(O)C","OCC(=O)O>>O=CC(=O)O","CC(C)O>>CC(C)=O","NCCO>>NCC=O"]
ecs = ["1.1.1.1","1.1.1.2","2.3.1.1","3.1.1.1","4.2.1.1","1.2.1.1","5.3.1.1","1.4.1.1"]
aa = "ACDEFGHIKLMNPQRSTVWY"
seqs = ["".join(rng.choice(aa) for _ in range(rng.randint(30, 50))) for _ in range(10)]
def rows(rs, n):
    out = []
    for i in range(n):
        r = rs[i % len(rs)]; k = rng.randrange(len(seqs))
        out.append({"rxn_smiles": r, "ec_number": ecs[rng.randrange(len(ecs))], "sequence": seqs[k]})
    return pd.DataFrame(out)
for name, rs, n in (("train", rxns[:10], 30), ("val", rxns[10:13], 6), ("test", rxns[13:], 6)):
    rows(rs, n).to_parquet(f"emulator_bench/tests/fixtures/split/{name}.parquet", index=False)
