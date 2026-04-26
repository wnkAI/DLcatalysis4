# /five analyze — Horizyn-1 (PNAS 2026) vs DLcatalysis 4.0

You are one of five independent analysts. Do NOT consult the others.
Form your own judgment.

## The paper (to react to)

Rocks JW, Truong DP, Rappoport D, et al. **"Dual-encoder contrastive
learning accelerates enzyme discovery."** *PNAS* 2026;121(12):e2520070123.
https://doi.org/10.1073/pnas.2520070123

Summary of key claims (translated from the Chinese digest; trust only
the structural claims, not the rhetorical ones):

- **Task**: given a target chemical reaction, retrieve the enzyme
  sequence most likely to catalyze it. This is a **retrieval /
  discovery** task, not a regression task.
- **Architecture**: two independent encoders, both trained with a
  symmetric contrastive (InfoNCE-style) objective.
  - Reaction encoder: Transformer over reaction SMILES
  - Protein encoder: pretrained PLM (ESM-2-ish)
  - Both embed into a shared latent space; cosine similarity →
    catalysis match score.
- **Training data**: 11,062 reactions × 4,742 proteins → 19,363
  reaction-enzyme pairs (union of UniProt / Rhea / BRENDA-style sources).
- **Evaluation**:
  - Rhea 2,843-reaction inference set
  - Ene reductase set: 406 reactions, 221 proteins, 903 pairs
  - Reported Top-10 accuracy >75%
- **Application vignettes**:
  1. Orphan reaction de-orphaning (18/24 succeed at top-5)
  2. Promiscuity prediction (enzyme ↔ multiple substrates)
  3. Non-natural reaction enzyme discovery
- **Claimed speedup**: ~100× vs BLAST / homology baselines.

## Our project — DLcatalysis 4.0 (the thing you're giving advice about)

Task: given a **(enzyme, substrate)** pair, predict **log10(kcat/Km)**
as a scalar regression target. This is NOT retrieval — we are
quantitatively ranking pairs on catalytic efficiency.

Current v4_ultimate architecture:
```
pred = y_seq + g_pair*(y_sub + y_rxn + y_int3d)
             + g_struct * y_struct + g_annot * y_annot
```
where:
- `y_seq`    — ProtT5 (frozen) + attn pool + MLP
- `y_sub`    — GINE on substrate SMILES + RXNMapper reaction-center flag
- `y_rxn`    — DRFP reaction fingerprint (2048-d, static hash-based)
- `y_int3d`  — pocket residue × substrate atom cross-attn + RBF
               distance + new NAC bias (catalytic residue × rxn center)
- `y_struct` — GVP-GNN on 8 Å pocket (K=32 residues)
- `y_annot`  — InterPro / Pfam / GO bag-of-ids

Training losses: logcosh regression + within-enzyme pairwise ranking
+ within-substrate pairwise ranking + optional PCC loss. Training-time
modality dropout on rxn / pocket / annot branches.

Dataset size and scope: ~comparable to Horizyn-1 (~30k pairs, 10-fold CV).

## The question

Give **three concrete, actionable ideas** we can borrow or adapt from
Horizyn-1 into DLcatalysis 4.0 without abandoning our regression task
or rebuilding from scratch. For each idea:

1. What to add / change (file or module level, not line level).
2. Why it's worth the effort for our task (not just "novel").
3. Trade-off: the actual cost (compute, data, engineering hours).
4. Would-I-do-it-now verdict: YES, LATER, or NO (with reasoning).

Then flag **one idea from the paper we should NOT copy**, and say why.

## Output format

Four bullet sections: three "ideas", one "what not to copy". Terse,
concrete. No executive summary. No rephrasing of the paper. End with
your single strongest recommendation in one sentence.
