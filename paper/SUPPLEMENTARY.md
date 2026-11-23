# Supplementary Information

Implementation-level detail and negative results referenced from the main text. These are
decisions we made and verified against real data, not speculative design notes — each is kept
here, rather than in the main Methods, because it documents how one specific component reached its
current form rather than describing the resource's overall scientific design.

## Note S1. Claim verification: language model vs. natural-language inference

Claim verification (Section 2.1) — confirming that an extracted activity claim is actually
supported by its source sentence, rather than transcribed from a nearby but unrelated sentence —
was first implemented with a natural-language-inference (NLI) entailment model, a common choice
for this kind of grounding check. A controlled evaluation on a 25-document sample found it rejected
essentially every correctly extracted activity (0 of 84 accepted) with no usable discriminative
signal between genuine and spurious extractions. It was replaced with LLM-based verification, not
tuned around, since a gate that rejects everything provides no information regardless of threshold
tuning.

## Note S2. Sequence embeddings: ESM-2 vs. ProtBERT and hand-crafted features

Earlier iterations of this pipeline evaluated physicochemical descriptors, amino-acid composition,
and ProtBERT embeddings, individually and in combination with ESM-2, as machine-learning feature
blocks (Section 2.8). The feature set was narrowed to ESM-2 alone based on those comparisons.
ProtBERT was additionally dropped from the codebase after its distributed checkpoint ships only in
a legacy PyTorch format (`pytorch_model.bin`, no `safetensors`) that newer `transformers` releases
refuse to load below a PyTorch version this project cannot use with its target CUDA 12.2 runtime;
ESM-2 is the sole protein-language-model feature block in the current pipeline.

## Note S3. AMPlify: benchmarking-only status

AMPlify was originally run as a live annotation predictor alongside Macrel, AntiCP2, HemoPI2, and
AlgPred2 (Section 2.4). It was removed from that role after failing on a double-digit percentage of
batches over the full corpus, traced to an input-shape defect in its publicly released model
(unrelated to this project's own code). It is retained only as a benchmarking comparator
(Section 2.10), where failures are retried at the level of individual sequences so a crash on one
input does not remove the remaining sequences from the comparison.

## Note S4. Near-identity clustering: the length-ratio constraint

Near-identity clustering (Section 2.6) pairs peptides by sequence identity via PepMatch. An earlier
version of this step used an identity threshold alone, with no constraint on aligned-length ratio.
Because a short peptide can align at high identity to a small window of a much longer sequence
without the two being biologically similar overall, this collapsed the large majority of the corpus
into a single spuriously connected component — every peptide effectively "clustered" with nearly
everything else through short shared motifs. Requiring both ≥80% identity and ≥80% aligned-length
ratio removed this failure mode.

## Note S5. Anticancer database integration: sources considered but not included

Beyond AntiCP2 (Section 2.3), three additional anticancer-peptide resources were considered during
development — CancerPPD, ApInAPDB, and DCTPep. Their public download endpoints were unavailable at
implementation time, and this project's policy is not to ship a parser built against an assumed,
unverified data format. They remain registry entries with no active ingestion path rather than
silently wrong or absent data; adding real support for any of them is future work once a working,
inspectable source becomes available.

## Note S6. Cell-line disease classification: a caught hand-curation error

Cell-line-to-cancer-type classification (Section 2.7) was first attempted with a hand-written
name-to-cancer-type lookup table. That table mapped **BGC-823** and **Bel-7402** to gastric and
liver cancer respectively, following the naming convention their names suggest. Both are in fact
STR-profiling-confirmed HeLa (cervical cancer) derivatives — a documented case of cross-contaminated
Chinese cell-bank lines from that era — an error a name-based heuristic has no way to catch. This
motivated switching to the Cellosaurus API as the authoritative source for every cell-line
classification in this resource, rather than trusting any hand-curated mapping, however plausible
it looks.
