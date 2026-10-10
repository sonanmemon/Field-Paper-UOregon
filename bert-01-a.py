
# ============================================================
# SBERT-01: TWO-STAGE INTERNATIONAL COORDINATION INDEX
# ============================================================


# ============================================================
# 1. IMPORT PACKAGES
# ============================================================

import numpy as np
import pandas as pd
import torch

from sentence_transformers import SentenceTransformer, util


# ============================================================
# 2. LOAD DATA AND PRETRAINED MODEL
# ============================================================

data = pd.read_csv("speeches8-cleaned.csv")

model = SentenceTransformer("all-mpnet-base-v2")

print("Number of speeches:", len(data))

# Speech-level embeddings are not needed because
# the two-stage index uses chunk-level embeddings.


# ============================================================
# 3. DEFINE INTERNATIONAL ANCHORS
# ============================================================

# These correspond to the original R trigger_words.

international_anchors = [

    # International scope
    "International economic and monetary policy.",
    "Global economic and financial developments.",
    "Multilateral economic and financial relations.",
    "Bilateral economic and monetary relationships.",
    "Cross-border financial and economic activities.",

    # Europe
    "Economic and monetary developments in Europe.",
    "European economic and financial policies.",
    "Economic policies of the European Union EU.",
    "Monetary policy in the Eurozone.",
    "Economic and monetary developments in the euro area.",

    # International institutions
    "The European Central Bank ECB.",
    "The Bank for International Settlements BIS.",
    "The International Monetary Fund IMF.",
    "The World Bank.",
    "The Organisation for Economic Co-operation and Development OECD.",
    "The Group of Seven G7 and G-7.",
    "The Group of Twenty G20 and G-20.",
    "The Financial Stability Board FSB."

]


# ============================================================
# 4. DEFINE FULL DICTIONARY ANCHORS
# ============================================================

# These correspond to the original R words dictionary.

dictionary_anchors = [

    # Coordination and cooperation
    "Coordination of economic and monetary policies.",
    "Cooperation among central banks and policymakers.",
    "Collaboration between monetary and financial authorities.",

    # International scope
    "International economic and monetary policy.",
    "Global economic and financial developments.",
    "Multilateral economic and financial relationships.",
    "Bilateral economic and monetary relationships.",
    "Cross-border economic and financial activities.",

    # Europe
    "Economic and monetary developments in Europe.",
    "European economic and financial policies.",
    "Economic policies of the European Union EU.",
    "Monetary policy in the Eurozone.",
    "Economic and monetary developments in the euro area.",

    # International institutions
    "The European Central Bank ECB.",
    "The Bank for International Settlements BIS.",
    "The International Monetary Fund IMF.",
    "The World Bank.",
    "The Organisation for Economic Co-operation and Development OECD.",
    "The Group of Seven G7 and G-7.",
    "The Group of Twenty G20 and G-20.",
    "The Financial Stability Board FSB.",

    # Integration
    "Economic and financial integration.",
    "Integrated economic and monetary policies.",

    # Harmonization and alignment
    "Harmonization of economic and financial policies.",
    "Alignment and synchronization of monetary policy strategies.",

    # Partnership and commitment
    "Partnerships among economic and monetary authorities.",
    "Commitment to shared economic and monetary objectives."

]


# ============================================================
# 5. GENERATE ANCHOR EMBEDDINGS
# ============================================================

international_embeddings = model.encode(
    international_anchors,
    convert_to_tensor=True,
    normalize_embeddings=True
)

dictionary_embeddings = model.encode(
    dictionary_anchors,
    convert_to_tensor=True,
    normalize_embeddings=True
)


# ============================================================
# 6. RECONSTRUCT SPEECH CHUNKS
# ============================================================

CHUNK_SIZE = 256
CHUNK_OVERLAP = 50

assert CHUNK_SIZE > CHUNK_OVERLAP

assert "text" in data.columns
assert "ymd" in data.columns

def create_chunks(text):

    if pd.isna(text) or not str(text).strip():
        return []

    tokens = model.tokenizer(
        str(text),
        add_special_tokens=False,
        truncation=False
    )["input_ids"]

    chunks = []

    step = CHUNK_SIZE - CHUNK_OVERLAP

    for start in range(0, len(tokens), step):

        chunk_tokens = tokens[start:start + CHUNK_SIZE]

        if not chunk_tokens:
            continue

        chunk_text = model.tokenizer.decode(
            chunk_tokens,
            skip_special_tokens=True
        )

        chunks.append(chunk_text)

        if start + CHUNK_SIZE >= len(tokens):
            break

    return chunks


chunk_records = []

for speech_id, row in data.iterrows():

    speech_chunks = create_chunks(row["text"])

    for chunk_id, chunk_text in enumerate(speech_chunks):

        chunk_records.append({
            "speech_id": speech_id,
            "chunk_id": chunk_id,
            "ymd": row["ymd"],
            "chunk_text": chunk_text
        })


chunks_df = pd.DataFrame(chunk_records)

if chunks_df.empty:
    raise ValueError("No valid speech chunks were generated.")

print("Total chunks:", len(chunks_df))


# ============================================================
# 7. GENERATE CHUNK EMBEDDINGS
# ============================================================

# Chunk-level embeddings are required.
# Speech-level embeddings cannot substitute for them.

chunk_embeddings = model.encode(
    chunks_df["chunk_text"].tolist(),
    batch_size=32,
    convert_to_tensor=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

print("Chunk embeddings:", chunk_embeddings.shape)


# ============================================================
# 8. COMPUTE INTERNATIONAL SIMILARITY
# ============================================================

international_scores = util.cos_sim(
    chunk_embeddings,
    international_embeddings
)

chunks_df["international_similarity"] = (
    international_scores.max(dim=1).values.cpu().numpy()
)


# ============================================================
# 9. COMPUTE FULL DICTIONARY SIMILARITY
# ============================================================

dictionary_scores = util.cos_sim(
    chunk_embeddings,
    dictionary_embeddings
)

chunks_df["dictionary_similarity"] = (
    dictionary_scores.max(dim=1).values.cpu().numpy()
)


# ============================================================
# 10. SAVE CHUNK-LEVEL DATA
# ============================================================

chunks_df.to_csv(
    "speeches8-SBERT-TwoStage-Chunks.csv",
    index=False
)

print("Saved chunk-level data.")


# ============================================================
# 11. SET INDEX PARAMETERS
# ============================================================

INTERNATIONAL_THRESHOLD = 0.35

DICTIONARY_THRESHOLD = 0.35

PERCENTILE = 0.90

assert 0 <= INTERNATIONAL_THRESHOLD <= 1
assert 0 <= DICTIONARY_THRESHOLD < 1
assert 0 < PERCENTILE < 1


# ============================================================
# 12. CONSTRUCT CHUNK-LEVEL SEMANTIC INTENSITY
# ============================================================

# Only dictionary similarities above the threshold
# contribute to semantic intensity.
#
# Rescaling by (1 - threshold) maps the retained
# similarity range to [0, 1].

chunks_df["dictionary_intensity"] = (
    (
        chunks_df["dictionary_similarity"]
        - DICTIONARY_THRESHOLD
    )
    .clip(lower=0)
    / (1 - DICTIONARY_THRESHOLD)
)


# ============================================================
# 13. CONSTRUCT SPEECH-LEVEL SCORES
# ============================================================

speech_scores = (
    chunks_df
    .groupby("speech_id")
    .agg(
        international_max=(
            "international_similarity", "max"
        ),

        dictionary_mean=(
            "dictionary_similarity", "mean"
        ),

        dictionary_intensity_mean=(
            "dictionary_intensity", "mean"
        ),

        total_chunks=(
            "chunk_id", "size"
        )
    )
    .reset_index()
)


# ============================================================
# 14. APPLY INTERNATIONAL TRIGGER
# ============================================================

# This matches the original keyword logic:
# at least one internationally relevant passage
# activates the speech-level trigger.

speech_scores["international_trigger"] = (
    speech_scores["international_max"]
    >= INTERNATIONAL_THRESHOLD
).astype(int)


# ============================================================
# 15. CONSTRUCT LENGTH-NORMALIZED SBERT INDEX
# ============================================================

speech_scores["SBERT_Intensity"] = (
    100
    * speech_scores["international_trigger"]
    * speech_scores["dictionary_intensity_mean"]
)

# This index ranges from 0 to 100 under the
# nonnegative similarity-threshold specification.


# ============================================================
# 16. APPLY 90TH PERCENTILE THRESHOLD
# ============================================================

cutoff = speech_scores["SBERT_Intensity"].quantile(
    PERCENTILE
)

speech_scores["SBERT_90"] = np.where(
    (
        (speech_scores["SBERT_Intensity"] >= cutoff)
        & (speech_scores["SBERT_Intensity"] > 0)
    ),
    speech_scores["SBERT_Intensity"],
    0.0
)

speech_scores["SBERT_90_Binary"] = (
    (
        speech_scores["SBERT_Intensity"] >= cutoff
    )
    & (
        speech_scores["SBERT_Intensity"] > 0
    )
).astype(int)


# ============================================================
# 17. LOG TRANSFORMATION
# ============================================================

# Positive shocks are logged.
# Zero shocks are retained as zero.
#
# This differs from the literal R log(0) = -Inf
# convention, but is more convenient for regressions.

speech_scores["SBERT_Log"] = np.where(
    speech_scores["SBERT_90"] > 0,
    np.log(
        speech_scores["SBERT_90"].clip(lower=1e-12)
    ),
    0.0
)


# ============================================================
# 18. MERGE SCORES WITH ORIGINAL SPEECH DATA
# ============================================================

speech_data = data.copy()

speech_data["speech_id"] = speech_data.index

speech_data = speech_data.merge(
    speech_scores,
    on="speech_id",
    how="left",
    validate="one_to_one"
)

# Missing scores correspond to speeches with no
# valid text chunks. Keep these missing rather
# than interpreting them as zero coordination.

speech_data.to_csv(
    "speeches8-SBERT-TwoStage.csv",
    index=False
)

print("Saved speech-level data.")


# ============================================================
# 19. CONSTRUCT DAILY SBERT INDEX
# ============================================================

# Multiple speeches on the same day are collapsed
# using the maximum score, matching the original
# daily keyword-index aggregation.

daily_sbert = (
    speech_data
    .groupby("ymd", as_index=False)
    .agg(
        SBERT_Intensity=(
            "SBERT_Intensity", "max"
        ),

        SBERT_Index=(
            "SBERT_90", "max"
        ),

        SBERT_Binary=(
            "SBERT_90_Binary", "max"
        ),

        SBERT_Log=(
            "SBERT_Log", "max"
        )
    )
)

daily_sbert = daily_sbert.sort_values("ymd")


# ============================================================
# 20. SAVE DAILY DATA
# ============================================================

daily_sbert.to_csv(
    "Coordination-Index-SBERT-90-Daily.csv",
    index=False
)

print("Saved daily SBERT index.")


# ============================================================
# 21. SUMMARY STATISTICS
# ============================================================

print("\nCHUNK-LEVEL SIMILARITY")

print(
    chunks_df[
        [
            "international_similarity",
            "dictionary_similarity",
            "dictionary_intensity"
        ]
    ].describe()
)


print("\nSPEECH-LEVEL INDEX")

print(
    speech_data[
        [
            "SBERT_Intensity",
            "SBERT_90",
            "SBERT_90_Binary"
        ]
    ].describe()
)


print("\nDAILY INDEX")

print(
    daily_sbert[
        [
            "SBERT_Intensity",
            "SBERT_Index",
            "SBERT_Binary"
        ]
    ].describe()
)


print("\n90th percentile cutoff:", cutoff)

print(
    "Internationally relevant speeches:",
    int(speech_scores["international_trigger"].sum())
)

print(
    "Positive speech-level shocks:",
    int(speech_scores["SBERT_90_Binary"].sum())
)

print(
    "Positive daily shocks:",
    int(daily_sbert["SBERT_Binary"].sum())
)

print(
    "Total daily observations:",
    len(daily_sbert)
)

print("\nSBERT-01 COMPLETED")
