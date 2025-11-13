# pip install sentence-transformers pandas evaluate

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import evaluate

# Load models
model = SentenceTransformer('all-MiniLM-L6-v2')  # For semantic similarity
rouge = evaluate.load("rouge")                   # For ROUGE metrics

def compute_semantic_similarity(text1, text2):
    emb1 = model.encode(text1, convert_to_tensor=True)
    emb2 = model.encode(text2, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))

def compute_rouge_scores(reference, prediction):
    results = rouge.compute(predictions=[prediction], references=[reference])
    return results  # returns dict with 'rouge1', 'rouge2', 'rougeL', etc.

def compare_answers_from_csv(input_csv_path, output_csv_path,
                             col_ground_truth='GroundTruth', col_generated='GeneratedAnswer'):
    # Load data
    df = pd.read_csv(input_csv_path, encoding='latin1')

    # Compute metrics
    similarity_scores = []
    rouge1_scores = []
    rouge2_scores = []
    rougel_scores = []

    for idx, row in df.iterrows():
        gt = str(row[col_ground_truth])
        gen = str(row[col_generated])
        # Semantic similarity (cosine)
        score = compute_semantic_similarity(gt, gen)
        similarity_scores.append(score)
        # ROUGE scores
        rouge_dict = compute_rouge_scores(gt, gen)
        rouge1_scores.append(rouge_dict.get('rouge1', 0))
        rouge2_scores.append(rouge_dict.get('rouge2', 0))
        rougel_scores.append(rouge_dict.get('rougeL', 0))

    # Add results to DataFrame
    df['SemanticSimilarity'] = similarity_scores
    df['ROUGE-1'] = rouge1_scores
    df['ROUGE-2'] = rouge2_scores
    df['ROUGE-L'] = rougel_scores

    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"✅ Saved comparison results to: {output_csv_path}")
    return df

# Example usage
if __name__ == "__main__":
    input_csv = "GenerateAnswers.csv"  # must contain "GroundTruth" and "GeneratedAnswer" columns
    output_csv = "combined_similarity_rouge_scores.csv"
    df_result = compare_answers_from_csv(input_csv, output_csv)
    print(df_result.head())
