import os
import re
import unicodedata
import pandas as pd
import numpy as np

# 1. Define the Multilingual Lexicon mapping target findings to keywords
LEXICON = {
    "ACL": {
        "keywords": [
            "acl", "lca", "cruzado anterior", "cruciate",
            "capraz", "kruisband", "ruptur", "ρήξη", "χιαστού"
        ]
    },
    "MCL": {
        "keywords": [
            "mcl", "lcm", "colateral medial", "collaterale",
            "yan bag", "innenband", "πλαγίου"
        ]
    },
    "Medial Meniscus": {
        "keywords": [
            "medial meniscus", "menisco medial", "menisco interno",
            "ic meniskus", "mediale meniscus", "innenmeniskus", "ménisque médial", "έσω μηνίσκου"
        ]
    },
    "Lateral Meniscus": {
        "keywords": [
            "lateral meniscus", "menisco lateral", "menisco externo",
            "dis meniskus", "laterale meniscus", "außenmeniskus", "ménisque latéral", "έξω μηνίσκου"
        ]
    },
    "Medial OA": {
        "keywords": [
            "medial oa", "artrosis medial", "artrosis femorotibial medial",
            "medial artroz", "mediale artrose", "mediale gonarthrose", "arthrose médiale"
        ]
    },
    "Lateral OA": {
        "keywords": [
            "lateral oa", "artrosis lateral", "artrosis femorotibial lateral",
            "dis artroz", "laterale artrose", "laterale gonarthrose", "arthrose latérale"
        ]
    },
    "PF OA": {
        "keywords": [
            "pf oa", "patellofemoral", "patelofemoral", "fémoro-patellaire",
            "femorotibial lateral", "patellofemorale", "επιγoνατιδoμηριαίας"
        ]
    },
    "Effusion": {
        "keywords": [
            "effusion", "derrame", "efüzyon", "fluid", "erguss",
            "effusie", "épanchement", "υγρού", "ύδραρθρο"
        ]
    },
    "Synovitis": {
        "keywords": [
            "synovitis", "sinovitis", "sinovit", "synoviite", "υμενίτιδα"
        ]
    },
    "Baker's": {
        "keywords": [
            "baker", "popliteal", "quiste de baker", "popliteal kist",
            "poplitealzyste", "kyste de baker", "κύστη baker"
        ]
    },
    "Contusion": {
        "keywords": [
            "contusion", "bruise", "bruising", "contusión",
            "kemik kontüzyonu", "contusie", "kontusion", "οίδημα"
        ]
    },
    "Fracture": {
        "keywords": [
            "fracture", "fractura", "kirik", "fractuur", "fraktur", "κατάγμα"
        ]
    }
}

# Compile negation terms (prefix and suffix)
PRE_NEGATIONS = r"\b(no|sin|not|negative for|without|free of|keine|kein|pas de|sans|normal|conservada|conservado|intacto|intacta)\b"
POST_NEGATIONS = r"\b(izlenmedi|saptanmadi|yok|not seen|absent|normaal|intact|normal)\b"

def clean_and_normalize(text):
    """Converts text to lowercase, strips accents, and cleans formatting."""
    if not isinstance(text, str):
        return ""
    # Strip accents for unicode compatibility (e.g. ó -> o)
    nfkd_form = unicodedata.normalize('NFKD', text)
    text = "".join([c for c in nfkd_form if not unicodedata.combining(c)])
    return text.lower()

def check_negation(text, keyword, match_start, window=5):
    """
    Checks if a matching keyword is negated by scanning words
    within a specific sliding window before and after the match.
    """
    # Get words leading up to the keyword
    prefix_text = text[max(0, match_start - 60):match_start]
    prefix_words = re.findall(r'\w+', prefix_text)[-window:]
    if any(re.match(PRE_NEGATIONS, w) for w in prefix_words):
        return True
        
    # Get words following the keyword (critical for Turkish post-term negation)
    match_end = match_start + len(keyword)
    suffix_text = text[match_end:match_end + 60]
    suffix_words = re.findall(r'\w+', suffix_text)[:window]
    if any(re.match(POST_NEGATIONS, w) for w in suffix_words):
        return True
        
    return False

def parse_report(report_text):
    """
    Parses a report string and returns a dictionary of label predictions
    (1 for present, 0 for absent/not-mentioned/negated).
    """
    cleaned_report = clean_and_normalize(report_text)
    parsed_labels = {}
    
    for label, info in LEXICON.items():
        label_found = 0
        for kw in info["keywords"]:
            # Search for keyword matches
            for match in re.finditer(re.escape(kw), cleaned_report):
                match_start = match.start()
                # If keyword is present and NOT negated, label as positive
                if not check_negation(cleaned_report, kw, match_start):
                    label_found = 1
                    break
            if label_found == 1:
                break
        parsed_labels[label] = label_found
        
    return parsed_labels

def main(data_dir="data"):
    print("=== Starting Pseudo-Label Generator ===")
    
    train_csv = os.path.join(data_dir, "train.csv")
    if not os.path.exists(train_csv):
        print(f"Error: Missing {train_csv}")
        return
        
    df = pd.read_csv(train_csv)
    print(f"Loaded train.csv containing {len(df)} studies.")
    
    target_cols = list(LEXICON.keys())
    
    # Track performance on ground-truth labeled subset if available
    labeled_mask = df[target_cols].notnull().any(axis=1)
    num_gt_labeled = labeled_mask.sum()
    print(f"Number of clinician ground-truth labeled records: {num_gt_labeled}")
    
    # Generate pseudo-labels
    print("Generating pseudo-labels from free-text reports...")
    pseudo_labeled_df = df.copy()
    parsed_counts = 0
    
    for idx, row in pseudo_labeled_df.iterrows():
        # Only pseudo-label rows that are completely unlabeled
        if pd.isnull(row[target_cols]).all():
            parsed_dict = parse_report(row["Report"])
            for col in target_cols:
                pseudo_labeled_df.at[idx, col] = parsed_dict[col]
            parsed_counts += 1
            
    print(f"Successfully generated labels for {parsed_counts} unlabeled records.")
    
    # Save output
    output_path = os.path.join(data_dir, "train_pseudo_labeled.csv")
    pseudo_labeled_df.to_csv(output_path, index=False)
    print(f"Saved completed dataset to: {output_path}")
    
    # Print label distribution in the final combined dataset
    print("\n--- Final Abnormality Prevalence (Ground-Truth + Pseudo-Labels) ---")
    for col in target_cols:
        pos_count = (pseudo_labeled_df[col] == 1.0).sum()
        neg_count = (pseudo_labeled_df[col] == 0.0).sum()
        print(f"  {col:<18} | Positive: {pos_count:>4} | Negative: {neg_count:>4}")
        
    print("\n=== Generation complete. ===")

if __name__ == "__main__":
    main()
