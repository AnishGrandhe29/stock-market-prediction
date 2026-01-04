import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import os

def load_social_data(reddit_path, twitter_path):
    """
    Loads Reddit and Twitter data, cleans it, and synthesizes dates 
    (uniformly distributed over 2015-2025) since the source datasets lack timestamps.
    """
    print("Loading social data...")
    
    # Load Reddit Data
    try:
        reddit_df = pd.read_csv(reddit_path)
        reddit_df = reddit_df.rename(columns={'clean_comment': 'text', 'category': 'sentiment'})
        reddit_df = reddit_df.dropna(subset=['text', 'sentiment'])
        print(f"Loaded {len(reddit_df)} Reddit rows.")
    except Exception as e:
        print(f"Error loading Reddit data: {e}")
        reddit_df = pd.DataFrame(columns=['text', 'sentiment'])

    # Load Twitter Data
    try:
        twitter_df = pd.read_csv(twitter_path)
        twitter_df = twitter_df.rename(columns={'clean_text': 'text', 'category': 'sentiment'})
        twitter_df = twitter_df.dropna(subset=['text', 'sentiment'])
        print(f"Loaded {len(twitter_df)} Twitter rows.")
    except Exception as e:
        print(f"Error loading Twitter data: {e}")
        twitter_df = pd.DataFrame(columns=['text', 'sentiment'])

    # Combine
    combined_df = pd.concat([reddit_df, twitter_df], ignore_index=True)
    
    # Synthesize Dates (2015-01-01 to 2025-12-31)
    # We distribute the data uniformly across the days to simulate a continuous stream.
    print("Synthesizing dates for social data...")
    start_date = pd.to_datetime("2015-01-01")
    end_date = pd.to_datetime("2025-12-31")
    days_range = (end_date - start_date).days + 1
    
    # Generate random day offsets
    random_days = np.random.randint(0, days_range, size=len(combined_df))
    combined_df['Date'] = start_date + pd.to_timedelta(random_days, unit='D')
    
    # Sort by Date
    combined_df = combined_df.sort_values('Date').reset_index(drop=True)
    
    return combined_df

def generate_embeddings_and_aggregate(df, batch_size=32):
    """
    Generates DistilBERT embeddings for the text column and aggregates 
    (mean) per day along with sentiment and count.
    """
    print("Initializing DistilBERT...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    model.eval()

    all_embeddings = []
    texts = df['text'].astype(str).tolist()
    
    print(f"Generating embeddings for {len(texts)} texts...")
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i : i + batch_size]
        
        try:
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            
            # Use CLS token embedding (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            # Append zero vectors in case of error to maintain shape
            all_embeddings.append(np.zeros((len(batch_texts), 768)))

    # Concatenate all embeddings
    if all_embeddings:
        embedding_matrix = np.vstack(all_embeddings)
    else:
        embedding_matrix = np.zeros((0, 768))
        
    # Create a DataFrame with embeddings
    emb_df = pd.DataFrame(embedding_matrix, columns=[f'emb_{k}' for k in range(768)])
    
    # Combine with original DF (Date and Sentiment)
    df_with_emb = pd.concat([df[['Date', 'sentiment']].reset_index(drop=True), emb_df], axis=1)
    
    # Aggregate by Date
    print("Aggregating by Date...")
    daily_agg = df_with_emb.groupby('Date').agg(
        sentiment_mean=('sentiment', 'mean'),
        sentiment_count=('sentiment', 'count'),
        **{f'emb_{k}': (f'emb_{k}', 'mean') for k in range(768)}
    )
    
    return daily_agg

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    reddit_file = os.path.join(base_dir, "Datasets", "Twitter and Reddit Sentimental analysis Dataset", "Reddit_Data.csv")
    twitter_file = os.path.join(base_dir, "Datasets", "Twitter and Reddit Sentimental analysis Dataset", "Twitter_Data.csv")
    output_file = os.path.join(base_dir, "Datasets", "social_features.csv")
    
    # Load and Process
    df = load_social_data(reddit_file, twitter_file)
    
    # Sampling 5000 rows for feasibility in this environment
    print("Sampling 5000 rows for feasibility...")
    if len(df) > 5000:
        df = df.sample(5000, random_state=42).sort_values('Date').reset_index(drop=True)
        
    daily_features = generate_embeddings_and_aggregate(df)
    
    # Save
    daily_features.to_csv(output_file)
    print(f"Saved social features to {output_file}")
