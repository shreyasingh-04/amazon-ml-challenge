class Matcher:
    """
    Predicts match / no-match per candidate pair.
    Precision-first: calibrated to avoid false merges (F0.5 penalizes false positives 2x).
    """
    def __init__(self, threshold=0.83):
        # We start with a high threshold. You should tune this on train_ground_truth.tsv!
        self.threshold = threshold
        
    def predict(self, features):
        """
        Returns True if the candidate is a match, False otherwise.
        """
        # Name score computation
        # Token set ratio handles word reorderings (e.g. "Amazon Inc" vs "Inc Amazon") very well
        name_score = (
            features['name_jaro_winkler'] * 0.4 + 
            features['name_token_set'] * 0.4 + 
            features['name_token_sort'] * 0.2
        )
                      
        # Address score computation
        addr_score = (
            features['addr_jaro_winkler'] * 0.3 + 
            features['addr_token_set'] * 0.4 + 
            features['addr_token_sort'] * 0.3
        )
                      
        # Adjust address score based on numeric tokens (street number / zip code)
        # Mismatching numbers is a very strong negative signal.
        num_match = features.get('addr_num_match', 0.5)
        if num_match == -1.0:
            addr_score -= 0.35 # Heavy penalty for mismatching numbers
        elif num_match == 1.0:
            addr_score += 0.1  # Bonus for exact number matches
            
        # Overall combined score - prioritizing name matches over address
        combined_score = name_score * 0.65 + addr_score * 0.35
        
        # Penalize if name length difference is large but token set is high 
        # (e.g., "McDonalds" vs "McDonalds Corporation Global HQ")
        if features['name_len_diff'] > 20 and name_score < 0.95:
            combined_score -= 0.15
            
        # Severe penalty if we know they are in different countries
        if features['same_country'] == 0.0:
            combined_score -= 0.4
            
        # Match if the score is above threshold
        # This provides an explicit path for predicting "no match" when unsure
        return combined_score >= self.threshold
