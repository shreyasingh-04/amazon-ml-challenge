import re
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

def extract_numbers(text):
    """Extracts all numeric tokens from a string."""
    return set(re.findall(r'\b\d+\b', text))

def compute_features(row_s1, row_s2):
    """
    Computes pairwise similarity features separately for name and address.
    Generates string similarity ratios (Jaro-Winkler, Token Sort, Token Set)
    and numeric match signals for addresses.
    """
    # Assuming Person A's normalized text is passed, but we add a defensive basic lowercase/clean
    def norm(text):
        if not isinstance(text, str):
            return ""
        # Keep it simple: lowercase and alphanumeric only for similarity metrics
        return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', text.lower())).strip()
        
    name1 = norm(row_s1.get('business_name', ''))
    name2 = norm(row_s2.get('business_name', ''))
    
    addr1 = norm(row_s1.get('business_address', ''))
    addr2 = norm(row_s2.get('business_address', ''))
    
    features = {}
    
    # 1. Name features (Using RapidFuzz for performance and accuracy)
    features['name_jaro_winkler'] = JaroWinkler.normalized_similarity(name1, name2) if name1 and name2 else 0.0
    features['name_token_sort'] = fuzz.token_sort_ratio(name1, name2) / 100.0 if name1 and name2 else 0.0
    features['name_token_set'] = fuzz.token_set_ratio(name1, name2) / 100.0 if name1 and name2 else 0.0
    features['name_len_diff'] = abs(len(name1) - len(name2))
    
    # 2. Address features
    features['addr_jaro_winkler'] = JaroWinkler.normalized_similarity(addr1, addr2) if addr1 and addr2 else 0.0
    features['addr_token_sort'] = fuzz.token_sort_ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0
    features['addr_token_set'] = fuzz.token_set_ratio(addr1, addr2) / 100.0 if addr1 and addr2 else 0.0
    features['addr_len_diff'] = abs(len(addr1) - len(addr2))
    
    # 3. Numeric-token match for address (strong signal for street numbers/zip codes)
    nums1 = extract_numbers(addr1)
    nums2 = extract_numbers(addr2)
    if not nums1 and not nums2:
        features['addr_num_match'] = 0.5 # Neutral - neither has numbers
    elif not nums1 or not nums2:
        features['addr_num_match'] = 0.0 # One has numbers, the other doesn't
    else:
        intersection = nums1.intersection(nums2)
        if len(intersection) == max(len(nums1), len(nums2)):
            features['addr_num_match'] = 1.0 # Perfect match of all numbers
        elif len(intersection) > 0:
            features['addr_num_match'] = 0.5 # Partial match
        else:
            features['addr_num_match'] = -1.0 # Mismatch penalty (strong negative signal)
            
    # Country signal
    c1 = str(row_s1.get('country', '')).strip().lower()
    c2 = str(row_s2.get('country', '')).strip().lower()
    features['same_country'] = 1.0 if (c1 and c2 and c1 == c2) else 0.0
            
    return features
