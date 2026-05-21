import torch
from prompts import prompt_for_pairs
from helper import hidden_states, create_inputs

def create_bias_subspace():
    
    def create_bias_sentence_pairs():
    
        pairs = []
        temp_pairs = create_inputs(prompt_for_pairs, 0.5, 70)
        for pair in temp_pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError(f"Invalid pair format: {pair}")

            sentence1, sentence2 = pair
            pairs.append((sentence1, sentence2))
                    
        return pairs
     
    def get_CPD():
        pairs = create_bias_sentence_pairs()
        diff_vectors = []
        for sentence1, sentence2 in pairs:
            
            hidden_states1 = hidden_states(sentence1)
            hidden_states2 = hidden_states(sentence2)
            
            final_layer1 = hidden_states1[-1] 
            final_layer2 = hidden_states2[-1]

            vector1 = final_layer1.mean(dim=1) # takes average of embedings of all tokens - maybe replace with only the embedding of key word
            vector1 = vector1.squeeze()
            
            vector2 = final_layer2.mean(dim=1)
            vector2 = vector2.squeeze()
            
            d = vector1 - vector2
            diff_vectors.append(d)
        
        return torch.stack(diff_vectors)
    
    
    diff_matrix = get_CPD().float().cpu()
    bias_mean = diff_matrix.mean(dim=0, keepdim=True)
    diff_matrix_centered = diff_matrix - bias_mean
    U, S, Vh = torch.linalg.svd(diff_matrix_centered, full_matrices=False)
    
    log_S = torch.log(S + 1e-8)
    log_drops = log_S[:-1] - log_S[1:] 
    k = torch.argmax(log_drops).item() + 1
    
    bias_subspace = Vh[:k]
    return bias_subspace, bias_mean

bias_subspace, bias_mean = create_bias_subspace()