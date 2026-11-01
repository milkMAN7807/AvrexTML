import numpy as np
import math

def pack_ternary(arr):
    num_in_pack = 4 # How many ternary weights are in the pack
    new_arr = np.zeros(int(math.ceil(len(arr) / num_in_pack)), dtype = int)

    cur_byte = 0
    for i, weight in enumerate(arr):
        cur_byte += (weight + 1) << (2 * (i % num_in_pack)) # Shift to current pos

        if ((i + 1) % num_in_pack) == 0:
            new_arr[int(i / 4)] = cur_byte
            cur_byte = 0 # Reset buffer
        elif i == len(arr) - 1:
            new_arr[int(i / 4)] = cur_byte

    return new_arr

def validate_ternary_tensor(tensor):
    tensor_vals = np.unique(tensor)

    if len(tensor_vals) > 3:
         return False
    elif len(tensor_vals) == 3:
        return (tensor_vals[0] == -tensor_vals[2]) and (tensor_vals[1] == 0)
    elif len(tensor_vals) == 1:
        return tensor_vals[0] == 0
    
    return False

# Should prob pass a valid tensor to this
def normalize_ternary_tensor(tensor):
    normalized_tensor = np.sign(tensor).astype(np.int8)
    return normalized_tensor

# Assuming valid tensor
def get_tensor_alpha(tensor):
    return np.max(np.unique(np.abs(tensor)))

def search_terms(alpha, max_exponent, min_exponent, gamma, terms: list, signs: list, height): # 2^n for exponents
    if alpha == 0:
        raise ValueError("alpha = 0 detected, compilation prevented because network is guaranteed to not work, fix please")
    if (height == 0) or (max_exponent < min_exponent):
        return gamma, terms, signs # beta, terms, signs

    optimal = gamma, terms, signs
    lowest_error = (math.inf, math.inf, math.inf)

    def error_less_than(e1, e2):
        for i in range(len(e1)):
            if e1[i] < e2[i]: return True
            elif e2[i] < e1[i]: return False

        return False # Identical solutions

    for i in range(max_exponent, min_exponent - 1, -1):
        pos_terms = terms + [i]
        neg_terms = terms + [i]

        new_p_gamma = gamma + 2 ** i
        pos_signs = signs + [1]
        p_beta, new_pos_terms, new_pos_signs = search_terms(alpha, i - 1, min_exponent, new_p_gamma, pos_terms, pos_signs, height - 1)

        new_n_gamma = gamma - 2 ** i
        neg_signs = signs + [-1]
        n_beta, new_neg_terms, new_neg_signs = search_terms(alpha, i - 1, min_exponent, new_n_gamma, neg_terms, neg_signs, height - 1)

        get_error = lambda a, b: np.abs(a - b)

        def sum_terms(l):
            sum = 0
            for v in l:
                sum += abs(v)

            return sum

        def get_min_term(s, f):
            min_term = math.inf
            for t in s:
                min_term = min(t[f], min_term)

            return min_term

        # Ideally we want the solution with the least error, but fewer terms are better, and then fewer shifts (avr has no barrel shifter)
        # This felt really wrong to assume but I found counterexample by chance so... 2^3 + 2^3 + 2^6 = 2^5 + 2^5 + 2^4 -> sum of terms_l = 12 sum of terms_r = 14
        # Also prioritizing fewer terms is better because avr-gcc can optimize single terms but not multiple
        p_error = (get_error(alpha, p_beta), len(new_pos_terms), sum_terms(new_pos_terms))
        n_error = (get_error(alpha, n_beta), len(new_neg_terms), sum_terms(new_neg_terms))
        stop_p_error = (get_error(alpha, new_p_gamma), len(pos_terms), sum_terms(pos_terms))
        stop_n_error = (get_error(alpha, new_n_gamma), len(neg_terms), sum_terms(neg_terms))

        error_tuple = (p_error, n_error, stop_p_error, stop_n_error)
        branch_error = (math.inf, math.inf, math.inf)

        for j in range(3):
            min_error = get_min_term(error_tuple, j)
            error_tuple = tuple(t for t in error_tuple if t[j] == min_error)

            list_t = list(branch_error)
            list_t[j] = min_error
            branch_error = tuple(list_t)

        if error_less_than(branch_error, lowest_error):
            if p_error == branch_error:
                optimal = p_beta, new_pos_terms, new_pos_signs
            elif n_error == branch_error:
                optimal = n_beta, new_neg_terms, new_neg_signs 
            elif stop_p_error == branch_error:
                optimal = new_p_gamma, pos_terms, pos_signs
            elif stop_n_error == branch_error:
                optimal = new_n_gamma, neg_terms, neg_signs

            lowest_error = branch_error

    return optimal

def generate_alpha_approx(alpha):
    num_terms = 3
    n_vals = []
    signs = []

    beta, n_vals, signs = search_terms(alpha, max_exponent = 15, min_exponent = -15, gamma = 0, terms = [], signs = [], height = num_terms)
    return n_vals, signs
        
def biases_to_fixedp(biases, dp):
    return np.round((2 ** dp) * biases).astype(np.int16)

"""
beta, terms, signs = search_terms(alpha = 0.625, max_exponent = 15, min_exponent = -15, gamma = 0, terms = [], signs = [], height = 3)

print(beta)

for i in range(len(terms)):
    print(signs[i])
    print(terms[i])
    print('\n')
    pass
"""