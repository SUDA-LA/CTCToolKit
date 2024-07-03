class Alignment:
    # Alignment adapted from: https://github.com/chrisjbryant/errant/blob/main/errant/alignment.py
    # Input 1: An original text string parsed by spacy
    # Input 2: A corrected text string parsed by spacy
    # Input 3: A flag for standard Levenshtein alignment
    def __init__(self, orig, cor):
        # Set orig and cor
        self.orig = orig
        self.cor = cor
        # Align orig and cor and get the cost and op matrices
        self.cost_matrix, self.op_matrix = self.align()
        # Get the cheapest align sequence from the op matrix
        self.align_seq = self.get_cheapest_align_seq()

    # Input: A flag for standard Levenshtein alignment
    # Output: The cost matrix and the operation matrix of the alignment
    def align(self):
        # Sentence lengths
        o_len = len(self.orig)
        c_len = len(self.cor)
        # Lower case token IDs (for transpositions)
        o_low = [o.lower() for o in self.orig]
        c_low = [c.lower() for c in self.cor]
        # Create the cost_matrix and the op_matrix
        cost_matrix = [[0.0 for j in range(c_len + 1)]
                       for i in range(o_len + 1)]
        op_matrix = [["O" for j in range(c_len + 1)] for i in range(o_len + 1)]
        # Fill in the edges
        for i in range(1, o_len + 1):
            cost_matrix[i][0] = cost_matrix[i - 1][0] + 1
            op_matrix[i][0] = "D"
        for j in range(1, c_len + 1):
            cost_matrix[0][j] = cost_matrix[0][j - 1] + 1
            op_matrix[0][j] = "I"

        # Loop through the cost_matrix
        for i in range(o_len):
            for j in range(c_len):
                # Matches
                if self.orig[i] == self.cor[j]:
                    cost_matrix[i + 1][j + 1] = cost_matrix[i][j]
                    op_matrix[i + 1][j + 1] = "M"
                # Non-matches
                else:
                    del_cost = cost_matrix[i][j + 1] + 1
                    ins_cost = cost_matrix[i + 1][j] + 1
                    trans_cost = float("inf")
                    # Standard Levenshtein (S = 1)
                    sub_cost = cost_matrix[i][j] + 1

                    # Costs
                    costs = [trans_cost, sub_cost, ins_cost, del_cost]
                    # Get the index of the cheapest (first cheapest if tied)
                    l = costs.index(min(costs))
                    # Save the cost and the op in the matrices
                    cost_matrix[i + 1][j + 1] = costs[l]
                    if l == 0: op_matrix[i + 1][j + 1] = "T" + str(k + 1)
                    elif l == 1: op_matrix[i + 1][j + 1] = "S"
                    elif l == 2: op_matrix[i + 1][j + 1] = "I"
                    else: op_matrix[i + 1][j + 1] = "D"
        # Return the matrices
        return cost_matrix, op_matrix

    # Get the cheapest alignment sequence and indices from the op matrix
    # align_seq = [(op, o_start, o_end, c_start, c_end), ...]
    def get_cheapest_align_seq(self):
        i = len(self.op_matrix) - 1
        j = len(self.op_matrix[0]) - 1
        align_seq = []
        # Work backwards from bottom right until we hit top left
        while i + j != 0:
            # Get the edit operation in the current cell
            op = self.op_matrix[i][j]
            # Matches and substitutions
            if op in {"M", "S"}:
                align_seq.append((op, i - 1, i, j - 1, j))
                i -= 1
                j -= 1
            # Deletions
            elif op == "D":
                align_seq.append((op, i - 1, i, j, j))
                i -= 1
            # Insertions
            elif op == "I":
                align_seq.append((op, i, i, j - 1, j))
                j -= 1
            # Transpositions
            else:
                # Get the size of the transposition
                k = int(op[1:])
                align_seq.append((op, i - k, i, j - k, j))
                i -= k
                j -= k
        # Reverse the list to go from left to right and return
        align_seq.reverse()
        return align_seq

