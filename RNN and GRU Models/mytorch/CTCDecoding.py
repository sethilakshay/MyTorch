import numpy as np

class GreedySearchDecoder(object):

    def __init__(self, symbol_set):
        """
        
        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        """

        self.symbol_set = symbol_set


    def decode(self, y_probs):
        """

        Perform greedy search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
            batch size for part 1 will remain 1, but if you plan to use your
            implementation for part 2 you need to incorporate batch_size

        Returns
        -------

        decoded_path [str]:
            compressed symbol sequence i.e. without blanks or repeated symbols

        path_prob [float]:
            forward probability of the greedy path

        """

        decoded_path = []
        blank = 0
        path_prob = 1
        
        # Iterate over sequence length - len(y_probs[0])
        for length in range(len(y_probs[0])):

            # Find the maximum probability in particular seq_length
            path_prob *= np.max(y_probs[:, length])

            # Find the most probable symbol in particular seq_length
            most_prob = np.argmax(y_probs[:, length])

            # Continuing if most probable symbol is blank and setting blank to 1
            if most_prob == 0:
                blank = 1   # Setting Blank to one
                continue
            
            # Checking if decoded path is empty or a unique symbol different from the last decode path symbol is encountered
            if decoded_path == [] or decoded_path[-1] != self.symbol_set[most_prob-1]:
                decoded_path.append(self.symbol_set[most_prob-1])
                blank = 0   # Setting Blank to zero

            # In case if blank = 1, adding the symbol regardless of repeat and setting blank to zero
            if blank == 1:
                decoded_path.append(self.symbol_set[most_prob-1])
                blank = 0   # Setting Blank to zero


        decoded_path = "".join(decoded_path)
        return decoded_path, path_prob



class BeamSearchDecoder(object):

    def __init__(self, symbol_set, beam_width):
        """

        Initialize instance variables

        Argument(s)
        -----------

        symbol_set [list[str]]:
            all the symbols (the vocabulary without blank)

        beam_width [int]:
            beam width for selecting top-k hypotheses for expansion

        """

        self.symbol_set = symbol_set
        self.beam_width = beam_width
        self.blank = 0

    def decode(self, y_probs):
        """
        
        Perform beam search decoding

        Input
        -----

        y_probs [np.array, dim=(len(symbols) + 1, seq_length, batch_size)]
			batch size for part 1 will remain 1, but if you plan to use your
			implementation for part 2 you need to incorporate batch_size

        Returns
        -------
        
        forward_path [str]:
            the symbol sequence with the best path score (forward probability)

        merged_path_scores [dict]:
            all the final merged paths with their scores

        """
        # For referance to the algorithim and pseudocode, refer Lecture-16 Slides 
        T = y_probs.shape[1]
        bestPath, FinalPathScore = None, None

        # First time instant: Initialize paths with each of the symbols,
        # including blank, using score at time t=1
        NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = self.InitializePaths(y_probs[:,0])

        for t in range(1, T):
            # Prune the collection down to the BeamWidth
            PathsWithTerminalBlank, PathsWithTerminalSymbol, self.BlankPathScore, self.PathScore = self.Prune(NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, 
                                                                                               NewBlankPathScore, NewPathScore)
            
            # First extend paths by a blank
            NewPathsWithTerminalBlank, NewBlankPathScore = self.ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, y_probs[:,t])
            # Next extend paths by a symbol
            NewPathsWithTerminalSymbol, NewPathScore = self.ExtendWithSymbol(PathsWithTerminalBlank,PathsWithTerminalSymbol, y_probs[:,t])

        # Merge identical paths differing only by the final blank
        MergedPaths, FinalPathScore = self.MergeIdenticalPaths(NewPathsWithTerminalBlank, NewBlankPathScore, NewPathsWithTerminalSymbol, NewPathScore)

        # Pick best path
        tempValue = 0
        for key, value in FinalPathScore.items():
            if value > tempValue:
                tempValue = value
                bestPath = key

        return bestPath, FinalPathScore


    # For referance to the algorithim and pseudocode, refer Lecture-16 Slides 
    def InitializePaths(self, y):

        InitialBlankPathScore = {} 
        InitialPathScore = {}
        
        # First push the blank into a path-ending-with-blank stack. No symbol has been invoked yet
        path = ''
        InitialBlankPathScore[path] = y[self.blank] # Score of blank at t=1
        InitialPathsWithFinalBlank = [path]

        # Push rest of the symbols into a path-ending-with-symbol stack
        InitialPathsWithFinalSymbol = []

        for idx in range(len(self.symbol_set)): # This is the entire symbol set, without the blank
            path = self.symbol_set[idx]
            InitialPathScore[path] = y[idx+1] # Score of symbol c at t=1
            InitialPathsWithFinalSymbol.append(path) # Set addition
        
        return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore


    # For referance to the algorithim and pseudocode, refer Lecture-16 Slides 
    def ExtendWithBlank(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, y):
        UpdatedPathsWithTerminalBlank = []
        UpdatedBlankPathScore = {}

        ## First work on paths with terminal blanks
        #(This represents transitions along horizontal trellis edges for blanks)

        for path in PathsWithTerminalBlank:
            UpdatedPathsWithTerminalBlank.append(path) # Set addition
            UpdatedBlankPathScore[path] = self.BlankPathScore[path]*y[self.blank]

        # Then extend paths with terminal symbols by blanks
        for path in PathsWithTerminalSymbol:
            # If there is already an equivalent string in UpdatesPathsWithTerminalBlank
            # # simply add the score. If not create a new entry
            if path in UpdatedPathsWithTerminalBlank:
                UpdatedBlankPathScore[path] += self.PathScore[path] * y[self.blank]
            else:
                UpdatedPathsWithTerminalBlank.append(path) # Set addition
                UpdatedBlankPathScore[path] = self.PathScore[path] * y[self.blank]
        
        return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore


    # For referance to the algorithim and pseudocode, refer Lecture-16 Slides 
    def ExtendWithSymbol(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, y):
        UpdatedPathsWithTerminalSymbol = []
        UpdatedPathScore = {}

        # First extend the paths terminating in blanks. This will always create a new sequence
        for path in PathsWithTerminalBlank:
            for idx in range(len(self.symbol_set)): # SymbolSet does not include blanks
                newpath = path + self.symbol_set[idx] # Concatenation
                UpdatedPathsWithTerminalSymbol.append(newpath) # Set addition
                UpdatedPathScore[newpath] = self.BlankPathScore[path] * y[idx+1]
    
        # Next work on paths with terminal symbols
        for path in PathsWithTerminalSymbol:
            # Extend the path with every symbol other than blank
            for idx in range(len(self.symbol_set)): # SymbolSet does not include blanks
                # Horizontal transitions don’t extend the sequence
                newpath = path if self.symbol_set[idx] == path[-1] else path + self.symbol_set[idx] 
                
                if newpath in UpdatedPathsWithTerminalSymbol:
                    # Already in list, merge paths
                    UpdatedPathScore[newpath] += self.PathScore[path] * y[idx+1]
                else:
                    # Create new path
                    UpdatedPathsWithTerminalSymbol.append(newpath) # Set addition
                    UpdatedPathScore[newpath] = self.PathScore[path] * y[idx+1]

        return UpdatedPathsWithTerminalSymbol, UpdatedPathScore


    # For referance to the algorithim and pseudocode, refer Lecture-16 Slides 
    def Prune(self, PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore):
        PrunedBlankPathScore = {}
        PrunedPathScore = {}
        scorelist = []

        # First gather all the relevant scores
        for p in PathsWithTerminalBlank:
            scorelist.append(BlankPathScore[p])
        
        for p in PathsWithTerminalSymbol:
            scorelist.append(PathScore[p])

        # Sort and find cutoff score that retains exactly BeamWidth paths
        scorelist.sort(reverse = True)
        cutoff = scorelist[self.beam_width-1] if self.beam_width < len(scorelist) else scorelist[-1]

        PrunedPathsWithTerminalBlank = []
        for p in PathsWithTerminalBlank:
            if BlankPathScore[p] >= cutoff:
                PrunedPathsWithTerminalBlank.append(p) # Set addition
                PrunedBlankPathScore[p] = BlankPathScore[p]

        PrunedPathsWithTerminalSymbol = []
        for p in PathsWithTerminalSymbol:
            if PathScore[p] >= cutoff:
                PrunedPathsWithTerminalSymbol.append(p) # Set addition
                PrunedPathScore[p] = PathScore[p]

        return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore


    # For referance to the algorithim and pseudocode, refer Lecture-16 Slides 
    def MergeIdenticalPaths(self, PathsWithTerminalBlank, BlankPathScore,PathsWithTerminalSymbol, PathScore):
        # All paths with terminal symbols will remain
        MergedPaths = PathsWithTerminalSymbol
        FinalPathScore = PathScore

        # Paths with terminal blanks will contribute scores to existing identical paths from 
        # PathsWithTerminalSymbol if present, or be included in the final set, otherwise
        for p in PathsWithTerminalBlank:
            if p in MergedPaths:
                FinalPathScore[p] += BlankPathScore[p]
            else:
                MergedPaths.append(p) # Set addition
                FinalPathScore[p] = BlankPathScore[p]

        return MergedPaths, FinalPathScore