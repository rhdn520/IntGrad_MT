import spacy
from zss import Node, simple_distance
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer

import spacy
from zss import Node, simple_distance  # Assuming you're using the zss library for tree operations.

class TreeEditDistanceCalculator:
    def __init__(self, model='en_core_web_trf'):
        """Initialize the calculator with a SpaCy model."""
        self.nlp = spacy.load(model)
        self.tree_cache = {}  # Cache to store sentence trees

    def build_tree(self, sentence):
        """Parse a sentence and build a tree from the parsing, or use cached tree if available."""
        if sentence in self.tree_cache:
            return self.tree_cache[sentence]
        
        doc = self.nlp(sentence)
        root = [token for token in doc if token.head == token][0]  # Root of the parsing tree

        def add_children(nlp_token):
            node = Node(nlp_token.orth_)
            for child in nlp_token.children:
                node.addkid(add_children(child))
            return node

        tree = add_children(root)
        self.tree_cache[sentence] = tree  # Cache the built tree
        return tree

    def cal_distance(self, sentence1, sentence2):
        """Calculate and return the tree edit distance between two sentences."""
        tree1 = self.build_tree(sentence1)
        tree2 = self.build_tree(sentence2)
        distance = simple_distance(tree1, tree2)
        return distance


class SBERTSimCalculator:
    def __init__(self, model='sentence-transformers/all-mpnet-base-v2'):
        """Initialize the calculator with a Sentence Transformers model."""
        self.model = SentenceTransformer(model, device="cpu")
        self.embed_cache = {}  # Cache to store sentence embeddings
    
    def get_embedding(self, sent):
        if sent in self.embed_cache:
            embedding = self.embed_cache[sent]
        else:
            encoded = self.model.encode(sent, convert_to_tensor=True)
            self.embed_cache[sent] = encoded
            embedding = encoded
        return embedding


    def cal_sbert_sim(self, sent1, sent2):
        """Calculate and return the SBERT cosine similarity between two sentences."""
        embeddings = []
        for sent in [sent1, sent2]:
            if sent in self.embed_cache:
                embeddings.append(self.embed_cache[sent])
            else:
                # Encode and store in cache if sentence is not already encoded
                encoded = self.model.encode(sent, convert_to_tensor=True)
                self.embed_cache[sent] = encoded
                embeddings.append(encoded)

        # Calculate cosine similarity using util.pytorch_cos_sim
        sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        return sim.item()
    
    def cal_sbert_euc_dist(self, sent1, sent2):
        """Calculate and return the SBERT cosine similarity between two sentences."""
        embeddings = []
        for sent in [sent1, sent2]:
            if sent in self.embed_cache:
                embeddings.append(self.embed_cache[sent])
            else:
                # Encode and store in cache if sentence is not already encoded
                encoded = self.model.encode(sent, convert_to_tensor=True)
                self.embed_cache[sent] = encoded
                embeddings.append(encoded)
        euc_dist = util.euclidean_sim(embeddings[0], embeddings[1])
        # euc_dist = np.linalg.norm(embeddings[0].cpu()-embeddings[1].cpu())
        return -(euc_dist.item())


class LevenshteinDistanceCalculator:
    def __init__(self, model="bert-base-cased"):
        """Initialize the tokenizer with a pretrained BERT model."""
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.token_cache = {}  # Cache to store tokenized sentences

    def tokenize(self, sentence):
        """Tokenize the sentence and cache the result if not already cached."""
        if sentence in self.token_cache:
            return self.token_cache[sentence]
        tokens = self.tokenizer.tokenize(sentence)
        self.token_cache[sentence] = tokens
        return tokens

    def cal_levenshtein_distance_w_tokens(self, sentence1, sentence2):
        """Calculate and return the Levenshtein distance between tokenized versions of two sentences."""
        tokens1 = self.tokenize(sentence1)
        tokens2 = self.tokenize(sentence2)

        # Create a distance matrix
        rows = len(tokens1) + 1
        cols = len(tokens2) + 1
        dist = [[0 for _ in range(cols)] for _ in range(rows)]

        # Initialize the distance matrix
        for i in range(1, rows):
            dist[i][0] = i
        for j in range(1, cols):
            dist[0][j] = j

        # Compute Levenshtein distance
        for i in range(1, rows):
            for j in range(1, cols):
                if tokens1[i - 1] == tokens2[j - 1]:
                    cost = 0
                else:
                    cost = 1
                dist[i][j] = min(
                    dist[i - 1][j] + 1,  # Deletion
                    dist[i][j - 1] + 1,  # Insertion
                    dist[i - 1][j - 1] + cost  # Substitution
                )

        return dist[-1][-1]