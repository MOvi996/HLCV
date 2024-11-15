import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, max_seq_length=20):
        
        super(DecoderLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.max_seq_length = max_seq_length
        
    def forward(self, features, captions, lengths):
        
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1) # joint vision and language embedding
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """
        Greedy search approach for generating captions.
        """
        sampled_ids = []
        inputs = features.unsqueeze(1)
        #########################
        # TODO
        # Use the lstm defined in init to generate token ids for the image features
        # For the first token, the input should be the image features
        # For each subsequent step, feed the predicted token as the input to the lstm
        # Before feeding the token to the lstm, embed the token using the self.embed layer
        # Note that at each step, the output is a linear transformation of the lstm hidden state
        # The output is a vector of size vocab_size
        # You can take hints from the sample_beam_search function below
        # idx_sequences = [[[], 0.0, inputs, states]]

        for _ in range(self.max_seq_length):
            all_candidates = []
            # for idx_seq in idx_sequences:
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))

            predicted_ids = torch.argmax(outputs, dim=1)

            sampled_ids.append(predicted_ids.item())

            inputs = self.embed(predicted_ids.unsqueeze(1))
        
        #########################
        return np.asarray(sampled_ids)
        
    def sample_beam_search(self, features, beam_width=5, states=None):
        """
        Beam search approach for generating captions.
        """
        
        # Top word idx sequences and their corresponding inputs and states
        inputs = features.unsqueeze(1)
        idx_sequences = [[[], 0.0, inputs, states]]
        
        for _ in range(self.max_seq_length):
            # Store all the potential candidates at each step
            all_candidates = []
            # Predict the next word idx for each of the top sequences
            for idx_seq in idx_sequences:
                hiddens, states = self.lstm(idx_seq[2], idx_seq[3])
                outputs = self.linear(hiddens.squeeze(1))
                
                # Transform outputs to log probabilities to avoid floating-point 
                # underflow caused by multiplying very small probabilities
                log_probs = F.log_softmax(outputs, -1)
                top_log_probs, top_idx = log_probs.topk(beam_width, 1)
                top_idx = top_idx.squeeze(0)
                
                # create a new set of top sentences for next round
                for i in range(beam_width):
                    next_idx_seq, log_prob = idx_seq[0][:], idx_seq[1]
                    next_idx_seq.append(top_idx[i].item())
                    log_prob += top_log_probs[0][i].item()
                    
                    # Indexing 1-dimensional top_idx gives 0-dimensional tensors.
                    # We have to expand dimensions before embedding them
                    inputs = self.embed(top_idx[i].unsqueeze(0)).unsqueeze(1)
                    all_candidates.append([next_idx_seq, log_prob, inputs, states])
            
            # Keep only the top sequences according to their total log probability
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            idx_sequences = ordered[:beam_width]
        
        return np.array(idx_sequences[0][0])
    
    def decode_caption(self, captions, vocab):
        """Accept a list of word ids and return the corresponding sentence."""
        singleton = False
        if captions.ndim == 1:
            singleton = True
            captions = captions[None]
        #########################
        # TODO
        # Your input caption can either be a single caption or a batch of captions
        # Your code should work for both cases
        # Refer to utils/build_vocab.py to see how the vocab object is constructed
        # That should give you an idea on how to use the vocab object to convert word ids to words
        # Specifically, your final sentences should not contain the <start>, <pad> and <end> tokens
        sentences = []

        for caption in captions:
            words = [vocab.idx2word[word_idx] for word_idx in caption if vocab.idx2word[word_idx] not in ['<pad>','<start>','<end>']]
            sentence = " ".join(words)
            sentences.append(sentence)
        
        if singleton:
            return sentences[0]
        return sentences
        
        # raise NotImplementedError
        #########################