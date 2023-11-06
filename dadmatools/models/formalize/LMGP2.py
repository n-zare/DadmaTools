from datetime import datetime
from itertools import product, islice
from transformers import AutoTokenizer, AutoModelWithLMHead
import torch
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import torch.nn as nn


class GP2LM:
    NIM_FASELE = '‌'
    def __init__(self):
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        # self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("saved_models/parsbert/parsbert/")
        self.model = AutoModelWithLMHead.from_pretrained("saved_models/parsbert/parsbert/",gradient_checkpointing=True)
        self.model = self.model.to(self.device)
        self.MAX_LEN = 15
        self.MAX_BATCH_SIZE = 64
        self.tokenizer.model_max_length = self.MAX_LEN

    def get_scores_by_gpt2(self, id_seqs):
        ppl_scores = {}
        input_ids = []
        words_count = []
        index2id = {}
        for index, id in enumerate(id_seqs):
            index2id[index] = id
            tokenize_input = self.tokenizer.encode(id_seqs[id])
            # for id in tokenize_input:
                # print(str(id), self.tokenizer.decode([id]))
            words_count.append(len(tokenize_input))
            input_ids.append(tokenize_input)
        max_len = min([len(t) for t in input_ids])
        input_ids = pad_sequences(input_ids,maxlen=max_len , dtype="long",
                                  value=0, truncating="post", padding="post")
        attention_masks = []
        for sent in input_ids:
        #     # Create the attention mask.
        #     #   - If a token ID is 0, then it's padding, set the mask to 0.
        #     #   - If a token ID is > 0, then it's a real token, set the mask to 1.
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)
        prediction_masks = torch.tensor(attention_masks).to(self.device)
        tensor_input = torch.tensor(input_ids).to(self.device)
        outputs = self.model(tensor_input, labels=tensor_input, attention_mask=prediction_masks)
        # outputs = self.model(tensor_input, labels=tensor_input)
        logits = outputs[1]
        for i in range(len(tensor_input)):
            tensor_input_i = tensor_input[i, :]
            logits_i = logits[i, :]
            if tensor_input is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits_i[..., :-1, :].contiguous()
                shift_labels = tensor_input_i[..., 1:].contiguous()
                # Flatten the tokens
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                ppl = np.exp(loss.cpu().detach().numpy())
                ppl_scores[index2id[i]] = ppl
                # ppl_scores.append(ppl)
        return ppl_scores

    def chunk(self, it, size):
        it = iter(it)
        return iter(lambda: tuple(islice(it, size)), ())

    def split_to_batch_size(self, list_of_candidates):
        batchs = []
        current_batch_size = 0
        current_batch_indexs = [0]

        for i in range(1, len(list_of_candidates)):
            if len(list_of_candidates[i]) + current_batch_size < self.MAX_BATCH_SIZE:
                current_batch_size += len(list_of_candidates)
                current_batch_indexs.append(i)
            else:
                new_b = [list_of_candidates[indx] for indx in current_batch_indexs]
                current_batch_indexs = [i]
                current_batch_size = len(list_of_candidates[i])
                batchs.append(new_b)
        new_b = [list_of_candidates[indx] for indx in current_batch_indexs]
        batchs.append(new_b)
        return batchs

    def get_batch(self, list_of_candidates):
        candidates_batchs = self.split_to_batch_size(list_of_candidates)
        batchs_size_items = []
        for l_c in candidates_batchs:
            batch = []
            sizes_of_items = []
            for item_index, item in enumerate(l_c):
                all_seqs = list(product(*item))
                sizes_of_items.append(len(all_seqs))
                batch.extend(all_seqs)
            batch = [' '.join(b) for b in batch]
            batchs_size_items.append((batch, sizes_of_items))
        return batchs_size_items

    def get_best_candidates(self, id_seqs_dict):
        id_seqs_list = [(id, txt) for id, txt in id_seqs_dict.items()]
        batchs = self.chunk(id_seqs_list, self.MAX_BATCH_SIZE)
        all_min = 10000000000000000000000
        for b in batchs:
            b_dict = {id:txt for id,txt in b}
            id_ppl_scores = self.get_scores_by_gpt2(b_dict)
            min_ppl = min(id_ppl_scores.values())
            if min_ppl < all_min:
                all_min = min_ppl
                min_id = [id for id, ppl in id_ppl_scores.items() if ppl ==min_ppl][0]
        return min_id, all_min
        # batchs_size_items = self.get_batch(list_of_candidates)
        # all_best_candidates = []
        # for batch, size_of_items in batchs_size_items:
            # sizes_of_items = [0] + size_of_items
            # ppl_scores_of_each_item = [ppl_scores[sum(sizes_of_items[:i]):sum(sizes_of_items[:i]) + sizes_of_items[i]] for i
            #                            in
            #                            range(1, len(sizes_of_items))]
            # best_candidates_indexs = [sum(sizes_of_items[:i + 1]) + np.argmin(item) for i, item in
            #                           enumerate(ppl_scores_of_each_item)]
            # best_candidates = [batch[i] for i in best_candidates_indexs]
            # all_best_candidates.extend(best_candidates)

        # return all_best_candidates

    def is_exist_lemmas_in_tokenizer(self, word):
                toks = word.split()
                for t in toks:
                    if len(self.tokenizer.encode(t)) > 3:
                        return False
                return True
if __name__ == '__main__':
    a = 'امروز منم از همینجا بهشون سلام میرسونم'
    all_a = [a] * 400
    inputs = {i : sent for i, sent in enumerate(all_a)}
    print(len(inputs))
    print(inputs)
    # inputs = {'1': 'چرخاندن', '2': 'چرخانده هستند', '3': 'چرخانده اما بودند'}

    t0 = datetime.now()
    lm = GP2LM()
    print(datetime.now() - t0)
    t1 = datetime.now()
    bests = lm.get_best_candidates(inputs)
    print(datetime.now() - t1)
    print(bests)
