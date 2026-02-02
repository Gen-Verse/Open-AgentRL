from accelerate.logging import get_logger
logger = get_logger(__name__, log_level="INFO")


import torch
class UniversalPrompting():
    def __init__(self, text_tokenizer,
                 max_prompt_len=8000, max_gen_length=377, ignore_id=-100):
        """
        :param text_tokenizer: original text tokenizer
        """
        self.text_tokenizer = text_tokenizer
        self.max_gen_length = max_gen_length
        self.max_prompt_len = max_prompt_len


    # language modeling
    def lm_prompt(self, text_ids_pairs):
        prompts_list, responses_list = text_ids_pairs
        pad_id = self.text_tokenizer.pad_token_id

        # 计算每条序列的总长度 = prompt + response + eos
        if responses_list.shape[1] < self.max_gen_length:
            max_seq_len = prompts_list.shape[1] + responses_list.shape[1]
        else:
            max_seq_len = prompts_list.shape[1] + self.max_gen_length

        sequence_ids = []
        attention_masks = []
        label_ids = []

        for prompt_ids, resp_ids in zip(prompts_list, responses_list):
            prompt_ids = prompt_ids.tolist()
            resp_ids   = resp_ids.tolist()

            # 拼接 prompt + response + EOS
            temp_ids = prompt_ids + resp_ids
            temp_masks = [1] * len(temp_ids)
            temp_labels = temp_ids.copy()

            # padding 或截断到 max_seq_len
            if len(temp_ids) < max_seq_len:
                pad_len = max_seq_len - len(temp_ids)
                temp_ids.extend([pad_id] * pad_len)
                temp_labels.extend([pad_id] * pad_len)
                temp_masks.extend([0] * pad_len)
            else:
                temp_ids = temp_ids[:max_seq_len]
                temp_labels = temp_labels[:max_seq_len]
                temp_masks = temp_masks[:max_seq_len]

            # 转为张量并累积
            sequence_ids.append(torch.tensor(temp_ids).unsqueeze(0))
            attention_masks.append(torch.tensor(temp_masks).unsqueeze(0))
            label_ids.append(torch.tensor(temp_labels).unsqueeze(0))

        input_ids = torch.cat(sequence_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        label_ids = torch.cat(label_ids, dim=0)
        

        return input_ids, label_ids, prompts_list.shape[1]

        
    

    def mask_prompt(self):
        pass

    def __call__(self, input):
        prompts, responses = input

        enc = self.text_tokenizer(
            prompts,
            padding=False,
            truncation=False,
            return_length=True
        )
        lengths = enc["length"]
        # 2) 过滤出长度 <= max_len 的 indices
        keep_indices = [i for i, L in enumerate(lengths) if L <= self.max_prompt_len]
        drop_num = len(prompts) - len(keep_indices)
        
        prompts  = [prompts[i]  for i in keep_indices]
        responses = [responses[i] for i in keep_indices]

        # 使用 tokenizer 将 raw text 转为 token ids
        prompt_ids = self.text_tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            padding_side = "left"
        )['input_ids']
        response_ids = self.text_tokenizer(
            responses,
            padding=True,
            return_tensors="pt",
            padding_side = "right"
        )['input_ids']
        input_ids_lm, labels_lm, start_pos = self.lm_prompt((prompt_ids, response_ids))
        return input_ids_lm, labels_lm, start_pos, drop_num


if __name__ == '__main__':
    pass