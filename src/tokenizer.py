import json
from typing import List, Dict, Tuple

def bytes_to_unicode() -> Dict[int, str]:
    """Ստեղծում է քարտեզագրում բայթերի (0..255) և Unicode սիմվոլների միջև:
    
    Սա թույլ է տալիս բացատները և հատուկ նշանները ճիշտ կոդավորել, ինչպես Qwen/GPT մոդելներում:
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + 
        list(range(ord("¡"), ord("¬") + 1)) + 
        list(range(ord("®"), ord("¶") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs_str = [chr(n) for n in cs]
    return dict(zip(bs, cs_str))


class CustomTokenizer:
    """Զրոյից իրականացված իսկական Byte-Level BPE Tokenizer:"""

    def __init__(self, vocab_path: str, merges_path: str):
        with open(vocab_path, 'r', encoding='utf-8') as f:
            self.vocab: Dict[str, int] = json.load(f)
        
        self.index_to_token: Dict[int, str] = {v: k for k, v in self.vocab.items()}
        
        # Բայթերի փոխարկիչ
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
        # Բեռնում ենք merges.txt-ը և ստեղծում ենք զույգերի հերթականությունը (առաջնահերթությունը)
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}
        with open(merges_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Եթե առաջին տողը մեկնաբանություն է (#version), բաց թողնել
            start_idx = 1 if lines[0].startswith("#") else 0
            for idx, line in enumerate(lines[start_idx:]):
                line = line.strip()
                if not line:
                    continue
                parts = tuple(line.split())
                if len(parts) == 2:
                    self.bpe_ranks[parts] = idx

    def _get_pairs(self, word: List[str]) -> set[Tuple[str, str]]:
        """Վերադարձնում է բառի մեջ կողք կողքի գտնվող բոլոր սիմվոլների զույգերը:"""
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i+1]))
        return pairs

    def encode(self, text: str) -> List[int]:
        """Տեքստը վերածում է ID-ների՝ ըստ Byte-Level BPE կանոնների:"""
        if not text:
            return []

        # 1. Տեքստի ամեն մի սիմվոլ վերածում ենք իր հատուկ BPE Unicode տեսքին
        tokenized_bytes = []
        for b in text.encode("utf-8"):
            tokenized_bytes.append(self.byte_encoder[b])
        
        # Սա մեր սկզբնական թոքենների ցուցակն է (տառ առ տառ/բայթ առ բայթ)
        word = tokenized_bytes
        pairs = self._get_pairs(word)

        if not pairs:
            return [self.vocab.get(tokenized_bytes[0])]

        # 2. Միավորման (Merge) հիմնական ցիկլը
        while True:
            # Գտնում ենք այն զույգը, որն ունի ամենաբարձր առաջնահերթությունը (ամենափոքր rank-ը)
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            
            # Եթե ընտրված զույգը չկա merges-ի մեջ, ավարտում ենք
            if bigram not in self.bpe_ranks:
                break
                
            # Միավորում ենք ընտրված զույգը ամբողջ ցուցակում
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == bigram:
                    new_word.append(word[i] + word[i+1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
            pairs = self._get_pairs(word)
            if not pairs:
                break

        # 3. Թոքենները վերածում ենք ID-ների
        return [self.vocab[t] for t in word if t in self.vocab]

    def decode(self, token_ids: List[int]) -> str:
        """ID-ները հետ է վերածում բնական տեքստի՝ հաշվի առնելով բայթերը:"""
        text = "".join([self.index_to_token.get(tid, "") for tid in token_ids])
        # Փոխարկում ենք Unicode նշանները հետ՝ հում բայթերի
        byte_tokens = bytes([self.byte_decoder[c] for c in text])
        return byte_tokens.decode("utf-8", errors="replace")