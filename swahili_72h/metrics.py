"""No downloaded evaluation packages; explicit exploratory text metrics."""
from collections import Counter
import math
import re


def chrf(ref,hyp):
    # chrF beta=2, character orders 1..6, whitespace removed, macro over orders.
    ref=''.join(ref.split());hyp=''.join(hyp.split());scores=[]
    for n in range(1,7):
        a=Counter(ref[i:i+n] for i in range(len(ref)-n+1));b=Counter(hyp[i:i+n] for i in range(len(hyp)-n+1))
        overlap=sum((a&b).values());r=overlap/max(1,sum(a.values()));p=overlap/max(1,sum(b.values()))
        scores.append(5*p*r/(4*p+r) if p+r else 0)
    return 100*sum(scores)/6


def diversity(texts):
    words=[re.findall(r'\w+|[^\w\s]',t) for t in texts]
    out=dict(samples=len(texts),empty=sum(not t.strip() for t in texts),unique=len(set(texts)),mean_words=sum(map(len,words))/max(1,len(words)))
    for n in (1,2,3,4):
        counts=Counter(tuple(w[i:i+n]) for w in words for i in range(len(w)-n+1));out[f'distinct_{n}']=len(counts)/max(1,sum(counts.values()))
    return out


def aggregate(scores):
    n=sum(x['tokens'] for x in scores);total=sum(x['nll'] for x in scores)
    return dict(samples=len(scores),empty=sum(x['empty'] for x in scores),tokens=n,nll=total,
                token_weighted_perplexity=math.exp(min(700,total/n)) if n else None,
                mean_sample_perplexity=sum(x['perplexity'] for x in scores if x['perplexity'] is not None)/max(1,sum(x['tokens']>0 for x in scores)))
