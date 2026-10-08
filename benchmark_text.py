"""Fixed tokenization and phrase rules; no fitted semantic representation."""
import re

TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*|[^\W\d_]+(?:['’][^\W\d_]+)*|[^\w\s]|_", re.UNICODE)

def tokenize(text):
    return TOKEN_RE.findall(text)

# Written before validation evaluation; matched case-insensitively at word boundaries.
# Natural topic words are intentional. No company names or complaint metadata.
LEXICAL_CUES = {
    'Advertising and marketing, including promotional offers':
        ['promotional offer','promotion','advertised','advertising','sign up bonus','signup bonus','welcome bonus'],
    'Closing your account':
        ['closed my account','account closure','close my account','closed account','account was closed','cancel my card'],
    'Fees or interest':
        ['interest rate','interest charge','annual fee','late fee','fees','apr'],
    'Getting a credit card':
        ['credit card application','application denied','denied my application','applied for','never applied','unsolicited card'],
    'Incorrect information on your report':
        ['inaccurate reporting','incorrect information','credit report','late payment remarks','reporting inaccurately'],
    'Other features, terms, or problems':
        ['rewards','reward points','balance transfer','credit limit','cash advance','customer service'],
    'Problem when making payments':
        ['autopay','automatic payment','payment hold','payment was not credited','make a payment','payment processing'],
    "Problem with a company's investigation into an existing problem":
        ['investigation','failed to investigate','dispute results','verified as accurate','reinvestigation'],
    'Problem with a purchase shown on your statement':
        ['unauthorized charge','unauthorized charges','fraudulent charge','fraudulent charges','chargeback','refund','merchant'],
    'Struggling to pay your bill':
        ['financial hardship','hardship','unable to pay','cannot afford','lost my job','payment assistance'],
    'Trouble using your card':
        ['card declined','card was declined','declined transaction','activate my card','card activation','replacement card','blocked my card'],
}

def lexical_predict(texts, labels, counts):
    patterns={label:[re.compile(r'(?<!\w)'+re.escape(cue)+r'(?!\w)',re.I)
                     for cue in LEXICAL_CUES[label]] for label in labels}
    predictions=[]; evidence=[]
    for text in texts:
        hits={label:[cue for cue,pattern in zip(LEXICAL_CUES[label],patterns[label]) if pattern.search(text)]
              for label in labels}
        scores={label:len(hits[label]) for label in labels}
        max_score=max(scores.values())
        tied=[label for label in labels if scores[label]==max_score]
        winner=sorted(tied,key=lambda label:(-counts[label],labels.index(label)))[0]
        predictions.append(winner)
        evidence.append({'matched_cues':' | '.join(hits[winner]),'max_score':max_score,
                         'tie_count':len(tied),'majority_fallback':max_score==0})
    return predictions,evidence
