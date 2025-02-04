# -*- coding:utf-8 -*-
from __future__ import annotations

import re
import evaluate
import pandas as pd

print(f"loading: {__file__}")

pattern_non_word_char_repetition = re.compile(r"[\s\W]{5,}")
pattern_text_repetitions = re.compile(
    r"(?P<repeat>.{5}.*?)(?:[\s\W]*(?P=repeat))+", re.M | re.DOTALL | re.IGNORECASE
)
# Explanation of the Regex Pattern:
#   (?P<repeat>.{5}.*?): Captures any sequence of characters with minimal length of 5 and names this group repeat.
#     .*?: Matches zero or more characters, non-greedily (as few as possible).
#   (?:[\s\W]+(?P=repeat))+: A non-capturing group that matches one or more repetitions of:
#     [\s\W]+: One or more whitespace or non-word characters (spaces, punctuation, etc.).
#     (?P=repeat): A backreference to the named group repeat.


def del_non_word_char_repetition(text, debug=False):
    count = 0

    if isinstance(text, str):
        if debug:
            print("----detect non-word characters repetition----")
        count = len(text)
        text = pattern_non_word_char_repetition.sub("\t", text)
        count -= len(text)
        if debug and count:
            print(f"removed non-word characters repetition: {count}")
    return text, count


# final version for repetition detection
def detect_text_repetitions(text, debug=False):
    count = 0

    if isinstance(text, str):
        if debug:
            print("----detect text repetitions----")
        matches = pattern_text_repetitions.finditer(text)
        for match in matches:
            if debug:
                print(match)
                for groupNum in range(0, len(match.groups())):
                    groupNum = groupNum + 1
                    print(
                        "Group {groupNum} found at {start}-{end}: `{group}`".format(
                            groupNum=groupNum,
                            start=match.start(groupNum),
                            end=match.end(groupNum),
                            group=match.group(groupNum),
                        )
                    )

            start, end = match.span()
            count += end - start - len(match.group(1))

    return count


def detect_repetitions(text, debug=False):
    if isinstance(text, str) is False:
        return 0, 0, 0
    text, count_non_word_char_repetition = del_non_word_char_repetition(
        text, debug=debug
    )
    count_text_repetitions = detect_text_repetitions(text, debug=debug)
    total_repetitions = count_non_word_char_repetition + count_text_repetitions

    result = (count_non_word_char_repetition, count_text_repetitions, total_repetitions)

    if debug:
        print(result)
    return result


def calc_perf_scores(predictions, references, debug=False):
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    bert_score = evaluate.load("bertscore")

    if debug:
        print("predictions:", predictions)
        print("references:", references)

    bleu_scores = bleu.compute(
        predictions=predictions, references=references, max_order=1
    )
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    bert_scores = bert_score.compute(
        predictions=predictions,
        references=references,
        lang="en",
        model_type="microsoft/deberta-large-mnli",
    )
    result = {
        "bleu_scores": bleu_scores,
        "rouge_scores": rouge_scores,
        "bert_scores": bert_scores,
    }

    if debug:
        print("result:", result)

    return result
