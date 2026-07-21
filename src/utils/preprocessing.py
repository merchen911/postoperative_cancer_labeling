import os
import re
import json
import numpy as np
import pandas as pd
from collections import defaultdict



def revision_checker(rdf, drop_duplicates = False):

    novel_case = []
    unique_normal_case = []
    multiple_normal_case = []
    for t, sdf in rdf.groupby('검사결과결론내용'):
        if sdf.shape[0] == 1:
            unique_normal_case.append(sdf)
            continue

        unique_labels = sdf.label.unique()
        if len(unique_labels) == 1:
            multiple_normal_case.append(sdf)
            continue
        else: 
            novel_case.append(sdf)

    concat_df = []
    if len(unique_normal_case) > 0:
        concat_df += unique_normal_case
        # unique_df = pd.concat(unique_normal_case)
    if len(multiple_normal_case) > 0:
        concat_df += multiple_normal_case
        # multiple_df = pd.concat(multiple_normal_case)
    # if drop_duplicates:
    #     multiple_df = multiple_df.drop_duplicates('검사결과결론내용')

    training_label_df = pd.concat(concat_df).sort_values('index')
    if drop_duplicates:
        training_label_df = training_label_df.drop_duplicates('검사결과결론내용')
    training_label_df = training_label_df.reset_index(drop = True) if len(concat_df) > 0 else None


    if len(novel_case) > 0:
        novel_df = pd.concat(novel_case)
        novel_counter = defaultdict(int)
        for nc in novel_case:
            novel_counter['-'.join(sorted(nc.label.to_list()))] += nc.shape[0]

    return dict(
        label_df = training_label_df,
        unique_samples = pd.concat(unique_normal_case).shape[0],
        multiple_samples = pd.concat(multiple_normal_case).shape[0],
        novel_df = novel_df if len(novel_case) > 0 else None,
        novel_counter = novel_counter if len(novel_case) > 0 else None
    )




class Preprocessor:
    def __init__(
            self, 
            df, 
            negative_patterns,
            uncertain_patterns,
            abbrev_path=os.path.join('.','src','utils','abbrevations.json'),
        ):

        with open(abbrev_path, 'r', encoding='utf-8') as f:
            self.abbrev_dict = json.load(f)

        self.df = df
        self.regex = re.compile(r'(?<!\S)(' + "|".join(map(re.escape, list(self.abbrev_dict.keys()))) + r')(?!\S)')
        self.negative_regex = re.compile(self.word_patterns(negative_patterns), re.IGNORECASE)
        self.uncertain_regex = re.compile(self.word_patterns(uncertain_patterns), re.IGNORECASE)
        self.df = df.apply(self.expand_abbrev_value)


    def replace_abbrev(self, m: re.Match) -> str:
        key = m.group(1)
        return self.abbrev_dict.get(key, key)
    

    def expand_abbrev_value(self, x):
        # NaN 등 비문자 -> 그대로
        if not isinstance(x, str):
            return x
        # 매칭이 없으면 원본 그대로 반환됨
        x = self.regex.sub(self.replace_abbrev, x)
        x = re.sub(r'\n[ \t\n]+', '\n', x)
        x = re.sub(r'\n[\=\-\~#]+>?\s?', '\n', x)
        x = re.sub(r'\n[\=\-\~#]+>', '', x)
        # x = re.sub(r'', )
        return x
    

    @staticmethod
    def word_patterns(patterns):
        patterns = [f'(^| ?){i}' for i in patterns]
        return '|'.join(patterns)


    @staticmethod
    def extract_parts(s:str, target_text: str) -> list[str]:
        parts = filter(None, re.split(r'(^\-\d+\.\s?|\n\d+\.\s?|\n\d+\))', s))
        target_parts = list(filter(lambda x: target_text in x.lower(), parts))
        return target_parts
    

    @staticmethod
    def count_patterns(text: str, regex) -> int:
        if not isinstance(text, str):
            return 0
        return len(regex.findall(text))


    def count_list(self, l: str) -> [int, int]: # type: ignore
        counter = [0, 0, 0]
        for i in l:
            counter[0] += 1
            counter[1] += self.count_patterns(i, self.negative_regex)
            counter[2] += self.count_patterns(i, self.uncertain_regex)
        return counter


    def target_filtering(self, target_text):
        target_text_df = self.df.loc[self.df.apply(lambda x: target_text in x.lower())]

        ftarget_text_df = target_text_df.apply(lambda x: self.extract_parts(x, target_text))
        target_counts = np.stack(ftarget_text_df.apply(self.count_list).values)

        unc_idx = target_counts[:,2] > 0
        pos_idx = (target_counts[:, 0] > target_counts[:, 1])
        pos_idx = ((pos_idx ^ unc_idx) & pos_idx)
        neg_idx = ~(pos_idx | unc_idx)

        return dict(
            positive = ftarget_text_df.loc[pos_idx].str.join('\n'),
            negative = ftarget_text_df.loc[neg_idx].str.join('\n'),
            uncertain = ftarget_text_df.loc[unc_idx].str.join('\n'),
        )

