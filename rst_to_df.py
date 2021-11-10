from bs4 import BeautifulSoup
import pandas as pd
from deeppavlov import build_model
from typing import Dict, Iterable
import os


def _parse_syntax(sents: Iterable[str]):
    """Parse syntax with deeppavlov model"""
    model = build_model("ru_syntagrus_joint_parsing")
    model['main'].to_output_string = False
    model['main'].output_format = 'json'
    output = model(sents)
    model.destroy()
    return output


def _preprocess_segment(segment: Dict):
    if segment['segment'].startswith(('О:', 'М:')):
        segment['segment'] = segment['segment'][2:]
        segment['is_prompt'] = 1
    else:
        segment['is_prompt'] = 0
    return segment


def _rst_to_df(rst_soup: BeautifulSoup):
    """Convert rst-annotated text to df"""
    segments = []
    for tag in rst_soup.find_all('segment'):
        segment = {'segment': tag.string}
        segment.update(tag.attrs)
        segment = _preprocess_segment(segment)
        segments.append(segment)

    rst_df = pd.DataFrame(segments)
    return rst_df


def _rst_df_to_tokens_df(rst_df):
    parsed_segments = []
    # for segment in rst_df['segment']:
    #     print(segment)
    #     parsed_segments.append(_parse_syntax([segment])[0])
    # parsed_segments = [_parse_syntax([s])[0] for s in rst_df['segment']]
    parsed_segments = _parse_syntax(rst_df['segment'])
    tokens = []
    for parsed, segment in zip(parsed_segments, rst_df.iterrows()):
        for token in parsed:
            token['word_id'] = token.pop('id')
            token['segment_id'] = segment[1]['id']
            token['parent'] = segment[1]['parent']
            token['relname'] = segment[1]['relname']
            token['is_prompt'] = segment[1]['is_prompt']
            tokens.append(token)
    tokens_df = pd.DataFrame(tokens)
    return tokens_df


def convert_rst_to_tokens_df(load_path, save_path):
    """Convert rst to a df of tokens with syntax info and save it"""
    with open(load_path) as f:
        soup = BeautifulSoup(f, 'xml')

    rst_df = _rst_to_df(soup)
    tokens_df = _rst_df_to_tokens_df(rst_df)
    tokens_df.to_csv(save_path, index=False)


def unite_dfs(folder_path: str):
    dfs = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path)
        df['file'] = file_path
        dfs.append(df)
    united_df = pd.concat(dfs)
    return united_df


def main():
    rst_texts_path = 'rst_texts'
    for filename in os.listdir(rst_texts_path):
        file_path = os.path.join(rst_texts_path, filename)
        if os.path.isfile(file_path):
            convert_rst_to_tokens_df(file_path, os.path.join('rst_df', os.path.splitext(filename)[0] + '.csv'))
    unite_dfs('rst_df').to_csv('bilingual.csv', index=False)


if __name__ == '__main__':
    main()
