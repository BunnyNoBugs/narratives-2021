from bs4 import BeautifulSoup
import pandas as pd
from deeppavlov import build_model
from typing import Dict, Iterable
import os
import networkx as nx
from scipy.sparse import csr_matrix


def _parse_syntax(sents: Iterable[str]):
    """Parse syntax with deeppavlov model"""
    model = build_model("ru_syntagrus_joint_parsing")
    model['main'].to_output_string = False
    model['main'].output_format = 'json'
    output = model(sents)
    model.destroy()
    return output


def _preprocess_segment(segment: Dict):
    segment['segment'].replace('"', '')
    if segment['segment'].startswith(('О:', 'М:', 'М\t', 'Э\t', 'В\t')):
        segment['segment'] = segment['segment'][2:]
        segment['is_prompt'] = 1
    else:
        if segment['segment'].startswith(('Р:', 'Р\t', 'О\t')):
            segment['segment'] = segment['segment'][2:]
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


def _txt_to_edu_df(text: str):
    segments = [{'segment': x} for x in text.split('\n') if x and not x.isspace()]
    segments = [_preprocess_segment(s) for s in segments]
    edu_df = pd.DataFrame(segments)
    edu_df['segment_id'] = pd.Series(range(1, len(edu_df) + 1))
    return edu_df


def _edu_df_to_tokens_df(edu_df):
    parsed_segments = _parse_syntax(edu_df['segment'])
    tokens = []
    for parsed, segment in zip(parsed_segments, edu_df.iterrows()):
        for token in parsed:
            token['segment_id'] = segment[1]['segment_id']
            token['is_prompt'] = segment[1]['is_prompt']
            tokens.append(token)
    tokens_df = pd.DataFrame(tokens)
    return tokens_df


def _rst_to_graph(soup):
    results = soup.findAll(lambda tag: 'id' in tag.attrs)
    ids = [tag.attrs['id'] for tag in results]
    graph = []
    for tag in results:
        node = [0] * len(results)
        if 'parent' in tag.attrs:
            node[ids.index(tag.attrs['parent'])] = 1
        graph.append(node)
    return graph


def _analyze_graph(graph):
    graph = csr_matrix(graph)
    graph = nx.DiGraph(graph)
    return [len(x) for x in nx.weakly_connected_components(graph)]


def convert_rst_to_tokens_df(load_path, save_path):
    """Convert rst to a df of tokens with syntax info and save it"""
    with open(load_path) as f:
        soup = BeautifulSoup(f, 'xml')

    rst_df = _rst_to_df(soup)
    tokens_df = _rst_df_to_tokens_df(rst_df)
    tokens_df.to_csv(save_path, index=False)


def convert_edu_to_tokens_df(load_path, save_path):
    with open(load_path) as f:
        text = f.read()
    edu_df = _txt_to_edu_df(text)
    tokens_df = _edu_df_to_tokens_df(edu_df)
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
    # RST to tokens
    # for group in ['adult', 'bilingual', 'monolingual']:
    #     group_folder_path = os.path.join('RST markup', group)
    #     for filename in os.listdir(group_folder_path):
    #         file_path = os.path.join(group_folder_path, filename)
    #         if os.path.isfile(file_path):
    #             convert_rst_to_tokens_df(file_path,
    #                                      os.path.join('rst_df', group, os.path.splitext(filename)[0] + '.csv'))

    # EDU to tokens
    # for group in ['adult', 'bilingual', 'monolingual']:
    #     group_folder_path = os.path.join('EDU markup', group)
    #     for filename in os.listdir(group_folder_path):
    #         file_path = os.path.join(group_folder_path, filename)
    #         convert_edu_to_tokens_df(file_path,
    #                                  os.path.join('edu_df', group, os.path.splitext(filename)[0] + '.csv'))

    # for group in ['adult', 'bilingual', 'monolingual']:
    #     group_folder_path = os.path.join('rst_df', group)
    #     unite_dfs(group_folder_path).to_csv(f'{group}_rst.csv', index=False)

    # testing
    # segment = {'segment': 'О: угу, молодец. Э, Сейчас я тебе покажу картинки из мультика, который ты только что посмотрела. Тебе надо будет ответить на вопросы к этим картинкам. Тебе понятно, что надо делать?'}
    # print(_preprocess_segment(segment))

    chain_sizes = []
    for group in ['adult', 'bilingual', 'monolingual']:
        group_folder_path = os.path.join('RST markup', group)
        for filename in os.listdir(group_folder_path):
            file_path = os.path.join(group_folder_path, filename)
            if os.path.isfile(file_path):
                with open(file_path) as f:
                    soup = BeautifulSoup(f, 'xml')
                graph = _rst_to_graph(soup)
                chain_sizes.extend([{'group': group, 'chain_size': i} for i in _analyze_graph(graph)])
    chain_sizes_df = pd.DataFrame(chain_sizes)
    chain_sizes_df.to_csv('chain_sizes.csv', index=False)


if __name__ == '__main__':
    main()
