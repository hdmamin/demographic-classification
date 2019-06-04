from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize
import os
import pandas as pd
from pathlib import Path


def get_paths(dirname):
    """Return list of paths a given directory.

    Parameters
    ----------
    dirname: str

    Returns
    -------
    List of path objects.
    """
    return list(Path(dirname).iterdir())


def get_text(path):
    """Read in xml file, extract relevant info, and return a dataframe.

    Parameters
    ----------
    path: pathlib.Path
        Specify which file to read in. Includes the name of the directory
        as well as the file.

    Returns
    -------
    pd.DataFrame containing a row for each sentence. There are also label
    columns with age and sex.
    """
    sex, age = path.parts[-1].split('.')[1:3]
    with open(path, 'r', encoding='latin1') as f:
        soup = BeautifulSoup(f, 'xml')
    posts = [t.text.replace('\n', ' ').replace('\t', ' ').strip()
             for t in soup.find_all('post')]
    sentences = [s for s in sent_tokenize(' '.join(posts))]
    return pd.DataFrame(dict(text=sentences, sex=sex, age=int(age)))


def main(output_dir='data', output_file='sentences.csv'):
    """Generate dataframe of sentences and labels."""
    dfs = [get_text(path) for path in get_paths('blogs')]
    df = pd.concat(dfs, axis=0).reset_index(drop=True)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    df.to_csv(os.path.join(output_dir, output_file), index=False)


if __name__ == '__main__':
    main()
