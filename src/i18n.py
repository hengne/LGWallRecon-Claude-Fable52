# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Language resources: every string shown in a figure or on the report page lives in web/i18n/<lang>.json."""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))


def load(lang):
    """Return the string table of one language ('zh' or 'en')."""
    with open(os.path.join(HERE, '..', 'web', 'i18n', f'{lang}.json'), encoding='utf-8') as f:
        return json.load(f)
