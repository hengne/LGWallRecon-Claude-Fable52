#!/usr/bin/env python3
# LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan
# Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>
"""Put the two-line project/author header at the top of every source file that lacks it.
usage: add_headers.py <file> [<file> ...]"""
import sys
L1 = 'LGWallRecon -- light-guide inner-wall reconstruction from an entrance-face scan'
L2 = 'Author: Hengne Li (李衡讷) <Hengne.Li@cern.ch>, <Hengne.Li@m.scnu.edu.cn>'
for path in sys.argv[1:]:
    s = open(path, encoding='utf-8').read()
    if 'Author: Hengne Li' in s[:600]:
        continue
    if path.endswith(('.py', '.sh', '.txt')):
        head = f'# {L1}\n# {L2}\n'
    elif path.endswith(('.cc', '.js')):
        head = f'// {L1}\n// {L2}\n'
    elif path.endswith('.html'):
        head = f'<!-- {L1}\n     {L2} -->\n'
    else:
        continue
    if s.startswith('#!'):                       # keep the shebang first
        first, rest = s.split('\n', 1); s = first + '\n' + head + rest
    elif path.endswith('.html') and s.startswith('<title>'):
        first, rest = s.split('\n', 1); s = first + '\n' + head + rest      # the page title must stay on the first line
    else:
        s = head + s
    open(path, 'w', encoding='utf-8').write(s)
    print('header ->', path)
