import os
import requests

for root, dirs, files in os.walk('.'):
    print('root', root)
    for name in files:
        if 'html' in name and '\n' not in name:
            print('name', name)
            with open(os.path.join(root, name)) as f:
                file_str = f.readlines()[0]
            if not file_str.startswith('<html><head><meta http-equiv="refresh" content="0; '): 
                if requests.get('https://lets-unify.ai/docs/memory/{}'.format(os.path.join(root, name)), timeout=5):
                    with open(os.path.join(root, name), 'w') as f:
                        f.write('<html><head><meta http-equiv="refresh" content="0; url=https://lets-unify.ai/docs/memory/{}"></head></html>'.format(os.path.join(root, name)))
                else:
                    with open(os.path.join(root, name), 'w') as f:
                        f.write('<html><head><meta http-equiv="refresh" content="0; url=https://lets-unify.ai/docs/memory/"></head></html>'.format(os.path.join(root, name)))
        elif name != 'walk.py':
            os.remove(os.path.join(root, name))