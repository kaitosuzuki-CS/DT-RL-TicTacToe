l = {(1,2): 1, 'b': 2, 'c': 3}

import json as json

with open('q_table.json', 'w') as f:
    json.dump(l, f)