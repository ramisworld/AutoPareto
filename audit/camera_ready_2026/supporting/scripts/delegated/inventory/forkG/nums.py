import sys, re
keys = ['1.99', '1.98', '2.0×', '41.7', '35.1', '50%', '1.141837', '1.099803', '0.042034', '383.02', '192.93', '201,326,592', '1,536', '131,072',
        '4-attempt', 'four completed', 'four attempts', '22 attempts', 'twenty-two', '21 conﬁrmation', '21 confirmation', 'three 25-attempt', '25 attempts', '75 ',
        'hypervolume', '428e22f1', 'b3a961ee', '8547b278', '62882a20', '55a7fd8b', '36e3ad91', '6d0d8d1d', 'AP292', 'AP299', 'AP294',
        'seeds 3', 'gpt-5.6-sol', 'gpt-5.6-terra', 'Codex CLI 0.153.0', '29.36M', '50.33M', '162.88', '250.88', '1.992', '3.984', '0.9697', '1.0939', '1.0942']
for p in sys.argv[1:]:
    t = open(p).read()
    out = []
    for k in keys:
        c = t.count(k)
        if c: out.append(f'{k}:{c}')
    print(p, '|', ' '.join(out))
