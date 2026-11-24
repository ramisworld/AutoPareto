import sys, re
t = open(sys.argv[1]).read()
lines = [l for l in t.splitlines() if not re.fullmatch(r'\s*\d{1,3}\s*', l) and not l.startswith('<<<PAGE')]
s = ' '.join(' '.join(lines).split())
s = s.replace('- ', '-')  # crude hyphen join
print(s)
