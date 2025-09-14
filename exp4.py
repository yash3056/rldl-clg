import numpy as np

def get_next_state(s, a):
    if s in [8, 2, 4]: return s, [10, -10, 0][s//4]
    r, c = divmod(s, 3)
    dr, dc = [(-1,0), (0,1), (1,0), (0,-1)][a]
    nr, nc = max(0, min(2, r+dr)), max(0, min(2, c+dc))
    ns = nr*3 + nc
    return (s, 0) if ns == 4 else (ns, -1 if ns != s else 0)

V, policy = np.zeros(9), np.ones((9, 4)) / 4
skip = [0, 2, 4, 8]

while True:
    # Policy Evaluation
    while True:
        delta = 0
        for s in range(9):
            if s in skip: continue
            v = sum(policy[s,a] * (r + 0.9*V[ns]) for a in range(4) for ns,r in [get_next_state(s,a)])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < 1e-6: break
    
    # Policy Improvement
    stable = True
    for s in range(9):
        if s in skip: continue
        old = np.argmax(policy[s])
        vals = [r + 0.9*V[ns] for a in range(4) for ns,r in [get_next_state(s,a)]]
        best = np.argmax(vals)
        policy[s] = np.eye(4)[best]
        if old != best: stable = False
    
    if stable: break

print("Values:\n", V.reshape(3,3))
actions = ['up', 'right', 'down', 'left']
labels = {0:'Start', 2:'Pit', 4:'Wall', 8:'Goal'}
for s in range(9):
    print(f"{s}: {labels.get(s, actions[np.argmax(policy[s])])}")