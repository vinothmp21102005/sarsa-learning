## SARSA LEARNING ALGORITHM
## AIM
To develop a Python program to find the optimal policy for the given RL environment using SARSA-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
The Frozen Lake problem is a reinforcement learning task in which an agent must navigate a 4x4 grid (16 states) to reach the goal state. The environment is slippery, meaning the agent has a chance of moving in the opposite direction of the intended action, making it harder to navigate. The agent's goal is to learn an optimal policy to reach the goal while avoiding hazards (like holes in the ice) through trial and error, adjusting its strategy over time.

## SARSA LEARNING FUNCTION
### Name: VINOTH M P
### Register Number: 212223240182

```
def sarsa(env,
          gamma=1.0,
          init_alpha=0.5,
          min_alpha=0.01,
          alpha_decay_ratio=0.5,
          init_epsilon=1.0,
          min_epsilon=0.1,
          epsilon_decay_ratio=0.9,
          n_episodes=3000):
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    # Write your code here
    select_action = lambda state, Q, epsilon:np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))
    alphas = decay_schedule(init_alpha, min_alpha,
                            alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon,
                              epsilon_decay_ratio, n_episodes)
    for e in tqdm(range(n_episodes), leave=False):
      state, done = env.reset(), False
      action = select_action(state, Q, epsilons[e])
      while not done:
        next_state, reward, done, _= env.step(action)
        next_action = select_action(next_state, Q, epsilons[e])
        td_target = reward + gamma * Q[next_state][next_action] * (not done)
        td_error = td_target - Q[state][action]
        Q[state][action] = Q[state][action] + alphas[e] * td_error
        state, action = next_state, next_action
      Q_track[e] = Q
      pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]
    return Q, V, pi, Q_track, pi_track

Q_sarsas, V_sarsas, Q_track_sarsas = [], [], []
for seed in tqdm(SEEDS, desc='All seeds', leave=True):
    random.seed(seed); np.random.seed(seed) ; env.seed(seed)
    Q_sarsa, V_sarsa, pi_sarsa, Q_track_sarsa, pi_track_sarsa = sarsa(env, gamma=gamma, n_episodes=n_episodes)
    Q_sarsas.append(Q_sarsa) ; V_sarsas.append(V_sarsa) ; Q_track_sarsas.append(Q_track_sarsa)
Q_sarsa = np.mean(Q_sarsas, axis=0)
V_sarsa = np.mean(V_sarsas, axis=0)
Q_track_sarsa = np.mean(Q_track_sarsas, axis=0)
del Q_sarsas ; del V_sarsas ; del Q_track_sarsas
```

## OUTPUT:
<img width="512" height="727" alt="image" src="https://github.com/user-attachments/assets/0c1b8181-a7ee-47e9-ac45-e1908760fbdb" />

## FVMC:
<img width="595" height="605" alt="image" src="https://github.com/user-attachments/assets/86e3d4b7-0d66-46ad-86de-066c622eb9e3" />
<img width="553" height="75" alt="image" src="https://github.com/user-attachments/assets/a398cdb0-2961-49f8-9078-91a9020c7dbd" />

## SARSA:
<img width="757" height="603" alt="image" src="https://github.com/user-attachments/assets/f6debd2a-b189-401d-9732-20bc8370e055" />

<img width="618" height="72" alt="image" src="https://github.com/user-attachments/assets/0608ce1f-dd46-4be0-adae-d6fe6ad6a2f3" />


## FVMC:
<img width="1477" height="482" alt="image" src="https://github.com/user-attachments/assets/e12a1de7-0e27-4199-a07e-25f439387bb6" />

## SARSA:
<img width="1422" height="482" alt="image" src="https://github.com/user-attachments/assets/a5195de8-7f3e-4837-a097-3b8fd1a4ee67" />

## RESULT:

Thus, SARSA learning successfully trained an agent for optimal policy.
