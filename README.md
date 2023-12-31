# Reinforcement Learning
This repository contains my implementation of different reinforcement learning algorithms, such as
 - Q-Learning
 - Double Q-Learning
 - Deep Q-Learning (DQN)
 - Double DQN

Future algorithms to be included:
 - Policy Gradient (PG) (REINFORCE)
 - Actor-Critic PG
 - Vanilla PG
 - Deep Deterministic PD (DDPG)
 - Twin-Delayed DDPG (TD3)
 - Option-Critic

Comments: -
 - While all the codes were indeed written and contain personal touches by myself, they are influenced by the works of [@philtabor](https://github.com/philtabor) and [@lweitkamp](https://github.com/lweitkamp)
 - All codes are written in PyTorch language
 - While all the codes learn to solve the control problem presented by the chosen environment, in no way are they the ideal solution. More fine tuning of the hyperparameters is needed. Nonetheless, the solutions offered are acceptable
 - Please read "RL Algorithms_v2.pdf" for more information

Q-Learning: -
 - Estimate $Q_{\pi}(s, a)$ via function approximation

   <img src="https://github.com/SaifAlWahaibi/ReinforcementLearning/assets/106843163/b69961a4-64ae-4f08-9533-ac9380f0c3bf" width="400" height="300">

 - Cost Function:
     - $J(\theta)=E_{\pi}[(\delta_{TD} - \hat{Q_{\theta}}(s, a))^{2}]$
     - $\delta_{TD} = r + \gamma \max_{a^{'}} \hat{Q}_{\theta}(s^{'}, a^{'})$

 - Pseudocode:
<br>Initialize $Q_{\theta}(s, a)$ with random weight
<br>**for** $episode = 1, 2, 3, ..., N$ **do**
<br>&nbsp; &nbsp; Initialize environment $s_{0}$
<br>&nbsp; &nbsp; **for** $t = 0, 1, 2, ..., T$ **do**
<br>&nbsp; &nbsp; &nbsp; &nbsp; Select action $a_{t}$ randomly with probability $\epsilon$, otherwise
<br>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; $a_{t} = \arg\max_{a_{t}} \hat{Q_{\theta}}(s_{t}, a)$
<br>&nbsp; &nbsp; &nbsp; &nbsp; Execute action $a_{t}$ in environment and observe $r_{t + 1}$, $s_{t + 1}$, terminal and truncate flags
<br>&nbsp; &nbsp; &nbsp; &nbsp; Set TD target $\delta_{TD} = r_{t + 1}$ if terminal or truncate flag is true, otherwise
<br>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; $\delta_{TD} = r_{t + 1} + \gamma \max_{a_{t + 1}} \hat{Q_{\theta}}(s_{t + 1}, a)$
<br>&nbsp; &nbsp; &nbsp; &nbsp; Perform a gradient descent step on
<br>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; $J(\theta)=E_{\pi}[(\delta_{TD} - \hat{Q_{\theta}}(s_{t}, a_{t}))^{2}]$
<br>&nbsp; &nbsp; &nbsp; &nbsp; Set $s_{t + 1}$ as current state
<br>&nbsp; &nbsp; **end for**
<br>**end for**

 - Example:
     - My solution for the "CartPole-v1" Gym environment:
       
       <img src="https://github.com/SaifAlWahaibi/ReinforcementLearning/assets/106843163/2b113d07-9ccd-47f0-9762-d1ed03df73d6" width="400" height="300">

Double Q-Learning: -
 - Estimate $Q_{\pi}(s, a)$ via function approximation
 - Use two Q-networks to handle maximization bias

   <img src="https://github.com/SaifAlWahaibi/ReinforcementLearning/assets/106843163/6184de42-a6b9-48d7-8579-82ec67121aea" width="500" height="300">

 - Cost Function:
     - $J(\theta_{i})=E_{\pi}[(\delta_{TD} - \hat{Q_{\theta_{i}}}(s, a))^{2}]$
     - $\delta_{TD} = r + \gamma \hat{Q_{\theta_{j}}}(s^{'}, \arg\max_{a^{'}} \hat{Q_{\theta_{i}}}(s^{'}, a^{'}))$

 - Pseudocode:
<br>Initialize $Q_{\theta_{x}}(s, a)$ with random weight
<br>**for** $episode = 1, 2, 3, ..., N$ **do**
<br>&nbsp; &nbsp; Initialize environment $s_{0}$
<br>&nbsp; &nbsp; **for** $t = 0, 1, 2, ..., T$ **do**
<br>&nbsp; &nbsp; &nbsp; &nbsp; Select action $a_{t}$ randomly with probability $\epsilon$, otherwise
<br>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; $a_{t} = \arg\max_{a_{t}} ((\hat{Q_{\theta_{1}}}(s_{t}, a_{t}) + \hat{Q_{\theta_{2}}}(s_{t}, a_{t})) / 2)$
<br>&nbsp; &nbsp; &nbsp; &nbsp; Execute action $a_{t}$ in environment and observe $r_{t + 1}$, $s_{t + 1}$, terminal and truncate flags
<br>&nbsp; &nbsp; &nbsp; &nbsp; Choose at random either to update 1 or 2
<br>&nbsp; &nbsp; &nbsp; &nbsp; **if** $i = 1$ or $2$ **then**
<br>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Set TD target $\delta_{TD} = r_{t + 1}$ if terminal or truncate flag is true, otherwise
<br>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; $a^{+}=\arg\max_{a_{t+1}} \hat{Q_{\theta_{i}}}(s_{t+1}, a_{t+1})$
<br>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; $\delta_{TD} = r_{t + 1} + \gamma \hat{Q_{\theta_{3-i}}}(s_{t + 1}, a^{+})$
<br>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Perform a gradient descent step on
<br>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; $J(\theta_{i})=E_{\pi}[(\delta_{TD} - \hat{Q_{\theta_{i}}}(s_{t}, a_{t}))^{2}]$
<br>&nbsp; &nbsp; &nbsp; &nbsp; **end if**
<br>&nbsp; &nbsp; &nbsp; &nbsp; Set $s_{t + 1}$ as current state
<br>&nbsp; &nbsp; **end for**
<br>**end for**

 - Example:
     - My solution for the "CartPole-v1" Gym environment:

       <img src="https://github.com/SaifAlWahaibi/ReinforcementLearning/assets/106843163/5a7496af-f4a9-4adc-94ce-1d5acb35f0aa" width="400" height="300">
