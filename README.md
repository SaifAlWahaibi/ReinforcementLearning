# ReinforcementLearning
This repository contains my implementation of different reinforcement learning algorithms, such as
 - Q-Learning

Future algorithms to be included:
 - Double Q-Learning
 - Deep Q-Learning (DQN)
 - Double DQN
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
 - Please read "RL Algorithms_v1.pdf" for more information

Q-Learning: -
 - Estimate $Q_{\pi}(s, a)$ via function approximation
![QL](https://github.com/SaifAlWahaibi/ReinforcementLearning/assets/106843163/573616c2-038b-4845-8654-36cf31e9ee19)

 - Cost Function:
  -- $J(\theta)=E_{\pi}[(\delta_{TD} - \hat{Q}_{\theta}(s, a))^{2}]$

