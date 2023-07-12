# ReinforcementLearning
Different reinforcement learning algorithm

Q-Learning Introduction: -

![QL_S1](https://github.com/SaifAlWahaibi/ReinforcementLearning/assets/106843163/1848529f-d46c-43fd-a3c3-0a36a040965e)

![QL_S2](https://github.com/SaifAlWahaibi/ReinforcementLearning/assets/106843163/31ae0a25-cd0b-45c0-8646-0bb20dabee0e)

![QL_S3](https://github.com/SaifAlWahaibi/ReinforcementLearning/assets/106843163/1c02fbe0-a5bd-40d4-ad2c-d91107ff1b14)

	Initialize 𝑄 ̂_𝜽 (𝑠,𝑎) with random weights
	for episode = 1, 2, 3, …, E do
		Initialize environment 𝑠_𝟎
		for t = 0, 1, 2, …, T do
			 Select action 𝑎_𝒕 randomly with probability 𝜖, otherwise
 				𝑎_𝒕=argmax┬(𝑎_𝒕 )⁡〖𝑄 ̂_𝜽 (𝑠_𝒕,𝑎_𝒕 )〗
			Execute action 𝑎_𝒕 in environment and observe 𝑟_(𝒕+𝟏), 𝑠_(𝒕+𝟏), 				and terminal or truncate flags
			Set TD target 𝛿_𝑇𝐷=𝑟_(𝒕+𝟏) if terminal or truncate, otherwise 				𝛿_𝑇𝐷=𝑟_(𝒕+𝟏)+𝛾  max┬(𝑎_(𝒕+𝟏) )⁡〖𝑄 ̂_𝜽 (𝑠_(𝒕+𝟏),𝑎_(𝒕+𝟏) )〗
		Perform a gradient descent step on 
			𝒥(𝜽)=𝔼_𝜋 [(𝛿_𝑇𝐷−𝑄 ̂_𝜽 (𝑠_𝒕,𝑎_𝒕 ))^2 ]
		Set 𝑠_(𝒕+𝟏) as current state
	end for
end for
