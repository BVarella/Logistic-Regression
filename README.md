# Logistic-Regression
Basic logistic regression implementation using Keras<br /><br />

Keras Sequential model <br />
Uma interface simples de criação de modelos com um conjunto de entradas e saídas e uma sucessão de camadas.<br />
Assim se torna possível adicionar mais camadas ao modelo no futuro.<br /><br />

optimizer = 'sgd' -> Stochastic gradient descent optimizer<br />
Foi utilizado a otimização estocástica por esse oscilador ser muito utilizado para tentar prever os movimentos nos preços de ativos.<br /><br />

loss = 'binary_crossentropy'<br />
Como a saída é binária (tenta-se prever se irá ocorrer alta ou não nos preços das ações), as variações são muito pequenas. Dessa forma, utilizar a fórmula de entropia cruzada como função de custo a ser minimizada se torna mais eficiente do que utilizar o erro quadrado, por exemplo.
