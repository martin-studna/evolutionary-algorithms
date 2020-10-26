import random
head = random.randint(0, 1)
individual = [head]
for _ in range(10 - 1):
    individual.append(int(not individual[-1])) 
print(individual)

for i in range(10):
  elem = i

print(elem)