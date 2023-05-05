import random
import numpy as np

def single(A, B, a):
    child1 = A[:a] + B[a:]
    child2 = B[:a] + A[a:]
    return child1, child2
def multi(A, B, a):
    for i in a:
        A, B = single(A, B, i)
    return A, B
def uniform(A, B, a):
    for i in range(len(a)):
        if a[i] < 0.5:
            temp = A[i]
            A[i] = B[i]
            B[i] = temp
    return A, B
def crossover(parent1,parent2,cross):
    #one point
    if cross == 0:
        a = random.randint(1, len(parent1)-1)
        return single(parent1, parent2, a)
    #multi point
    if cross == 1:
        a = random.randint(0, len(parent1)-1)
        num = []
        for i in range(a):
            num.append(random.randint(0, len(parent1)-1))
        num.sort()
        num = [*set(num)]
        return multi(parent1, parent2, num)
    #uniform
    if cross == 2:
        num = np.random.rand(len(parent1))
        return uniform(parent1, parent2, num)
def mutate(parent, mutate):
    #Random reset
    if mutate == 0:
        a = random.randint(0, len(parent)-1)
        parent[a] = random.randint(0,9)
        return parent
    #Swap
    if mutate == 1:
        a = 0
        b = 0
        while True:
            a = random.randint(0, len(parent) - 1)
            b = random.randint(0, len(parent) - 1)
            if a != b:
                break
        temp = parent[a]
        parent[a] = parent[b]
        parent[b] = temp
        return parent
    #Mutate Scramble and Inverse by Brandon Da Silva https://brandinho.github.io/genetic-algorithm/
    #Scramble
    if mutate == 2:
        child = np.array(parent)
        lower_limit = np.random.randint(0, child.shape[0]-1)
        upper_limit = np.random.randint(lower_limit+1, child.shape[0])
        scrambled_order = np.random.choice(np.arange(lower_limit, upper_limit+1), upper_limit + 1 - lower_limit, replace = False)
        child[lower_limit:upper_limit+1] = child[scrambled_order]
        return child.tolist()
    #Inverse
    if mutate == 3:
        child = np.array(parent)
        lower_limit = np.random.randint(0, child.shape[0]-1)
        upper_limit = np.random.randint(lower_limit+1, child.shape[0])
        child[lower_limit:upper_limit+1] = child[lower_limit:upper_limit+1][::-1]
        return child.tolist()

class Individual(object):
    def __init__(self, add, goal):
        self.goal = goal
        if add == "Create":
            self.chromosome = self.create()
        else:
            self.chromosome = add
        self.fitness = self.calc_fitness()
    def create(self):
        chromosome = []
        for i in range(len(self.goal)):
            chromosome.append(random.randint(0,9))
        return chromosome
    def calc_fitness(self):
        fitness = 0
        for ip, gl in zip(self.chromosome, self.goal):
            if ip == gl: fitness+= 3
            if ip in self.goal: fitness+= 1
        return fitness

def main(population_size, goal, selection, cross, mutation, mutation_rate):
    population = []
    if selection != 0 and selection != 1:
        print("Selection must be 0 or 1, provided:", selection)
        return
    if cross != 0 and cross != 1 and cross != 2:
        print("Crossover must be 0, 1, or 2, provided:", cross)
        return
    for i in range(len(mutation)):
        if mutation[i] != 0 and mutation[i] != 1 and mutation[i] != 2 and mutation[i] != 3:
            print("Mutation must be 0, 1, 2, or 3 provided:", mutation)
            return
    if mutation_rate < 0 or isinstance(mutation_rate, int) == False:
        print("Mutation rate must but a positive number and an integer, provided:", mutation_rate)
        return
    for e in range(population_size):
        population.append(Individual("Create", goal))
    best_fit = len(goal) * 3 + len(goal)
    #Tournament
    if selection == 0:
        generation = 0
        limit = 0
        previous_individual = []
        new_pop = []
        k = 5
        while True:
            parents = []
            children = []
            sorted_population = sorted(population, key=lambda x:x.fitness,reverse=True)
            if all(x == y for x, y in  zip(previous_individual,sorted_population[0].chromosome)):
                limit += 1
            if previous_individual != sorted_population[0].chromosome:
                limit = 0
            previous_individual = sorted_population[0].chromosome
            print("Best individual in generation:", generation, sorted_population[0].chromosome)
            if limit > 500:
                print("No Change in last 500 generations, best individual found in generation:",  generation, sorted_population[0].chromosome)
            if sorted_population[0].fitness == best_fit:
                print("found in generation", generation)
                break
            while len(parents) != 50:
                tmp = []
                tmp2 = []
                choose = random.sample(range(0,len(population)), k)
                for i in choose:
                    tmp.append(population[i])
                tmp = sorted(tmp, key=lambda x:x.fitness)

                choose2 = random.sample(range(0,len(population)), k)
                for i in choose2:
                    tmp2.append(population[i])
                tmp2 = sorted(tmp2, key=lambda x:x.fitness)
                if tmp[0] in parents and tmp2[0] in parents:
                    continue
                else:
                    parents.append(tmp[0])
                    parents.append(tmp2[0])
            for i in range(len(parents)):
                if i%2 == 1:
                    continue
                child1, child2 = crossover(parents[i].chromosome, parents[i+1].chromosome, cross)
                mutate_rate = random.randint(1,mutation_rate)      
                if mutate_rate == 1:
                    child1 = mutate(child1, random.choice(mutation))
                mutate_rate = random.randint(1,mutation_rate)      
                if mutate_rate == 1:
                    child2 = mutate(child2, random.choice(mutation))
                children.append(Individual(child1,goal))
                children.append(Individual(child2,goal))
            for i in range(len(children)):
                population.append(children[i])
            sort = sorted(population, key=lambda x:x.fitness,reverse=True)
            for i in range(50):
                sort.pop()
            population = sort
            generation += 1
            
    #Roulette
    if selection == 1:
        generation = 0
        limit = 0
        previous_individual = []
        k = 20
        while True:
            new_pop = []
            children = []
            sorted_population = sorted(population, key=lambda x:x.fitness,reverse=True)
            if all(x == y for x, y in  zip(previous_individual,sorted_population[0].chromosome)):
                limit += 1
            if previous_individual != sorted_population[0].chromosome:
                limit = 0
            previous_individual = sorted_population[0].chromosome
            print("Best individual in generation:", generation, sorted_population[0].chromosome)
            if limit > 500:
                print("No Change in last 500 generations, best individual found in generation:",  generation, sorted_population[0].chromosome)
                break
            if sorted_population[0].fitness == best_fit:
                print("found in generation", generation)
                break
            while len(new_pop) != 50:
                population_fitness = sum([chromosome.fitness for chromosome in population])
                # Computes for each chromosome the probability 
                chromosome_probabilities = [chromosome.fitness/population_fitness for chromosome in population]
                
                # Selects one chromosome based on the computed probabilities
                parent1 = np.random.choice(population, p=chromosome_probabilities)
                parent2 = np.random.choice(population, p=chromosome_probabilities)
                if parent1 in new_pop and parent2 in new_pop:
                    continue
                else:
                    new_pop.append(parent1)
                    new_pop.append(parent2)
            
            for i in range(len(new_pop)):
                if i%2 == 1:
                    continue
                child1, child2 = crossover(new_pop[i].chromosome, new_pop[i+1].chromosome, cross)
                mutate_rate = random.randint(1,mutation_rate)
                if mutate_rate == 1:
                    child1 = mutate(child1, random.choice(mutation))
                mutate_rate = random.randint(1,mutation_rate)
                if mutate_rate == 1:
                    child2 = mutate(child2, random.choice(mutation))
                children.append(Individual(child1,goal))
                children.append(Individual(child2,goal))
            for i in range(len(children)):
                population.append(children[i])
            sort = sorted(population, key=lambda x:x.fitness,reverse=True)
            for i in range(50):
                sort.pop()
            population = sort
            generation += 1



#Amount of individuals in population
pop = 100

#Num Goal you want to achieve
goal = 123456
goal = [int(x) for x in str(goal)]   #Converts goal into list, no touchy >:(

#Selection 0 = Tournament
#Selection 1 = Roulette
selection = 1

#Crossover 0 = One Point
#Crossover 1 = Multi Point
#Crossover 2 = Uniform
cross = 1

#Provide a list of numbers to have multiple mutations, provide multiple of
#the same number to have better odds of that mutation
#Mutation 0 = Random Reset
#Mutation 1 = Swap
#Mutation 2 = Scramble
#Mutation 3 = Inverse
mutation = [0,0,0,0,1,1,2]

#Mutation Rate calculated by 1 / number provided
mutation_rate = 5

main(pop, goal, selection, cross, mutation, mutation_rate)











