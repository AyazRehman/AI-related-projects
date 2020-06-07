
# Set up library imports.
import random
from collections import Counter
from itertools import chain

# install bitstring 
import subprocess
import sys
def install(package):
	subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install("bitstring")

# import bitstring
from bitstring import *

########################################################
'''
	- This module assumes you first read the Jupyter notebook. 
	- You are free to add other members functions in class GeneticAlgorithm
	  as long as you do not modify the code already written. If you have justified
	  reasons for making modifications in code, come talk to us. 
	- Our implementation uses recursive solutions and some flavor of 
	  functional programming (maps/lambdas); You're not required to do so.
	  Just Write clean code. 
'''
########################################################

class GeneticAlgorithm(object):

	def __init__(self, POPULATION_SIZE, CHROMOSOME_LENGTH, verbose):
		self.wall_bit_string_raw = "01010101011001101101010111011001100101010101100101010101"
		self.wall_bit_string = ConstBitStream(bin = self.wall_bit_string_raw)
		self.population_size = POPULATION_SIZE
		self.chromosome_length = CHROMOSOME_LENGTH # this is the length of self.wall_bit_string
		self.terminate = False
		self.verbose = verbose # In verbose mode, fitness of each individual is shown. 

	def run_genetic_alg(self):
		'''  
		The pseudo you saw in slides of Genetic Algorithm is implemented here. 
		Here, You'll get a flavor of functional 
		programming in Python- Those who attempted ungraded optional tasks in tutorial
		have seen something similar there as well. 
		Those with experience in functional programming (Haskell etc)
		should have no trouble understanding the code below. Otherwise, take our word that
		this is more or less similar to the generic pseudocode in Jupyter Notebook.

		'''
		"You may not make any changes to this function."

		# Creation of Population
		solutions = self.generate_candidate_sols(self.population_size) # arg passed for recursive implementation.

		# Evaluation of individuals
		parents = self.evaluate_candidates(solutions)

		while(not self.terminate):
			# Make pairs
			pairs_of_parents = self.select_parents(parents)

			# Recombination of pairs.
			recombinded_parents = list(chain(*map(lambda pair: \
				self.recombine_pairs_of_parents(pair[0], pair[1]), \
					pairs_of_parents))) 

			# Mutation of each individual
			mutated_offspring = list(map(lambda offspring: \
				self.mutate_offspring(offspring), recombinded_parents))

			# Evaluation of individuals
			parents = self.evaluate_candidates(mutated_offspring) # new parents (offspring)
			if self.verbose and not self.terminate:
				self.print_fitness_of_each_indiviudal(parents)

######################################################################
###### These two functions print fitness of each individual ##########

# *** "Warning" ***: In this function, if an individual with 100% fitness is discovered, algorithm stops. 
# You should implement a stopping condition elsewhere. This codition, for example,
# won't stop your algorithm if mode is not verbose.
	def print_fitness_of_one_individual(self, _candidate_sol):
		_WallBitString = self.wall_bit_string
		_WallBitString.pos = 0
		_candidate_sol.pos = 0
		
		matching_bit_pairs = 0
		try:
			if not self.terminate:
				while (_WallBitString.read(2).bin == _candidate_sol.read(2).bin):
					matching_bit_pairs = matching_bit_pairs + 1
				print('Individual Fitness: ', round((matching_bit_pairs)/28*100, 2), '%')
		except: # When all bits matched. 
			pass
			return

	def print_fitness_of_each_indiviudal(self, parents):
		if parents:
			for _parent in parents:
				self.print_fitness_of_one_individual(_parent)

###### These two functions print fitness of each individual ##########
######################################################################

	def select_parents(self, parents):
		'''
		args: parents (list) => list of bitstrings (ConstbitStream)
		returns: pairs of parents (tuple) => consecutive pairs.
		'''

		# **** Start of Your Code **** #
		parent=[]
		# i=0
		print("\n\n\nlenght: ",len(parents))
		for i in range(0,len(parents),2):
			parent.append((parents[i],parents[i+1]))
		return parent
#         pass
		# **** End of Your Code **** #


	# A helper function that you may find useful for `generate_candidate_sols()`
	def random_num(self):
		random.seed()
		return random.randrange(2**14) ## for fitting in 14 bits.

	def generate_candidate_sols(self, n): 
		'''
		args: n (int) => Number of cadidates solutions to generate. 
		retruns: (list of n random 56 bit ConstBitStreams) 
				 In other words, a list of individuals: Population.

		Each cadidates solution is a 56 bit string (ConstBitStreams object). 

		One clean way is to first get four 14 bit random strings then concatenate
		them to get the desired 56 bit candidate. Repeat this for n candidates.
		'''
		pop=[]
		for i in range(n):
			random_string=''
			for j in range(4):
				num=format(self.random_num(),"014b")
				random_string=random_string+str(num)
			pop.append(ConstBitStream(bin=(random_string)))
		return pop
		
		
		# **** Start of Your Code **** #
#         pass
		# **** End of Your Code **** # 

	def recombine_pairs_of_parents(self, p1, p2):
		"""
		args: p1, and p2  (ConstBitStream)
		returns: p1, and p2 (ConstBitStream)

		split at .6-.9 of 56 bits (CHROMOSOME_LENGTH). i.e. between 31-50 bits
		"""

		random.seed()
		num= random.randrange(31,51)
		peice1left=p1[:num]
		peice1right=p1[num:]
		piece2left=p2[:num]
		piece2right=p2[num:]

		return (peice1left+piece2right,piece2left+peice1right)
		# **** Start of Your Code **** #
		# pass
		# **** End of Your Code **** #

	def mutate_offspring(self, p):
		''' 
			args: individual (ConstBitStream)
			returns: individual (ConstBitStream)
		'''
		
		p.pos=0
		mutation_string = BitStream(p)
		mutate_rate =0.005  
		for i in range(len(p)):
			chance = random.choices([0, 1], [(1 - mutate_rate), mutate_rate], k = 1)
			if chance==[1]:
				mutation_string.invert(i)
		return ConstBitStream(mutation_string)
		
	def evaluate_candidates(self, candidates): 
		'''
		args: candidate solutions (list) => each element is a bitstring (ConstBitStream)
		
		returns: parents (list of ConstBitStream) => each element is a bitstring (ConstBitStream) 
					but elements are not unique. Fittest candidates will have multiple copies.
					Size of 'parents' must be equal to population size.  
		'''
		# **** Start of Your Code **** #
		if self.wall_bit_string in candidates:
			print("The program has coverged at 100 percent")
			self.terminate=True

		total_similar_bits = []
		i=0
		for i in range(0,len(candidates),1):
			similarity_index = 0
			self.wall_bit_string.pos = 0
			candidates[i].pos = 0
			limit=56
			while self.wall_bit_string.read('bin:2') == candidates[i].read('bin:2') and candidates[i].pos < limit and self.wall_bit_string.pos < limit:
				similarity_index += 1
			self.wall_bit_string.pos = 0
			candidates[i].pos = 0
			total_similar_bits.append(similarity_index)

		f_list = []
		for j in range(len(candidates)):
			f_list.append((total_similar_bits[j]/28.0))
		f_avg=((sum(f_list))/28)/len(candidates)
		final_list=[]
		for j in range(len(candidates)):
    			final_list.append(f_list[j]/f_avg)
    		
		return random.choices(candidates, final_list, k = len(candidates))


		# **** Start of Your Code **** #
		# pass
		# **** End of Your Code **** # 



