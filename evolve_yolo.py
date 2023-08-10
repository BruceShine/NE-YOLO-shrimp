import os
import math
import yaml
from copy import deepcopy
from random import choice, choices
from itertools import count

import train
from genotype import Genotype
from search_operation import SearchOperation
from species import SpeciesSet


class Evolution(object):
    def __init__(self, genotype_config, species_config, search_config, train_config,
                 from_scratch=True, start_generation=1, train_epoch=10):
        self.genotype_config = genotype_config
        self.species_config = species_config
        self.search_config = search_config
        self.train_config = train_config

        self.from_scratch = from_scratch
        self.generation = start_generation
        self.population = {}

        if self.from_scratch:
            for i in range(1, self.genotype_config['genotype_number_init'] + 1):
                new_genotype = Genotype(i, genotype_config)
                new_genotype.init_genotype(self.genotype_config['init_genotype'])
                self.population[i] = new_genotype
        else:
            self.load_population('./genotypes/', start_generation)

        self.search = SearchOperation(search_config, genotype_config)

        self.speciation = SpeciesSet(species_config)

        if from_scratch:
            self.genotype_ceiling = self.genotype_config['genotype_number_limit']
            self.species_ceiling = self.species_config['species_number_limit']
        else:
            self.genotype_ceiling = self.genotype_config['genotype_number_limit'] - \
                                    10 * math.floor(self.generation / 5)
            self.species_ceiling = self.species_config['species_number_limit'] - \
                                   5 * math.floor(self.generation / 5)

        self.train_epoch = train_epoch

        self.genotype_indexer = count(max([genotype.genotype_key for genotype in self.population.values()]) + 1)

    def evolve_population(self, max_key=None):

        while self.generation <= self.search_config['generation_limit']:

            print(f'|========== Generation {self.generation} ==========|')

            # [Evolution]
            print('\n|---------- Take Evolution ----------|')
            parent_max_key = max([genotype_key for genotype_key in self.population.keys()])
            if self.from_scratch:
                offsprings = {}
                for genotype in self.population.values():
                    new_genotype = deepcopy(genotype)
                    new_genotype.genotype_key = next(self.genotype_indexer)
                    new_genotype.init_genotype()
                    offsprings[new_genotype.genotype_key] = new_genotype
                self.population.update(offsprings)

                offspring_list = [genotype for genotype_key, genotype in self.population.items()
                                  if genotype_key in list(offsprings.keys())]

                # keys of genetype evolved
                self.search.add_cell(offspring_list)

                self.search.modify_cell(offspring_list)

                self.search.crossover(offspring_list)

                # [Save genotypes]
                for genotype in self.population.values():
                    genotype.save_genotype(self.generation)

            # [Speciation]
            self.speciation.speciate(self.population)

            # [Training]
            if not self.from_scratch:
                parent_max_key = max_key
            train_dict = {}
            train_genotypes = []
            for species_key, species in self.speciation.species_dict.items():
                new_members = [member for member in species.members.values()
                               if member.genotype_key > parent_max_key]
                if len(new_members) > 0:
                    train_dict[species_key] = choice(new_members)
                else:
                    train_dict[species_key] = species.representation
                train_genotypes.append(train_dict[species_key])

            genotype_index = 1
            for genotype in train_genotypes:
                print(f'Generation {self.generation} '
                      f'genotype {genotype.genotype_key}({genotype_index}-{len(train_genotypes)}): ')
                genotype_fitness = genotype.genotype_dict['fitness']
                if genotype_fitness == 0.0:
                    results = train.run(data='shrimp.yaml', cfg=genotype.genotype_dict,
                                        imgsz=1000, batch_size=8, weights='', epochs=self.train_epoch, device='0')
                    genotype.genotype_dict['fitness'] = float(results[2]) \
                        if float(results[2]) > 0.0 else float(results[2]) + 1e-6
                    genotype.save_genotype(self.generation)
                    print(f'individual {genotype.genotype_key} fitness: {results[2]}\n')
                else:
                    print(f'individual {genotype.genotype_key} fitness: {genotype_fitness}\n')

                if genotype.genotype_dict['fitness'] > self.train_config['fitness_threshold'] and \
                        genotype.genotype_dict['final_fitness'] == 0.0:
                    print(f'individual {genotype.genotype_key} reached fitness threshold...')
                    results = train.run(data='shrimp.yaml', cfg=genotype.genotype_dict,
                                        imgsz=1000, batch_size=8, weights='', epochs=300, device='0')
                    genotype.genotype_dict['final_fitness'] = float(results[2])
                    genotype.save_genotype(self.generation)
                    print(f'individual {genotype.genotype_key} final fitness {results[2]}')

                    if genotype.genotype_dict['final_fitness'] > self.train_config['final_fitness_threshold']:
                        f = open(self.search_config['output_file'], 'a')
                        print(f'\nThe SATISFIED individual:', file=f)
                        genotype.print_architecture(f, True)
                        f.close()
                        return

                genotype_index += 1

            # [Fitness]
            for species_key, trained_member in train_dict.items():
                species_fitness = trained_member.genotype_dict['fitness']

                species_representation = \
                    sorted([member for member in self.speciation.species_dict[species_key].members.values()],
                           key=lambda m: m.genotype_dict['fitness'], reverse=True)[0]
                self.speciation.species_dict[species_key].update(species_fitness, species_representation)

            # [Selection]
            # control spec+ies number
            current_genotype_number = len(self.population)
            sorted_species = sorted(self.speciation.species_dict.values(),
                                    key=lambda s: s.fitness, reverse=True)
            if len(self.speciation.species_dict) > self.species_ceiling:
                new_species_dict = {}
                for i in range(self.species_ceiling):
                    new_species_dict[sorted_species[i].species_key] = sorted_species[i]
                self.speciation.species_dict = new_species_dict

                current_genotype_number = 0
                for species in self.speciation.species_dict.values():
                    current_genotype_number += len(species.members)

            # control each species member number
            if current_genotype_number > self.genotype_ceiling:
                for species in self.speciation.species_dict.values():
                    reserved_number = math.ceil(len(species.members) / current_genotype_number *
                                                self.genotype_ceiling)
                    reserved_members = [species.representation]
                    if reserved_number > 1:
                        member_list = [member for member in species.members.values()
                                       if member.genotype_key != species.representation.genotype_key]
                        choice_weights = [member.genotype_dict['fitness'] for member in member_list]
                        reserved_members += choices(member_list, weights=choice_weights, k=reserved_number-1)
                    new_members = {}
                    for member in reserved_members:
                        new_members[member.genotype_key] = member
                    species.members = new_members

            # construct population for next generation
            next_population = {}
            for species in self.speciation.species_dict.values():
                for genotype in species.members.values():
                    next_population[genotype.genotype_key] = genotype
            self.population = next_population

            print('\n|========== Species ==========|')
            for species_id, species in self.speciation.species_dict.items():
                print(f'species id      : {species_id}')
                print(f'species members : {list(species.members.keys())}')

            # output evolution information in current generation
            f = open(self.search_config['output_file'], 'a')
            self.output_evolution_info(f)
            self.speciation.output_species_dict_info(f)
            f.close()

            # [evolution setting]
            if (self.generation + 1) % 5 == 0 and self.genotype_ceiling > 20:
                self.genotype_ceiling -= 10

            if (self.generation + 1) % 5 == 0 and self.species_ceiling > 5:
                self.species_ceiling -= 5

            if self.train_epoch < 150:
                self.train_epoch += 10
            self.from_scratch = 1

            self.generation += 1

    def load_population(self, genotype_dir, generation_index):
        genotype_files = os.listdir(genotype_dir)
        for genotype_file in genotype_files:
            if len(str(generation_index)) == 1:
                genotype_generation = genotype_file[9]
            else:
                genotype_generation = genotype_file[9:11]

            if genotype_generation == str(generation_index):
                with open(genotype_dir+genotype_file, encoding='ascii', errors='ignore') as f:
                    genotype_dict = yaml.safe_load(f)
                f.close()
                new_genotype = Genotype(genotype_dict['key'], self.genotype_config)
                new_genotype.genotype_dict = genotype_dict
                self.population[new_genotype.genotype_key] = new_genotype

    def output_evolution_info(self, output_file):
        print(f'|========== Generation {self.generation} ==========|\n', file=output_file)
        print(f'genotype ceiling: {self.genotype_ceiling}', file=output_file)
        print(f'species ceiling: {self.species_ceiling}', file=output_file)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, 'evolve_config.yaml')

    with open(config_file, encoding='ascii', errors='ignore') as f:
        evolve_config = yaml.safe_load(f)
    f.close()
    genotype_config = evolve_config['genotype_config']
    species_config = evolve_config['species_config']
    search_config = evolve_config['search_config']
    train_config = evolve_config['train_config']

    evolve_yolo = Evolution(genotype_config, species_config, search_config, train_config)
    # evolve_yolo = Evolution(genotype_config, species_config, search_config, train_config, from_scratch=False,
    #                         start_generation=12, train_epoch=120)
    evolve_yolo.evolve_population()


