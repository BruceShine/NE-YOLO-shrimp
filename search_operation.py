import math
import sys
from copy import deepcopy
from random import choice, choices, sample


class SearchOperation(object):
    def __init__(self, search_config, genotype_config):
        self.search_config = search_config
        self.genotype_config = genotype_config

        self.backbone_cell_types = self.genotype_config['backbone_cell_types']
        self.head_cell_types = self.genotype_config['head_cell_types']

    def add_cell(self, genotype_list):
        valid_list = [genotype for genotype in genotype_list
                      if len(genotype.genotype_dict['backbone']) + len(genotype.genotype_dict['head']) <
                      self.search_config['length_limit']]
        chosen_genotypes = sample(valid_list,
                                  k=int(math.ceil(len(genotype_list) * (self.search_config['add_cell_prob']/100.0))))

        for genotype in chosen_genotypes:
            print(f'genotype {genotype.genotype_key}')

            chosen_organization = choices(['backbone', 'head'], weights=self.search_config['organization_prob'], k=1)[0]
            chain_num = genotype.genotype_dict['chain_num']
            chosen_chain = choice([i for i in range(chain_num)])

            cell_list = [i for i in genotype.genotype_dict['chains'][chosen_organization][chosen_chain]]
            if len(cell_list) == 0:
                chosen_location = len(genotype.genotype_dict['head']) - 1
                insert_loc = 0
            else:
                chosen_location = choice(cell_list)
                insert_loc = \
                    genotype.genotype_dict['chains'][chosen_organization][chosen_chain].index(chosen_location)

            if chosen_organization == 'backbone':
                chosen_cell_type = choice(self.backbone_cell_types)
            else:
                chosen_cell_type = choice(self.head_cell_types)

            new_cell_gene = None
            candidate_cells = [cell for cell in genotype.genotype_dict[chosen_organization]
                               if cell[2] == chosen_cell_type and len(cell[-1]) < 4]

            if len(candidate_cells) > 0:
                new_cell_gene = deepcopy(choice(candidate_cells))
            else:
                if chosen_cell_type == 'Conv':
                    out_channels = self.genotype_config['conv_out_channels']
                    kernel_size = 3 if chosen_organization == 'backbone' else 1
                    stride = 2 if chosen_organization == 'backbone' else 1
                    new_cell_gene = [-1, 1, 'Conv', [out_channels, kernel_size, stride]]

                elif chosen_cell_type == 'C3':
                    out_channels = self.genotype_config['c3_out_channels']
                    shortcut = True if chosen_organization == 'backbone' else False
                    new_cell_gene = [-1, 1, 'C3', [out_channels, shortcut]]

                elif chosen_cell_type == 'nn.Upsample':
                    new_cell_gene = [-1, 1, 'nn.Upsample', [None, 2, 'nearest']]

            if new_cell_gene is not None:
                genotype.genotype_dict[chosen_organization].insert(chosen_location, new_cell_gene)

                genotype.genotype_dict['chains'][chosen_organization][chosen_chain].insert(insert_loc,
                                                                                           chosen_location)

                for i in range(chain_num):
                    if i == chosen_chain:
                        for j in range(len(genotype.genotype_dict['chains'][chosen_organization][i])):
                            if j > insert_loc:
                                genotype.genotype_dict['chains'][chosen_organization][i][j] += 1
                    elif i > chosen_chain:
                        for j in range(len(genotype.genotype_dict['chains'][chosen_organization][i])):
                            genotype.genotype_dict['chains'][chosen_organization][i][j] += 1

                # update the relation according to the chians dict
                for cell in genotype.genotype_dict['backbone']:
                    cell[0] = -1

                for cell in genotype.genotype_dict['head']:
                    cell[0] = -1

                for i in range(chain_num):
                    b_start = genotype.genotype_dict['chains']['backbone'][i][0]
                    genotype.genotype_dict['backbone'][b_start][0] = 0

                    if len(genotype.genotype_dict['chains']['head'][i]):
                        h_start = genotype.genotype_dict['chains']['head'][i][0]
                        h_start_value = genotype.genotype_dict['chains']['backbone'][i][-1]
                        genotype.genotype_dict['head'][h_start][0] = h_start_value

                genotype.genotype_dict['head'][-1][0] = []
                backbone_length = len(genotype.genotype_dict['backbone'])
                for i in range(chain_num):
                    head_length = len(genotype.genotype_dict['chains']['head'][i])
                    if head_length > 0:
                        head_end = genotype.genotype_dict['chains']['head'][i][-1]
                        genotype.genotype_dict['head'][-1][0].append(backbone_length + head_end)
                    else:
                        backbone_end = genotype.genotype_dict['chains']['backbone'][i][-1]
                        genotype.genotype_dict['head'][-1][0].append(backbone_end)

                print(f'add cell in {chosen_organization} - '
                      f'cell location: {chosen_location}, cell type: {chosen_cell_type}')
                print()

        print()

    def modify_cell(self, genotype_list):
        chosen_genotypes = sample(genotype_list,
                                  k=int(math.ceil(len(genotype_list) *
                                                  (self.search_config['modify_cell_prob']/100.0))))

        for genotype in chosen_genotypes:
            chosen_organization = choice(['backbone', 'head'])
            chosen_cell_list = genotype.genotype_dict[chosen_organization][:-1]

            if len(chosen_cell_list) > 0:
                chosen_cell = choice(genotype.genotype_dict[chosen_organization][:-1])

                if chosen_cell[2] == 'Conv':
                    chosen_cell_attribution = choices(['out_channels', 'kernel', 'stride'],
                                                      weights=self.search_config['attribution_prob'], k=1)[0]

                    if chosen_cell_attribution == 'out_channels':
                        chosen_cell[-1][0] += 8
                    elif chosen_cell_attribution == 'kernel':
                        chosen_cell[-1][1] += 2
                    elif chosen_cell_attribution == 'stride':
                        chosen_cell[-1][2] += 1

                    print(f'genotype {genotype.genotype_key}')
                    print(f'modify cell {chosen_cell} - {chosen_cell_attribution}')

                    if chosen_organization == 'backbone':
                        genotype.genotype_dict[chosen_organization][-1][-1][0] = \
                            genotype.genotype_dict[chosen_organization][-2][-1][0]

                elif chosen_cell[2] == 'C3':
                    chosen_cell[-1][0] += 8
                    print(f'genotype {genotype.genotype_key}')
                    print(f'modify cell {chosen_cell}')

                    if chosen_organization == 'backbone':
                        genotype.genotype_dict[chosen_organization][-1][-1][0] = \
                            genotype.genotype_dict[chosen_organization][-2][-1][0]

                # elif chosen_cell[2] == 'SPPF':
                #     chosen_cell_attribution = choices(['out_channels', 'pooling_kernel'],
                #                                       weights=[80, 20])[0]
                #
                #     if chosen_cell_attribution == 'out_channels':
                #         chosen_cell[-1][0] += 8
                #     else:
                #         chosen_cell[-1][1] += 2

        print()

    def crossover(self, genotype_list):
        chosen_genotypes = sample(genotype_list, k=int(math.ceil(len(genotype_list) *
                                                                 (self.search_config['crossover_prob']/100.0))))

        crossover_list_length = math.floor(len(chosen_genotypes) / 2)

        crossover_list = [[chosen_genotypes[i], chosen_genotypes[i+crossover_list_length]]
                          for i in range(crossover_list_length)]

        for parent in crossover_list:
            chosen_organization = choice(['backbone', 'head'])
            parent0_organ = deepcopy(parent[0].genotype_dict[chosen_organization])
            parent1_organ = deepcopy(parent[1].genotype_dict[chosen_organization])
            parent[0].genotype_dict[chosen_organization] = parent1_organ
            parent[1].genotype_dict[chosen_organization] = parent0_organ

            parent0_chains = deepcopy(parent[0].genotype_dict['chains'][chosen_organization])
            parent1_chains = deepcopy(parent[1].genotype_dict['chains'][chosen_organization])
            parent[0].genotype_dict['chains'][chosen_organization] = parent1_chains
            parent[1].genotype_dict['chains'][chosen_organization] = parent0_chains

            for genotype in parent:
                chain_num = genotype.genotype_dict['chain_num']
                for cell in genotype.genotype_dict['backbone']:
                    cell[0] = -1

                for cell in genotype.genotype_dict['head']:
                    cell[0] = -1

                for i in range(chain_num):
                    b_start = genotype.genotype_dict['chains']['backbone'][i][0]
                    genotype.genotype_dict['backbone'][b_start][0] = 0

                    if len(genotype.genotype_dict['chains']['head'][i]):
                        h_start = genotype.genotype_dict['chains']['head'][i][0]
                        h_start_value = genotype.genotype_dict['chains']['backbone'][i][-1]
                        genotype.genotype_dict['head'][h_start][0] = h_start_value

                genotype.genotype_dict['head'][-1][0] = []
                backbone_length = len(genotype.genotype_dict['backbone'])
                for i in range(chain_num):
                    head_length = len(genotype.genotype_dict['chains']['head'][i])
                    if head_length > 0:
                        head_end = genotype.genotype_dict['chains']['head'][i][-1]
                        genotype.genotype_dict['head'][-1][0].append(backbone_length + head_end)
                    else:
                        backbone_end = genotype.genotype_dict['chains']['backbone'][i][-1]
                        genotype.genotype_dict['head'][-1][0].append(backbone_end)

            print(f'crossover parents: {parent[0].genotype_key, parent[1].genotype_key}')

        print()

