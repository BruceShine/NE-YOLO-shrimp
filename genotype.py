import yaml


class Genotype(object):
    def __init__(self, genotype_key, genotype_config):
        self.genotype_key = genotype_key
        self.genotype_config = genotype_config

        self.genotype_dict = {}

    def init_genotype(self, genotype_file=None):
        if genotype_file is not None:
            with open(genotype_file, encoding='ascii', errors='ignore') as f:
                self.genotype_dict = yaml.safe_load(f)
            f.close()

        self.genotype_dict['key'] = self.genotype_key
        self.genotype_dict['fitness'] = 0.0
        self.genotype_dict['final_fitness'] = 0.0

    def print_architecture(self, output_file, print_fitness=False):
        print('genotype key', self.genotype_key, file=output_file)
        print('genotype dict:', file=output_file)
        for cell_gene in self.genotype_dict['backbone']:
            print(cell_gene, file=output_file)
        for cell_gene in self.genotype_dict['head']:
            print(cell_gene, file=output_file)
        if print_fitness:
            print('fitness:', self.genotype_dict['fitness'], file=output_file)
            if self.genotype_dict['final_fitness'] > 0.0:
                print('final fitness:', self.genotype_dict['final_fitness'], file=output_file)
        print()

    def save_genotype(self, generation):
        genotype_file_name = './genotypes/genotype-' + str(generation) + '-' + str(self.genotype_key) + '.yaml'
        with open(genotype_file_name, 'w') as file:
            yaml.safe_dump(self.genotype_dict, file, sort_keys=False)
        file.close()



