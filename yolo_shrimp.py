import train


train.run(data='shrimp.yaml',
          cfg='./genotypes/genotype0.yaml',
          imgsz=1000,
          batch_size=8,
          weights='',
          epochs=300,
          device='0')



