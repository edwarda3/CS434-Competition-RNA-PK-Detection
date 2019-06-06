import pandas
import numpy

f103 = ['predictions/on_testing/features103_pred{}.txt'.format(i) for i in range(1,4)]
fall = ['predictions/on_testing/featuresall_pred{}.txt'.format(i) for i in range(1,4)]

files = f103 + fall
for f in files:
    df = pandas.read_csv(f,header=None)
    pos = numpy.sum(df.values[:,1])
    print('{}/{} ({:.4f}%)  predicted positive'.format(pos,len(df),pos/len(df)))