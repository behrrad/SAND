from relationGraph import RelationGraph
from annotate import Table, Annotation
import pickle
import numpy as np
from confusion import ConfusionFactor

def main():
    #     bunzip2 -c /Volumes/T7\ Touch/latest-all.json.bz2 | python3 script.py
    path = 'data/'
    outpath = './'
    
    g = RelationGraph()
    g.typeHierarchy()
    # g.extractType()
    return
    #g.extractAliases()
    #g.extractFromNT()
    #g.extractFromJson()
    #g.pushToDB('relation')
    #g.writeToFile('./')
    #g.__resolveUnits__()
    #g.__resolveSymbols__()
    #g.clean()

    # t = Table()
    # t.processTable()

    sampleSize = 10
    a = Annotation()
    a.connect()

    with open('dbpedia/distribution.pickle', 'rb') as f:
        columns = pickle.load(f)
    
    for k in list(columns.keys())[:200]:
        query = []
        k = [i for i in k]

        column = columns[(k[0], k[1])]
        if len(column) < 10: continue
        query = list(np.random.choice(column, min(sampleSize, int(len(column)/10)), replace=False))

        eType = k[0]
        allkeys = [i for i in columns.keys() if i[0] == k[0]]
        distributions = {}
        for i in allkeys:
            distributions[i] = columns[i]

        pred = a.predict(eType, query, distributions)
        print ('label: ', k[0], k[1], ' prdiction:', pred)

    a.clean()
    
if __name__ == '__main__':
    main()
