import sqlite3, json, math, pickle, os, signal
from annotate import Annotation
from itertools import combinations
from statistics import mean
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, mannwhitneyu

class TimeoutError(Exception):
    pass

def TimeoutHandler(signum, frame):
    raise TimeoutError


class ConfusionFactor():
    def __init__(self):
        self.conn = None
        self.cursor = None
        self.wikitable = 'wikiTables.txt'
        self.dbname = 'relation.db'
        self.annotation = Annotation()

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.dbname)
            self.cursor = self.conn.cursor()
        except:
            return 1

    def clean(self):
        try:
            self.cursor = None
            self.conn.close()
            return 0

        except:
            self.cursor = None
            self.conn = None

            return 1


    def computeFactor2(self, table):
        '''
        parameter: table in json format

        '''
        self.connect()

        headers = table['header']
        values = table['values']
        entities = table['entity']
        unit = table['unit']

        # resolve entities symbols
        wids = []
        for entity in entities:
            wid = self.findWikiID(entity)
            if wid == None:
                continue
            else:
                wids.extend(wid)

        wids = list(set(wids))

        eType = self.annotation.getEntityType(wids)
        if eType == None:
            return None

        # size of the given column
        s = len(values)
        m = mean(list( map(float, values) ))
        r = 40

        summation = 0
        total = 0

        allTypes = self.getAllTypes()
        print(len(allTypes))
        i = 0
        for t in allTypes:
            i+=1
            if (i % 500 == 0): print (i)
            if (i > 8000): break
            distributions = self.annotation.getDistribution([t])
            for key in distributions:
                distribution = distributions[key]
                if len(distribution) == 0:
                    continue
                num = 0
                denom = 0

                total += 1
                # compute number of columns in within distance r
                m_prime = mean(distribution)
                if abs(m - m_prime) <= r:
                    num += 1
                else:
                    continue

                # generate subsets
                if (len(distribution) < s):
                    continue

                subsets = combinations(distribution, s)
                for subset in subsets:
                    m_doubleprime = mean(list(subset))
                    if abs(m - m_doubleprime) <= r:
                        denom += 1
                        break

                try:
                    summation += (num / denom)
                except:
                    pass

        factor = summation / total
        print (eType, factor)   


        self.clean()

        return (eType, factor)


    def resolve(self, wid):
        query = '''select label from mapping where wid=?; '''
        self.cursor.execute(query, (wid,))
        row = self.cursor.fetchone()
        if row == None:
            return None

        return row[0]

    def resolveTriple(self, triple):
        rv = tuple()
        for wid in triple:
            rv += (self.resolve(wid),)

        return rv


    def findWikiID(self, label):
        query = '''select wid from mapping where label=?; '''
        self.cursor.execute(query, (label,))

        ids = []
        row = self.cursor.fetchone()
        while row != None:
            ids.append(row[0])
            row = self.cursor.fetchone()

        if len(ids) == 0:
            return None
        else:
            return ids



    def getAllTypes(self):
        query = '''select type, property, unit from relation; '''
        self.cursor.execute(query)

        rv = self.cursor.fetchall()

        return rv

    def getDistribution(self, key):
        query = '''select amount from distribution where type = ? and property = ? and unit = ?; '''


        self.cursor.execute(query, key)
        d = []
        row = self.cursor.fetchone()
        while row != None:
            d.append(row[0])
            row = self.cursor.fetchone()

        return np.array(d)

    def computeCost(self, arr1, arr2):
        # arr1 always have smaller size
        if len(arr1) > len(arr2): raise ValueError('size of query column is greater than data column')

        # the algorithm may not work with float weights
        factor = 10e4
        arr1 = [int(factor*x) for x in arr1]
        arr2 = [int(factor*x) for x in arr2]

        # we can assume the given lists are ordered
        #arr1 = sorted(arr1)
        #arr2 = sorted(arr2)

        # time limit is 10 min
        signal.alarm(600)

        G = nx.DiGraph()
        G.add_node('s', demand = -1 * len(arr1))
        G.add_node('t', demand = len(arr1))
        for i in range(len(arr1)):
            # connect source node to all query nodes
            G.add_edge('s', i, capacity=1, weight=0)
            for j in range(len(arr2)):
                nodenum = j + len(arr1)
                # may have different distance functions
                distance = abs(arr2[j] - arr1[i])

                G.add_edge(i, nodenum, capacity=1, weight= distance)

        # connect all data nodes to sink. cost of edges are 0
        for j in range(len(arr2)):
            nodenum = j + len(arr1)
            G.add_edge(nodenum, 't', capacity=1, weight=0)

        # compute flow with min cost
        #mincostFlow = nx.max_flow_min_cost(G, 's', 't')
        mincostFlow = nx.min_cost_flow(G)
        mincost = nx.cost_of_flow(G, mincostFlow)

        # reset timer
        signal.alarm(0)
        
        return mincost/factor

    def estimate(self, arr1, arr2):
        # use greedy algorithm to compute an estimate of cost
        # we can assume the given lists are ordered
        #arr1 = sorted(arr1)
        #arr2 = sorted(arr2)

        # arr1 always have smaller size
        if len(arr1) > len(arr2): arr1, arr2 = arr2, arr1

        est = 0
        for src in arr1:
            diff = np.abs(np.subtract(arr2, src))
            idx = np.argmin(diff)
            
            est += diff[idx]
            np.delete(arr2, idx)
            #arr2.pop(idx)



        '''
        est = 0
        curr = 0
        matchedIdx = set()

        prev, curr = None, None

        # find the best match for the first number from source, record the matched index
        src = arr1[0]
        for i in range(1, len(arr2)):
            target = arr2[i]
            prevCost = abs(arr2[0] - src)
            cost = abs(target - src)

            if cost > prevCost:
                est += cost

                # remove the matched number from target
                arr2.pop(i - 1)
                prev = i - 1
                curr = min(i, len(arr2)-1)
                break

        # if cost never increase, then the number matches with the last one.
        if prev == None:
            cost = abs(target - src)
            est += cost

            arr2.pop(i - 1)
            prev = i - 1
            curr = len(arr2) - 1   


        # keep looking for best matches for the remaining numbers using greedy algorithm
        for i in range(1, len(arr1)):
            src = arr1[i]
            prevCost = abs(arr2[prev] - src)

            found = 0
            state = None
            direction = None
            cost = None
            while not found:
                currentCost = abs(arr2[curr] - src)

                if state == None:
                    if currentCost > prevCost:
                        curr -= 1
                        direction = '-'
                    elif currentCost < prevCost:
                        curr += 1
                        direction = '+'
                    else:
                        found = 1
                        cost = 0

                    state = 'begin'


                elif state == 'begin':
                    if currentCost > prevCost:
                        if direction == '+':
                            found = 1
                            cost = abs(arr2[prev] - src)
                        curr -= 1

                    elif currentCost < prevCost:
                        if direction == '-':
                            found = 1
                            cost = abs(arr2[prev] - src)

                        curr += 1
                    else:
                        found = 1
                        cost = 0


                prev = curr
                prevCost = currentCost
                if found:
                    arr2.pop(prev)

            est += cost
        '''

        return est

    

    def analyze(self):
        self.connect()
        signal.signal(signal.SIGALRM, TimeoutHandler)

        cfvalues = {}

        dataset = {}

        #fname = 'datafiles/CFresult_1std.pickle'
        #with open(fname, 'rb') as f:
        #    cfvalues = pickle.load(f)


        with open('allColumns.pickle', 'rb') as f:
            poplist = []
            dataset = pickle.load(f)
            for key in dataset:
                dataset[key] = np.array(dataset[key])
                if len(dataset[key]) == 1 or len(dataset[key]) > 1000:
                    poplist.append(key)

        for p in poplist:
            dataset.pop(p)

        allTypes = set()
        for key in dataset:
            allTypes.add(key[0])

        allTypes = np.random.choice(list(allTypes), size=int(0.5*len(allTypes)))

        a = {}
        l = []
        
        result_map = {1:0, 3:0, 5:0}
        result_mean = {1:0, 3:0, 5:0}
        total = 0
        
        for t in allTypes:
            columnKeys = []
            for key in dataset:
                if key[0] == t: columnKeys.append(key)

            tp = self.resolve(t)
            if tp == None: tp = 'unknown'


            for key in columnKeys:
                if total%100 == 0: print(total)

                pLabel = self.resolve(key[1])
                c = dataset[key]
                np.random.shuffle(c)

                samplesize = min(int(0.25*len(c)), 20)
                sample = c[:samplesize]
                c = c[samplesize:]

                if len(c) == 0: continue

                m = np.mean(c)
                r = np.std(c)

                unit = key[2]
                u = self.resolve(unit)
                if u == None: 
                    u = '1'

                matches = 0
                matched_mean = set()

                #l.append((key[0], key[1], key[2], m, pLabel,))
                frequentTerm = {}

                costs = {}

                try:
                
                    for key2 in columnKeys:
                        # ignore columns with different entity type
                        #if key == key2: continue
                        
                        c2 = dataset[key2]
                        if len(c2) < len(c): continue

                        m2 = np.mean(c2)

                        unit2 = key2[2]

                        if abs(m - m2) <= r:# and unit==unit2:
                            matches += 1
                            matched_mean.add(key2)

                        costs[key2] = self.computeCost(c, c2)

                except TimeoutError:
                    continue

                total += 1
                    
                l = sorted(costs.items(), key=lambda x:x[1])

                # top 1 result
                matched = l[:1]
                top = [x[0] for x in matched]
                if key in top: result_map[1] += 1

                # top 3 result
                matched = l[:3]
                top = [x[0] for x in matched]
                if key in top: result_map[3] += 1

                # top 5 result
                matched = l[:5]
                top = [x[0] for x in matched]
                if key in top: result_map[5] += 1

                
                if matches == 1 and key in matched_mean:
                    result_mean[1] += 1

                if matches <= 3 and key in matched_mean:
                    result_mean[3] += 1

                if matches <= 3 and key in matched_mean:
                    result_mean[5] += 1

                    

                #frequentTerm = sorted(frequentTerm.items(), key=lambda x: x[1], reverse=True)
                #freq = []
                #for i, t in enumerate(frequentTerm):
                #    if i < 10:
                #        freq.append(t[0])
                #freq = ' '.join(freq)

                #a[(tp, m, matches, pLabel)] = '2'



        #self.cursor.executemany('insert into means(type, property, unit, mean, plabel) VALUES (?,?,?,?,?);', l)
        #self.conn.commit()

        print('total: ', total)
        print('top k nearest neighbour precision (mapping)')
        print('k = 1: ', result_map[1]/total)
        print('k = 3: ', result_map[3]/total)
        print('k = 5: ', result_map[5]/total)

        print('--------------------')
        print('top k nearest neighbour precision (mean distance)')
        print('k = 1: ', result_mean[1]/total)
        print('k = 3: ', result_mean[3]/total)
        print('k = 5: ', result_mean[5]/total)

        '''
        b = sorted(a.keys(), key=lambda x: x[2], reverse=True)
        with open('results.txt', 'w') as f:
            for item in b:
                item = list(map(str,item))
                s = ', \t'.join(item)
                s += '\n'
                f.write(s)
        '''
        
        
            


        
        '''
        # compute percentage of hard to distinguish property for each entity type
        type_prop = {}
        for key in dataset:
            tp = key[0]
            prop = key[1]
            if tp in type_prop:
                if prop in type_prop[tp]:
                    type_prop[tp][prop].append(key)
                else:
                    type_prop[tp][prop] = [key]
            else:
                type_prop[tp] = {}
                type_prop[tp][prop] = [key]

        count = {}
        count['1-5'] = 0 # 1-5
        count['6-10'] = 0 # 6-10
        count['11-20'] = 0 # 11-20
        count['>20'] = 0 # >20

        res = []

        for tp in type_prop:
            props = type_prop[tp]
            columnLabels = []

            numDistinct = 0

            
            totalColumns = sum([len(props[p]) for p in props])

            for p in props:
                columnLabels = props[p]
                for cl in columnLabels:
                    c = dataset[cl]
                    m = np.mean(c)
                    r = np.std(c)
                    matches = 0

                    for p2 in props:
                        if p == p2: continue
                        columnLabels2 = props[p2]
                        for cl2 in columnLabels2:
                            c2 = dataset[cl2]

                            #(statistic, pvalue) = mannwhitneyu(c, c2, alternative="two-sided")
                            #(statistic, pvalue) = ttest_ind(c, c2, equal_var=False)
                            #if pvalue > 0.1: matches += 1

                            
                            # using mean distance
                            m2 = np.mean(c2)
                            if abs(m-m2) <= r:
                                matches += 1
                            


                if (totalColumns <= 10 and matches < 3) or \
                    (totalColumns > 10 and matches < 0.2*totalColumns):
                    numDistinct += 1

            ratio = numDistinct/totalColumns
            if ratio < 0.5: 
                if totalColumns <= 5:
                    count['1-5'] += 1
                elif totalColumns > 5 and totalColumns <= 10:
                    count['6-10'] += 1
                elif totalColumns > 10 and totalColumns <= 20:
                    count['11-20'] += 1
                else:
                    count['>20'] += 1

            if totalColumns > 20:
                if self.resolve(tp) != None:
                    res.append([self.resolve(tp), numDistinct/totalColumns, totalColumns])
                #print(self.resolve(tp), '{0:.3}'.format(numDistinct/totalColumns), totalColumns)

        res = sorted(res, key=lambda x:x[1], reverse=True)
        for r in res:
            print(r[0], '{0:.3}'.format(r[1]), r[2])
        #for key in count:
        #    print(key, count[key])
        '''
                

            



        

        '''
        with open('expUTest.pickle', 'wb') as f:
            exp = {}
            for key in dataset:
                column = dataset[key]
                if len(column) == 1: continue

                m = np.mean(column)
                r = np.std(column)
                expvalue = 0

                for key2 in dataset:
                    c = dataset[key2]
                    if len(c) == 1: continue

                    (statistic, pvalue) = mannwhitneyu(column, c, alternative="two-sided")
                    if pvalue < 0.005: expvalue += 1

                exp[key] = expvalue

            pickle.dump(exp, f)
        '''
        
        '''
        exp = {}
        with open('expectedValues.pickle', 'rb') as f:
            exp = pickle.load(f)

        
        match = {}
        confuse = {}
        for key in exp:
            c = dataset[key]
            if len(c) == 1: continue

            if exp[key] <= 5000:     
                match[key] = np.mean(c)
            else:
                confuse[key] = np.mean(c)


        print(len(match), len(confuse))
        '''


        '''
        matchone = {}
        low = {}
        high = {}
        for key in cfvalues:
            if cfvalues[key] == 0:
                matchone[key] = 0
            elif cfvalues[key] < 0.0001:
                low[key] = cfvalues[key]
            else:
                high[key] = cfvalues[key]

        print (len(matchone), len(low), len(high))

        data = []
        for key in matchone:
            if len(dataset[key]) == 1: continue
            m = np.mean(dataset[key])
            data.append(m)            
        for key in low:
            m = np.mean(dataset[key])
            data.append(m)
        for key in high:
            m = np.mean(dataset[key])
            data.append(m)

        data = np.array(data)
        bins = [-30, 0, 30, 60]

        
        histogram, bins = np.histogram(data, bins=bins)
        print(histogram)
        histogram = np.divide(histogram, len(data))

        bin_centers = 0.5*(bins[1:] + bins[:-1])

        plt.figure(figsize=(6, 4))
        plt.plot(bin_centers, histogram, label="Histogram of samples")
        plt.legend()
        plt.show()
        '''

            

        self.clean()



    def KBStat(self):
        '''
        compute the Confusion factor for columns (properties) in knowledge base

        '''

        getDistribution = '''select amount from distribution where type = ? and property = ? and unit = ? '''
        # for each property we find the CF value
        self.connect()
        columns = {}
        cfvalues = {}
        pairDistance = {}

        if (not os.path.exists('./types.pickle')) or (not os.path.exists('./allColumns.pickle')):

            types = self.getAllTypes()
            with open('types.pickle', 'wb') as f:
                pickle.dump(types, f)

            for t in types:
                d = self.getDistribution(t)
                if len(d) != 0:
                    columns[t] = d

            with open('allColumns.pickle', 'wb') as f:
                pickle.dump(columns, f)

        else:
            with open('allColumns.pickle', 'rb') as f:
                columns = pickle.load(f)

        for key in columns:
            columns[key] = np.array(columns[key])

        data = columns.values()
        r = 20   #TODO: vary the value of r

        f2 = open('cf.txt', 'w')
        for i, key in enumerate(columns):
            if i % 1000 == 0:
                print(i)
                f2.flush()
            r = np.std(columns[key])
            cf = self.computeCF(columns[key], data, r, pairDistance)
            cfvalues[key] = cf

            f2.write(str(key) + ': ' + str(cf) + '\n')
            
        f2.close()
        with open('datafiles/CFresult.pickle', 'wb') as f:
            pickle.dump(cfvalues, f)

        return 0

    def validate(self):
        size1 = 20
        size2 = 40
        size3 = 60

        mean1 = 20
        mean2 = 40
        mean3 = 60
        mean4 = 100
        mean5 = 200
        d = {}

        sizes = [size1, size2, size3]
        means = [mean1, mean2, mean3, mean4, mean5]

        length = range(5,20)

        r = 0.4
        size = 20
        s = 5
        # generate different distributions
        # 2 random distributions for each mean
        normals = []
        for mean in means:
            normals.append(np.random.normal(loc=mean, scale=10, size=size))
            normals.append(np.random.normal(loc=mean, scale=10, size=size))
        
        s = 5
        size = 20
        r = 0.2
        total = 0
        for i in range(100):
            uniforms = []
            for mean in means:
                uniforms.append(np.random.uniform(0, 2*mean, size=size))
                uniforms.append(np.random.uniform(0, 2*mean, size=size))
            
            query = np.random.uniform(0, 2*means[0], size=s)
            total += self.computeCF(query, uniforms, r, d)
        print(total/100)



    def computeCF(self, column, dataset, r, d):
        m = np.mean(column)

        matched_means = []
        for c in dataset:
            m_prime = np.mean(c)
            if abs(m - m_prime) <= r:
                matched_means.append(m_prime)

            #print(m, m_prime)

        #print(matched_means)
        if len(matched_means) == 0:
            return 0
        elif len(matched_means) == 1:
            return 0

        total = 0.0
        for m_prime in matched_means:
            total += self.sigmoid(abs(m - m_prime))
        return total / len(dataset)


    def sigmoid(self, x):
        s = 1 / (1 + np.exp(-0.1*x))

        return (-2*s + 2)


def meangroup(prop, label, mean):

    tpcount = {}
    total = {}
    bins = [-100, 0, 100, 200, 500, 700, 1000, 100000, 1000000]
    h = []

    for i in range(len(bins)):
        h.append(0)
    

    with open('results.txt') as f:
        line = f.readline().strip()
        data = []

        cnt = [0, 0, 0]
        while line != '':
            l = line.split(', \t')
            tp = l[-1]
            #unit = l[1]
            mean = float(l[1])
            matches = int(l[2])
            if tp in total:
                total[tp] += 1
            else:
                total[tp] = 1

            if matches == 0:
                cnt[0] += 1
            elif matches == 1:
                cnt[1] += 1
            

            if matches > 2:
                if tp in tpcount:
                    tpcount[tp] += 1
                else:
                    tpcount[tp] = 1

                data.append(mean)
                for i in range(len(bins)):
                    cur = bins[i]
                    if i == len(bins) - 1:
                        if mean >= cur: h[i] += 1
                    else:
                        upper = bins[i+1]
                        if mean >= cur and mean < upper:
                            h[i] += 1



            line = f.readline().strip()

    cnt[2] = cnt[0] + cnt[1]
    print (cnt, cnt[0] / cnt[2])
    #print(h)
    #print(sum(h))

    '''
    # number of columns vs range graph
    ticklabel = []
    for i,j in enumerate(bins):
        if i == len(bins) - 1:
            ticklabel.append('> '+str(bins[-1]))
        else:
            ticklabel.append('{} - {}'.format(bins[i],bins[i+1]))

    fig,ax = plt.subplots()
    ax.bar(range(len(h)),h,width=1,align='center',tick_label=ticklabel, color='skyblue')
    fig.autofmt_xdate()
    plt.show()
    '''
    
    #plt.hist(data, bins=100)
    #plt.show()
    #histogram, bins = np.histogram(data, bins=bins)
    #print(histogram)
    #histogram = np.divide(histogram, len(data))

    #bin_centers = 0.5*(bins[1:] + bins[:-1])

    #plt.figure(figsize=(6, 4))
    #plt.plot(bins, h, label="#hardcolumns vs ranges")
    #plt.legend()
    


    '''
    # number of hard columns vs property graph
    for key in tpcount:
        tpcount[key] /= total[key]
    a = sorted(tpcount.items(), key=lambda x:x[1], reverse=True)
    label = []
    share = []
    others = 0

    for i, item in enumerate(a):
        if item[0] == 'unknown': continue
        if i < 20:
            label.append(item[0])
            share.append(item[1])
        else:
            others += item[1]

    #label.append('others')
    #share.append(others)

    for i, j in enumerate(label):
        print(label[i], share[i])

    #fig,ax = plt.subplots()
    #ax.bar(range(len(share)),share,width=1,align='center',tick_label=label, color='green')
    #fig.autofmt_xdate()
    #plt.show()
    '''

    
    




    return 0



if __name__ == '__main__':
    c = ConfusionFactor()
    c.annotation.connect()
    table = '''{"header": ["Drug", "Biological half-life [h]"], "label": "half-life", "values": ["98.7", "99.8", "15", "95", "70", "5", "13"], "entity": ["Losartan", "EXP 3174", "Candesartan", "Valsartan", "Irbesartan", "Telmisartan", "Eprosartan", "Olmesartan", "Sources:"], "unit": "h"} '''
    #table = json.loads(table)
    #c.validate()
    #c.KBStat()
    c.analyze()
    #meangroup('1', '1', '1')
    c.annotation.clean()
    