from relationGraph import RelationGraph
import sqlite3, json, math, pickle, os, signal, re, requests
from itertools import combinations
import networkx as nx
import numpy as np
import heapq
import time
from scipy.stats import ttest_ind, mannwhitneyu, ks_2samp, zscore
from wikimapper import WikiMapper
from get_dbpedia_data import get_features_and_kinds_multi_thread


class TimeoutError(Exception):
    pass


def TimeoutHandler(signum, frame):
    raise TimeoutError


class Annotation():
    def __init__(self):
        self.tables = []
        self.graph = RelationGraph()

        self.conn = None
        self.cursor = None
        self.wdc = 'dataset/wdc.txt'
        self.dbname = 'relation_v2.db'
        # self.dbname = 'relation-dbpedia.db'
        self.getEntityQuery = 'https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&ids=%s&languages=en'
        self.searchQuery = 'https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&search=%s&language=en'

        self.k = 3

        self.ineligible = 1
        self.wikimapper = WikiMapper("index_enwiki-latest.db")

    def connect(self):
        if self.conn == None:
            self.conn = sqlite3.connect('./' + self.dbname)
            self.cursor = self.conn.cursor()

    def clean(self):
        self.cursor = None
        if self.conn != None:
            self.conn.close()
            self.conn = None

    def __loadTable__(self, table):
        columns = []
        table = json.loads(table)

        return columns

    def resolve(self, symbol, reverse=False):
        if reverse:
            rv = []
            query = '''select wid from mapping where label = ?; '''
            self.cursor.execute(query, (symbol,))
            row = self.cursor.fetchone()
            while row != None:
                rv.append(row[0])
                row = self.cursor.fetchone()
            return rv

        else:
            query = '''select label from mapping where wid=?; '''
            self.cursor.execute(query, (symbol,))
            row = self.cursor.fetchone()
            if row == None:
                return None

            return row[0]

    def resolveTriple(self, triple):
        if len(triple) == 0: return None

        rv = tuple()
        for wid in triple:
            rv += (self.resolve(wid),)

        return rv

    def get_parents_to_the_root(self, graph, wid):
        data = []
        data.extend(wid)
        checked_parents = []
        while len(data) > 0:
            edges = graph.in_edges(data[0])
            for edge in list(edges):
                if list(edge)[0] in checked_parents:
                    continue
                checked_parents.append(list(edge)[0])
            data = list(dict.fromkeys(data))
            # checked_parents.append(data[0])
            del data[0]
        return checked_parents

    def createOrder(self, query, dataColumns):
        # create an ordering of processing based on estimated lower bound cost
        queue = []
        topK = []
        for i in range(self.k):
            heapq.heappush(topK, (np.inf, tuple()))

        for item in dataColumns:
            key = item[0]
            column = item[1]
            # NEW CODE
            if len(column) < 3: continue
            # if len(query) > len(column): continue
            cost = self.estimateByRange(query, column)

            heapq.heappush(queue, (cost, key,))
        return queue, topK

    def estimateByRange(self, query, data):
        rd = (min(data), max(data))
        rq = (min(query), max(query))

        cost = 0

        # query column is within data column
        if (rq[0] > rd[0] and rq[1] < rd[1]):
            cost = 0
        # entire query column is less than data column
        elif (rq[1] < rd[0]):
            for num in rq:
                cost += abs(rd[0] - num)
        # entire query column is greaater than data column
        elif (rq[0] > rd[1]):
            for num in rq:
                cost += abs(num - rd[1])

        # there exist partial overlap between columns
        # map non overlap numbers to the minimum value
        elif (rq[0] < rd[0]):
            for num in rq:
                if num < rd[0]: cost += abs(num - rd[0])
        # map non overlap numbers to the maximum value
        elif (rq[1] > rd[1]):
            for num in rq:
                if num > rd[1]: cost += abs(num - rd[1])
        # new code
        if len(query) > len(data):
            cost = cost * min(50, len(query)) / min(50, len(data))
        return cost

    def pruneUp(self, query, data, topK):
        # if the estimated lower bound is greater than the largest cost in topK, we can prune this column
        if len(data) < 3:
            return -1
        est = self.estimateByRange(query, data)

        currentMax = topK[-1][0]  # heapq.nlargest(1, self.topK, key=lambda x:x[0])

        return est

    def pruneDown(self, query, data, topK):
        # find lower bound by mapping each element from query to the nearest element in data
        data = np.array(data)
        currentMax = topK[-1][0]

        est = 0
        for num in query:
            est += np.min(np.abs(data - num))
        # new code
        if len(query) > len(data):
            est = est * min(len(query), 50) / min(len(data), 50)
        return est

    def testReduction(self):
        '''
        test for 
        (1): when the query columns is large (larger than some threshold), pick n sample subsets and
             compare the precision when varying n.

        (2): when |q| > |c|, allowing |q|/|c| replacements for each element.
            partition the knowledge columns at ratio 4:6, and pick the larger columns as query column.
        '''

        with open('reformed.pkl', 'rb') as f:
            allColumns = pickle.load(f)

        query = {}
        data = {}
        ratio = 0.2
        for eType in allColumns:
            query[eType] = {}
            data[eType] = {}
            for prop in allColumns[eType]:
                v = allColumns[eType][prop]
                if len(v) < 100 or len(v) > 1000: continue
                qsize = math.ceil(ratio * len(v))

                np.random.shuffle(v)
                query[eType][prop] = v[:qsize]
                data[eType][prop] = v[qsize:]

        allTypes = list(allColumns.keys())
        ind = np.arange(len(allColumns))
        np.random.shuffle(ind)

        samplesize = 0.1
        numSamples = np.arange(1, 11)
        total = [0 for x in range(len(numSamples))]
        correct = [0 for x in range(len(numSamples))]
        times = [0 for x in range(len(numSamples))]

        m = 0
        for i in ind[:200]:
            print(m)
            m += 1
            t = allTypes[i]
            if len(query[t]) == 1 or len(query[t]) == 0: continue

            columns = list(query[t].items())
            np.random.shuffle(columns)
            q = columns[0]

            label, values = q[0], q[1]
            np.random.shuffle(values)
            for k, s in enumerate(numSamples):
                s = max(math.ceil(len(q[1]) * samplesize * s), len(q[1]))

                test = q[1][:s]

                scores = []
                for pr in data[t]:
                    try:
                        start = time.time()
                        cost = self.computeCost(test, data[t][pr])
                        end = time.time()
                        scores.append([pr, cost])
                        # times[k] += (end - start)
                        # total[k] += 1
                    except:
                        continue
                    scores = sorted(scores, key=lambda x: x[1])
                    prediction = scores[0][0]

                    if prediction == label:
                        correct[k] += 1
                    total[k] += 1

        print("k \t correct \t total \t precision")
        for k in range(len(numSamples)):
            print(numSamples[k], correct[k], total[k], correct[k] / total[k])

    # test (1) |q| > |c|
    def testReduction2(self):
        ratio = 0.6
        with open('reformed.pkl', 'rb') as f:
            allColumns = pickle.load(f)

        query = {}
        data = {}
        for eType in allColumns:
            query[eType] = {}
            data[eType] = {}
            for prop in allColumns[eType]:
                v = allColumns[eType][prop]
                if len(v) > 1000 or len(v) < 10: continue
                np.random.shuffle(v)

                qsize = int(len(v) * ratio)
                query[eType][prop] = v[:qsize]
                data[eType][prop] = v[qsize:]

        allTypes = list(query.keys())
        np.random.shuffle(allTypes)
        offsets = [1.0, 1.5, 2.0, 2.5, 'unlimited']
        corrects = [0 for x in offsets]
        total = [0 for x in offsets]
        m = 0
        for eType in allTypes[:500]:
            m += 1
            print(m)
            for i, offset in enumerate(offsets):
                if len(query[eType]) == 1 or len(query[eType]) == 0: continue

                columns = list(query[eType].items())
                np.random.shuffle(columns)
                q = columns[0]
                label, values = q[0], q[1]
                scores = []

                if offset == 'unlimited':
                    for pr in data[eType]:
                        cost = 0
                        for num in values:
                            cost += np.min(np.abs(np.array(data[eType][pr]) - num))
                        scores.append([pr, cost])

                else:
                    for pr in data[eType]:
                        baseReplace = math.ceil(len(q[1]) / len(data[eType][pr]) * offset)
                        # baseReplace = min(2 * baseReplace, baseReplace + offset)
                        datacolumn = []
                        for x in range(baseReplace):
                            datacolumn.extend(data[eType][pr])
                        # print(query[key], datacolumn)
                        cost = self.computeCost(values, datacolumn)
                        scores.append([pr, cost])

                scores = sorted(scores, key=lambda x: x[1])
                if label == scores[0][0]:
                    corrects[i] += 1

                total[i] += 1

        print("offset, corrects, total, precision")
        for i in range(len(offsets)):
            print(offsets[i], corrects[i], total[i], corrects[i] / total[i])

    def testReduction3(self):
        '''
        varying the length of subset chose, and plot against precision, running time
        '''

        with open('reformed.pkl', 'rb') as f:
            allColumns = pickle.load(f)

        query = {}
        data = {}
        ratio = 0.4
        for eType in allColumns:
            query[eType] = {}
            data[eType] = {}
            for prop in allColumns[eType]:
                v = allColumns[eType][prop]
                if len(v) < 50: continue
                qsize = math.ceil(ratio * len(v))

                np.random.shuffle(v)
                query[eType][prop] = v[:qsize]
                data[eType][prop] = v[qsize:]

        allTypes = list(allColumns.keys())
        ind = np.arange(len(allColumns))
        np.random.shuffle(ind)

        sizes = [60]
        sizes = sizes[::-1]
        ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
        total = [0 for x in range(len(sizes))]
        correct = [0 for x in range(len(sizes))]
        times = [0 for x in range(len(sizes))]
        np.random.seed(10000)
        print(len(ind))
        m = 0
        for i in ind[:2000]:
            print(m)
            m += 1
            t = allTypes[i]
            if len(query[t]) == 1 or len(query[t]) == 0: continue

            columns = list(query[t].items())
            np.random.shuffle(columns)
            q = columns[0]

            label, values = q[0], q[1]
            np.random.shuffle(values)
            skip = 0
            for k, s in enumerate(sizes):
                # for k, s in enumerate(ratios):
                # s = min(100, math.ceil(s * len(q[1])))
                # if s > 100: continue
                if s <= len(q[1]):
                    test = q[1][:s]

                    scores = []
                    for pr in data[t]:
                        try:
                            start = time.time()
                            cost = self.computeCost(test, data[t][pr])
                            end = time.time()
                            scores.append([pr, cost])
                            times[k] += (end - start)
                            total[k] += 1
                        except:
                            skip = 1
                            break
                    if skip == 1:
                        skip = 0
                        continue
                    scores = sorted(scores, key=lambda x: x[1])
                    prediction = scores[0][0]

                    if prediction == label:
                        correct[k] += 1
                    # total[k] += 1
        print(times, total)
        for i in range(len(times)):
            print(times[i] / total[i])
        # print("k \t correct \t total \t precision")
        # for k in range(len(sizes)):
        #    print(sizes[k], correct[k], total[k], correct[k]/total[k])
        # print (times[k]/total[k], end=',')

    def testefficiency(self):
        self.connect()

        with open('allColumns.pickle', 'rb') as f:
            poplist = []
            dataset = pickle.load(f)
            for key in dataset:
                dataset[key] = np.array(dataset[key])
                if len(dataset[key]) == 1 or len(dataset[key]) > 1000:
                    poplist.append(key)

        for p in poplist:
            dataset.pop(p)

        allcolumns = dataset.items()

        with open(self.wikitable, 'r') as tbf:
            line = tbf.readline().strip()
            for i in range(10):
                pruned1 = 0
                pruned2 = 0
                table = json.loads(line)

                headers = table['header']
                values = table['values']
                entities = table['entity']
                unit = table['unit']

                values = ';'.join(values).replace(',', '').split(';')
                values = list(map(float, values))

                queue, topK = self.createOrder(values, allcolumns)
                print('total columns: ', len(queue))

                while len(queue) != 0:
                    data = heapq.heappop(queue)
                    datacolumn = dataset[data[1]]
                    if self.pruneUp(values, datacolumn, topK):
                        pruned1 += 1

                    elif self.pruneDown(values, datacolumn, topK):
                        pruned2 += 1

                    else:
                        cost = self.computeCost(values, datacolumn)
                        # update top k
                        if cost < topK[-1][0]:
                            topK[-1] = (cost, data[1],)
                            topK.sort(key=lambda x: x[0])

                line = tbf.readline().strip()

                print(headers)
                print('columns pruned by range:', pruned1)
                print('columns pruned by mapping:', pruned2)
                for t in topK:
                    print('predicted label: ', self.resolveTriple(t[1]))
                print('------------------------------------')

    def remove_outliers(self, datacolumn):
        zscores = zscore(datacolumn)
        new_data = []
        for i in range(len(zscores)):
            if 3 > zscores[i] > -3:
                new_data.append(datacolumn[i])
        return new_data

    def update_topk(self, topK, type_property_unit, cost):
        for i in range(len(topK)):
            if topK[i][0] == np.inf:
                break
            if topK[i][1][0] == type_property_unit[0] and topK[i][1][1] == type_property_unit[1]:
                if topK[i][0] > cost:
                    topK[i] = (cost, type_property_unit, )
                    topK.sort(key=lambda x: x[0])
                return topK
        topK[-1] = (cost, type_property_unit,)
        topK.sort(key=lambda x: x[0])
        return topK


    def predict(self, eType, values, distributions, dtype='int'):
        # retrieve labels by mapping function
        # set the threshold for min-cost
        thres = abs(0.3 * sum(values))
        isPattern = False
        # check pre-defined patterns
        if dtype == 'int':
            (isPattern, l) = self.patternMatching(values)

        # TODO: REMOVE PATTERN MATCHING FOR TESTING PURPOSE
        isPattern = False

        allcolumns = distributions.items()

        # pruning
        queue, topK = self.createOrder(values, allcolumns)
        numCols = len(queue)

        pruned1, pruned2 = 0, 0
        while len(queue) != 0:
            data = heapq.heappop(queue)
            # new code
            datacolumn = distributions[data[1]]
            currentMax = topK[-1][0]
            # new code
            datacolumn = self.remove_outliers(datacolumn)
            est = self.pruneUp(values, datacolumn, topK)
            if est > currentMax or est == -1:
                # new code
                pruned1 += 1

            elif self.pruneDown(values, datacolumn, topK) > currentMax:
                # new code
                pruned2 += 1

            else:
                try:
                    if len(values) < 51:
                        # NEW CODE
                        if len(values) > len(datacolumn):
                            np.random.shuffle(values)
                            cost = self.computeCost(values[:len(datacolumn)], datacolumn)
                            cost = cost * len(values) / len(datacolumn)
                        else:
                            cost = self.computeCost(values, datacolumn)
                    else:
                        sumcost = 0
                        # numsubsets = int(math.log(len(values)/50)) + 1
                        numsubsets = 1
                        for s in range(numsubsets):
                            np.random.shuffle(values)
                            # NEW CODE
                            if len(values) > len(datacolumn):
                                np.random.shuffle(values)
                                cost = self.computeCost(values[:min(len(datacolumn), 50)], datacolumn)
                                cost = cost * 50 / len(datacolumn)
                            else:
                                cost = self.computeCost(values[:50], datacolumn)
                            sumcost += cost
                        avgcost = sumcost / numsubsets
                        cost = avgcost  # * len(values)/50

                    # if cost > thres: continue

                except TimeoutError:
                    return None
                # update top k
                # new code
                if cost < topK[-1][0]:
                    # topK = self.update_topk(topK, data[1], cost)
                    topK[-1] = (cost, data[1],)
                    topK.sort(key=lambda x: x[0])

        # no matched columns within the given threshold. reject this column
        # if len(topK) == 0:
        #    return None

        predictions = []
        # for x in topK:
        #     predictions.append(x)
        for x in topK:
            # COMMENT
            result = self.resolveTriple(x[1])
            # COMMENT END
            # result = x[1]
            if result == None:
                continue
            elif len(result) < 3:
                continue

            predictions.append(list((result, x[0])))

        # if a pattern is detected, add label to return list
        if isPattern:
            if len(predictions) != 0:
                predictions[-1] = ('t', l, None)
            else:
                predictions.append(('t', l, None))

        return predictions

    def ksdistance(self, eType, values, distributions, dtype='int'):

        allcolumns = distributions.items()
        scores = []

        for c in allcolumns:
            if len(c[1]) == 0:
                scores.append([c[0], -1])
                continue
            statistics, pval = ks_2samp(values, c[1])
            scores.append([c[0], pval])

        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:min(self.k, len(scores))]
        predictions = []
        for x in scores:
            result = self.resolveTriple(x[0])
            if result == None:
                continue
            elif len(result) < 3:
                continue

            result = list(result)
            if result[2] == None:
                result[2] == ' '

            predictions.append(result)

        return predictions

    def find_in_dbpedia(self, eType):
        rv = {}
        query = '''SELECT * FROM distribution WHERE type LIKE ?; '''
        self.cursor.execute(query, ("%" + eType + "%",))
        row = self.cursor.fetchone()
        while row != None:
            if (row[0], row[1], row[2]) not in rv:
                rv[(row[0], row[1], row[2])] = []
            rv[(row[0], row[1], row[2])].append(row[3])
            row = self.cursor.fetchone()
        return rv

    def annotate(self):
        self.connect()
        signal.signal(signal.SIGALRM, TimeoutHandler)

        total = 0
        correctUnit = 0
        correctLabel = 0
        unable = 0

        # outfile = open('WTResult.txt','w')
        short = open('result_short.txt', 'w')
        # ent = open('typeDetectWDC.txt', 'w')
        eTypePred = 0
        wikitable = "dataset/wiki_annotated.txt"
        wdc = "dataset/wdc_annotated.txt"
        tdv = "dataset/T2Dv2_v2.txt"
        tdv_v4 = "dataset/T2Dv2_v4.txt"
        with open('./typeHierarchy.pickle', 'rb') as f:
            hierarchy_graph = pickle.load(f)
        with open(wikitable, 'r') as tbf:
            i = -1
            # for i in range(110): line = tbf.readline().strip()
            line = tbf.readline().strip()
            x = 0
            try:
                while line != '':
                    print(correctLabel, correctUnit, total)
                    i += 1
                    table = json.loads(line)

                    headers = table['header']
                    values = table['values']
                    entities = table['entity']

                    unit = table['unit']
                    semantic = table['semantic'].lower()
                    # year is usually not in kb
                    if semantic == 'year':
                        line = tbf.readline().strip()
                        continue
                    eType = table['eType']

                    try:
                        eType = self.resolve(table['eType'], reverse=True)
                        if len(eType) == 0:
                            line = tbf.readline().strip()
                            continue
                    except:
                        eType = self.getEntityType(entities)
                        # cannot find matched type
                        if eType == None:
                            line = tbf.readline().strip()
                            continue

                        types = [self.resolve(t) for t in eType]
                        # ent.write("header: " + headers[0] + '\n sample entity: ' + entities[0] + '\n')
                        # ent.write("predicted type: " + str(types) + '\n--------------\n\n')
                        # ent.flush()
                        # line = tbf.readline().strip()
                        # continue
                    '''
                    eType = self.getEntityType(entities)
                    # cannot find matched type
                    if eType == None:
                        line = tbf.readline().strip()
                        continue
                    '''

                    ### test entity type prediction
                    '''
                    eType = table['eType'].lower()
                    predType = self.getEntityType(entities)
                    # cannot find matched type
                    if eType == None and predType == None:
                        line = tbf.readline().strip()
                        continue
                    total += 1
                    types = [self.resolve(t).lower() for t in predType]
                    if eType in types:
                        eTypePred += 1
                    print (eType, types)
                    print (eTypePred, total)
                    line = tbf.readline().strip()
                    continue
                    '''
                    # print(eType, values, i)
                    eType = self.get_parents_to_the_root(hierarchy_graph, eType)
                    distributions = self.getDistribution(eType)
                    if len(distributions) == 0:
                        line = tbf.readline().strip()
                        continue
                    dtype = 'float'
                    # values = ';'.join(values).replace(',', '').split(';')
                    # NEW CODE
                    values = [v.replace(',', '') for v in values]
                    try:
                        values = list(map(int, values))
                        dtype = 'int'
                    except ValueError:
                        try:
                            values = list(map(float, values))
                        except ValueError:
                            t = []
                            for x in values:
                                try:
                                    t.append(float(x))
                                except ValueError:
                                    pass
                            if len(t) < len(values) / 2:
                                line = tbf.readline().strip()
                                continue
                            values = t
                    # the largest value is the 'total' of all other values
                    v = sorted(values, reverse=True)
                    if sum(values[1:]) == v[0]: values.remove(v[0])

                    predictions = self.predict(eType, values, distributions, dtype)
                    print("semantic: ", semantic)
                    print("prediction: ", predictions)
                    # line = tbf.readline().strip()
                    # continue
                    # predictions = self.ksdistance(eType, values, distributions, dtype)

                    if predictions == None:
                        unable += 1
                        line = tbf.readline().strip()
                        continue
                    elif len(predictions) == 0:
                        unable += 1
                        line = tbf.readline().strip()
                        continue

                    p_semantic = [x[0][1] for x in predictions]
                    p_unit = [x[0][2] for x in predictions]

                    # check correctness
                    total += 1
                    for p_sem in p_semantic:
                        if p_sem and (semantic in p_sem or p_sem in semantic):
                            correctLabel += 1
                            break
                        elif p_sem and 'number of' in p_sem and 'number of' in semantic:
                            correctLabel += 1
                            break

                    if unit in p_unit: correctUnit += 1

                    print(i, headers, unit, file=short)
                    print('labels given by mapping:', file=short)
                    for x in predictions: print(x, file=short)
                    print('\n -------------------------', file=short)

                    # short.flush()

                    line = tbf.readline().strip()
                print('k=', self.k, correctLabel, correctUnit, total)

            except EOFError:
                pass

        # outfile.close()
        short.close()
        # ent.close()

        self.clean()

    def read_values(self, file_name):
        values = []
        with open(file_name, 'r') as tbf:
            line = tbf.readline().strip()
            try:
                while line != '':
                    table = json.loads(line)
                    value = table['values']
                    values.append(value)
                    line = tbf.readline().strip()
            except EOFError:
                pass
        return values

    def annotate_dbpedia(self):
        self.connect()
        signal.signal(signal.SIGALRM, TimeoutHandler)

        total = 0
        correctUnit = 0
        correctLabel = 0
        unable = 0

        # outfile = open('WTResult.txt','w')
        short = open('result_short.txt', 'w')
        # ent = open('typeDetectWDC.txt', 'w')
        eTypePred = 0
        wikitable = "dataset/wiki_annotated_dbpedia4.txt"
        wdc = "dataset/wdc_dbpedia3.txt"
        tdv = "dataset/T2Dv2_v3.txt"
        tdv_v4 = "dataset/T2Dv2_v4.txt"
        tdv_v4_values = self.read_values(tdv_v4)
        with open(wdc, 'r') as tbf:
            i = -1
            # for i in range(110): line = tbf.readline().strip()
            line = tbf.readline().strip()
            skip_counter = 0
            tedad = 0
            try:
                while line != '':
                    print(correctLabel, correctUnit, total)
                    i += 1
                    table = json.loads(line)

                    headers = table['header']
                    values = table['values']
                    # if values in tdv_v4_values:
                    #     line = tbf.readline().strip()
                    #     skip_counter += 1
                    #     print("skip: ", str(skip_counter))
                    #     continue
                    entities = table['entity']

                    unit = table['unit']
                    semantic = table['semantic'].lower()
                    # year is usually not in kb
                    if semantic == 'year':
                        line = tbf.readline().strip()
                        continue
                    eType = table['eType']
                    distributions = get_features_and_kinds_multi_thread(eType)
                    if len(distributions) == 0:
                        line = tbf.readline().strip()
                        continue
                    dtype = 'float'
                    # values = ';'.join(values).replace(',', '').split(';')
                    # NEW CODE
                    values = [v.replace(',', '') for v in values]
                    try:
                        values = list(map(int, values))
                        dtype = 'int'
                    except ValueError:
                        try:
                            values = list(map(float, values))
                        except ValueError:
                            t = []
                            for x in values:
                                try:
                                    t.append(float(x))
                                except ValueError:
                                    pass
                            if len(t) < len(values) / 2:
                                line = tbf.readline().strip()
                                continue
                            values = t
                    # the largest value is the 'total' of all other values
                    v = sorted(values, reverse=True)
                    if sum(values[1:]) == v[0]: values.remove(v[0])

                    print("total: ", str(total))
                    predictions = self.predict(eType, values, distributions, dtype)
                    tedad += 1
                    print("semantic: ", semantic)
                    print("prediction: ", predictions)
                    # line = tbf.readline().strip()
                    # continue
                    # predictions = self.ksdistance(eType, values, distributions, dtype)

                    if predictions == None:
                        unable += 1
                        line = tbf.readline().strip()
                        continue
                    elif len(predictions) == 0:
                        unable += 1
                        line = tbf.readline().strip()
                        continue

                    p_semantic = [x[0][1].split('/')[-1] for x in predictions]
                    p_unit = [x[0][2] for x in predictions]

                    # check correctness
                    total += 1
                    for p_sem in p_semantic:
                        if p_sem and (semantic in p_sem or p_sem in semantic):
                            correctLabel += 1
                            break
                        elif p_sem and 'number of' in p_sem and 'number of' in semantic:
                            correctLabel += 1
                            break

                    if unit in p_unit: correctUnit += 1

                    print(i, headers, unit, file=short)
                    print('labels given by mapping:', file=short)
                    for x in predictions: print(x, file=short)
                    print('\n -------------------------', file=short)

                    # short.flush()

                    line = tbf.readline().strip()
                print('k=', self.k, correctLabel, correctUnit, total)

            except EOFError:
                pass

        # outfile.close()
        short.close()
        # ent.close()

        self.clean()

    def verify(self, semanticLabel, unit, labels):
        resolveQuery = '''select label from mapping where wid = ?;'''
        propertyMatch = 0
        unitMatch = 0

        for l in labels:
            prop = l[1]
            unit = l[2]

            try:
                self.cursor.execute(resolveQuery, (prop,))
                stringRepr = self.cursor.fetchone()[0]
                print(stringRepr, semanticLabel)
                if stringRepr.lower() == semanticLabel[1].lower():
                    propertyMatch = 1

                self.cursor.execute(resolveQuery, (unit,))
                stringRepr = self.cursor.fetchone()[0]
                print(stringRepr)
                if stringRepr.lower() == unit.lower():
                    unitMatch = 1
            except TypeError:
                continue

        return propertyMatch, unitMatch

    def computeCost(self, arr1, arr2, dist='abs'):
        # arr1 always have smaller size
        if len(arr1) > len(arr2): raise ValueError('size of query column is greater than data column')

        # the algorithm may not work with float weights
        # if values are float, keep 4 decimal places
        try:
            arr1 = list(map(int, arr1))
            arr2 = list(map(int, arr2))
            factor = 1
        except ValueError:
            factor = 10000
            arr1 = [int(factor * x) for x in arr1]
            arr2 = [int(factor * x) for x in arr2]

        # time limit is 10 min
        signal.alarm(300)

        G = nx.DiGraph()
        G.add_node('s', demand=-1 * len(arr1))
        G.add_node('t', demand=len(arr1))
        for i in range(len(arr1)):
            # connect source node to all query nodes
            G.add_edge('s', i, capacity=1, weight=0)
            for j in range(len(arr2)):
                nodenum = j + len(arr1)
                # may have different distance functions
                distance = abs(arr2[j] - arr1[i])

                G.add_edge(i, nodenum, capacity=1, weight=distance)

        # connect all data nodes to sink. cost of edges are 0
        for j in range(len(arr2)):
            nodenum = j + len(arr1)
            G.add_edge(nodenum, 't', capacity=1, weight=0)

        # compute flow with min cost
        # mincostFlow = nx.max_flow_min_cost(G, 's', 't')
        mincostFlow = nx.min_cost_flow(G)
        mincost = nx.cost_of_flow(G, mincostFlow)

        # reset timer
        signal.alarm(0)

        return mincost / factor
        # return mincost/factor, mincostFlow

    def matchDistribution(self, column, distributions, method='mapping'):
        # top k nearest match
        k = 3
        costs = {}

        if method == 'mapping':
            # method using mapping function
            # allcolumns = distributions.items()
            for key in distributions:
                dataColumn = distributions[key]

                try:
                    cost = self.computeCost(column, dataColumn)
                    costs[key] = cost
                except TimeoutError:
                    continue
                except ValueError:
                    continue

            topMatch = sorted(costs.items(), key=lambda x: x[1])
            labels = [x[0] for x in topMatch[:k]]

        elif method == 'meandist':
            m = np.mean(column)
            r = np.std(column)

            for key in distributions:
                values = distributions[key]
                m2 = np.mean(values)
                diff = np.abs(m2 - m)

                if diff < r:
                    costs[key] = diff
            topMatch = sorted(costs.items(), key=lambda x: x[1])
            labels = [x[0] for x in topMatch[:k]]

        '''
        # method using statistical test
        labels = []
        alpha = 0.05
        for key in distributions:
            values = distributions[key]
            (statistic, pvalue) = mannwhitneyu(column, values, alternative="two-sided")
        
            if pvalue > alpha:
                labels.append(key)
        '''

        return labels

    def patternMatching(self, distribution):
        # check if column matches any pre-defined pattern. number, year, etc

        # detect rank
        distribution = sorted(distribution)
        rank = 1
        if distribution[0] == 0 or distribution[0] == 1:
            for i in range(1, len(distribution)):
                if int(distribution[i] - distribution[i - 1]) != 1:
                    rank = 0
                    break
        else:
            rank = 0

        if rank: return (True, 'rank/number',)

        # detect year
        year = 1
        for num in distribution:
            if num < 1700 or num > 2050:
                year = 0
                break

        if year: return (True, 'year',)

        # other patterns
        return (False, None)

    def getDistribution(self, eType):
        query = '''select amount from distribution where type = ? and property = ? and unit = ?; '''
        getRelationQuery = '''select type, property, unit from relation where type = ?; '''

        rv = {}
        for t in eType:
            if type(t) == str:
                self.cursor.execute(getRelationQuery, (t,))
                relations = self.cursor.fetchall()

                for relation in relations:
                    self.cursor.execute(query, relation)
                    d = []
                    row = self.cursor.fetchone()
                    while row != None:
                        d.append(row[0])
                        row = self.cursor.fetchone()

                    rv[relation] = d

            else:
                self.cursor.execute(query, t)
                d = []
                row = self.cursor.fetchone()
                while row != None:
                    if len(row[0]) != 0:
                        d.append(row[0])
                    row = self.cursor.fetchone()
                rv[t] = d

        return rv

    def getEntityType(self, entities):
        '''
        return value: 
            a set containing matched entity types
            The types are represented by wikiID.
        '''

        getWikiId = '''select wid from mapping where label = ?; '''
        getType = '''select types from type where wid = ?; '''

        types = {}
        final = set()

        # thres = math.floor(len(entities) *0.5)
        # thres = math.floor(len(entities) * 0.6)
        thres = math.floor(len(entities) * 0.75)
        # thres = math.floor(len(entities) * 0.9)

        allTypes = []

        for entity in entities:
            self.cursor.execute(getWikiId, (entity,))
            wids = [x[0] for x in self.cursor.fetchall()]

            # no matching wikidata IDs with the given entity
            if wids == None or len(wids) == 0: continue

            for wid in wids:
                self.cursor.execute(getType, (wid,))
                entityType = self.cursor.fetchone()
                if entityType == None: continue

                entityType = entityType[0].split(',')

                for t in entityType:
                    if t in types:
                        types[t] += 1
                    else:
                        types[t] = 1

        if len(types) == 0:
            return None

        l = sorted(types.items(), key=lambda x: x[1], reverse=True)
        maximum = l[0][1]

        # not enough entities agree on the same type
        if maximum < thres: return None

        for t in l:
            # if t[1] == maximum:
            if t[1] >= thres:
                final.add(t[0])

        return final


class Table():
    def __init__(self):
        self.col = []
        self.header = []
        self.searchQuery = 'https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&search=%s&language=en'

        self.conn = None
        self.cursor = None
        self.dbname = 'relation.db'
        self.wikitable = 'wikiTables.txt'

    def processTable(self, dataset="wikiTable"):
        p = re.compile('[a]')

        noheader = 0
        total = 0

        if dataset == 'wikiTable':
            j = 0
            with open(self.wikitable, 'w') as f:
                # for k in range(400000):
                #    line = input().strip()

                line = input().strip()

                while True:
                    # for k in range(400000):
                    # extract entity column and numeric column
                    try:
                        line = input().strip()
                        tb = json.loads(line)
                        total += 1
                    except EOFError:
                        break
                    except:
                        continue

                    index_of_columns = tb['numericColumns']

                    # no numeric columns
                    if len(index_of_columns) == 0:
                        continue

                    headers = tb['tableHeaders'][0]
                    body = tb['tableData']

                    # extract text from all cells
                    columns = []
                    numColumns = len(headers)
                    for i in range(numColumns):
                        columns.append([])

                    for rowid, rows in enumerate(body):
                        for cellNum, cell in enumerate(rows):
                            # if cell['isNumeric']:
                            index = cellNum % numColumns
                            if len(cell['text']) > 0:
                                columns[index].append(cell['text'])

                    # unique columns should have ratio = 1
                    entity = None
                    numeric = []
                    header = []
                    for i, col in enumerate(columns):
                        if len(col) == 0: continue

                        uniqueness = len(set(col)) / len(col)
                        # the left most, non numeric and most unique column
                        if (uniqueness > 0.9) and (i not in index_of_columns):
                            entity = col
                            header.append(headers[i]['text'])
                            break
                    if entity == None or entity == []: continue

                    for i in index_of_columns:
                        # print (i, len(columns), len(headers))
                        if len(columns[i]) == 0: continue
                        numeric.append(columns[i])
                        header.append(headers[i]['text'])

                    if len(header) == 1 or "" in header:
                        noheader += 1

                    # we want each table to have binary relation: one entity column and one numeric column
                    for i, col in enumerate(numeric):
                        h = [header[0], header[i + 1]]
                        table = {}
                        table['entity'] = entity
                        table['values'] = col
                        table['header'] = h

                        l = json.dumps(table)
                        f.write(l + '\n')
            print(noheader, total)

        elif dataset == "wdc":
            path = "../wdc/00/0/"

            alltbs = os.listdir(path)
            with open('wdc.txt', 'w') as outf:
                for fname in alltbs:
                    tb = None
                    table = {}
                    try:
                        with open(path + fname, 'r') as f:
                            tb = json.load(f)
                    except:
                        continue

                    # skip tables without header or key column
                    if not tb['hasHeader'] or not tb['hasKeyColumn']: continue

                    columns = tb['relation']

                    header = []
                    numeric = []

                    # extract header
                    headeridx = tb['headerRowIndex']
                    allheader = [col[headeridx] for col in columns]

                    # remove columns with empty header (for evaluation purpose only, can still annotate them)
                    if '' in allheader: continue

                    # remove header from columns
                    for i in range(len(columns)):
                        columns[i].pop(headeridx)

                    # extract entity column (key column)
                    keyidx = tb['keyColumnIndex']
                    try:
                        keyColumn = columns[keyidx]
                    except IndexError:
                        continue
                    keyheader = allheader[keyidx]

                    columns.pop(keyidx)
                    allheader.pop(keyidx)

                    # extract numeric columns
                    for i, col in enumerate(columns):
                        try:
                            c = list(map(float, col))
                            numeric.append(c)
                            header.append(allheader[i])
                        except:
                            continue

                    # no numeric columns
                    if len(numeric) == 0: continue

                    for i, col in enumerate(numeric):
                        h = [keyheader, header[i]]
                        table = {}
                        table['entity'] = keyColumn
                        table['values'] = col
                        table['header'] = h

                        l = json.dumps(table)
                        outf.write(l + '\n')

    def verify(self):
        with open(self.wikitable, 'r') as f:
            for i in range(100):
                line = f.readline().strip()
                tb = json.loads(line)
                print(json.dumps(tb, indent=4))
                print('\n\n')

    def jsonToXml(self, inputFile):
        '''
        transform data into acceptable format for KDD 14' method
        '''

        i = 0
        output = '/home/think2/Documents/dataset.xml'
        fout = open(output, 'w')
        fout.write('''<?xml version="1.0" encoding="UTF-8"?>\n<root>''')
        with open(inputFile, 'r') as f:
            line = f.readline().strip()
            while i < 180:  # line != '':
                print(line, i)
                js = json.loads(line)
                firstContent = js['values'][0]
                header = js['header'][1]
                u = js['unit']

                h = ''
                for ch in header:
                    if ch == '(' or ch == ')':
                        continue
                    elif ch.isalpha() or ch.isdigit() or ch == ' ':
                        h += ch
                    else:
                        h += ' ' + ch + ' '

                ht = str(h.split())

                xmlString = "<r>" + \
                            "<c>" + str(firstContent) + "</c>\n" + \
                            "<h>" + h + "</h>\n" + \
                            "<ht>" + ht + "</ht>\n" + \
                            "<u>" + u + "</u>\n" + \
                            "</r>"

                fout.write(xmlString + '\n')
                line = f.readline().strip()
                i += 1

        fout.write('</root>')
        fout.close()


def organizeColumns():
    with open('allColumns.pickle', 'rb') as f:
        allColumns = pickle.load(f)

    new = {}
    for key in allColumns:
        eType = key[0]
        prop = key[1]
        unit = key[2]

        v = allColumns[key]
        if eType in new:
            if prop not in new[eType]:
                new[eType][prop] = v

        else:
            new[eType] = {}
            new[eType][prop] = v

    with open('reformed.pkl', 'wb') as f:
        pickle.dump(new, f)


if __name__ == '__main__':
    # k = 5 55 27 133
    # k = 1 28 14 134
    # k = 1 11 x 67 td
    # t = Table()
    # t.processTable()
    # t.processTable(dataset="wdc")
    # t.verify()
    a = Annotation()
    start = time.time()
    a.annotate()
    end = time.time()
    print(end - start)
    # a.annotate_dbpedia()
    # a.testefficiency()
    # a.testReduction()
    # a.testReduction2()
    # a.testReduction3()
    # t.jsonToXml("dataset/unitlabels.txt")

    '''
    arr1 = [234,456, 23,135, 756, 324,21,3]
    arr2 = [np.random.randint(1000) for i in range(100)]

    cost, flow = a.computeCost(arr1, arr2)
    mset = set()
    for qnode in flow:
        if qnode == 's' or qnode == 't':
            continue
        
        for dnode in flow[qnode]:
            if flow[qnode][dnode] == 1:
                dnodeIdx = dnode - len(arr1)
                mset.add(arr2[dnodeIdx])
    
    for value in mset:
        arr2.remove(value)
    '''