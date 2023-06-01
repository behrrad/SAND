import os, re, json, math, pickle
import requests, sqlite3
import networkx as nx

nonletter = re.compile('[^a-zA-Z ]')

synonyms = {}
relation = {}


class RelationGraph():
    def __init__(self):
        # url for searching entity by name
        self.findQidQuery = 'https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&search=%s&language=en'
        # url for getting entity by qid
        self.getEntityQuery = 'https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&ids=%s&languages=en'

        self.graph = nx.Graph()

        self.relations = set()
        self.idMapping = {}
        self.reverseMapping = {}

        self.synonyms = {}
        self.propertyDistribution = {}
        self.distribution = []
        self.rid = 0

        self.dbname = 'relation.db'
        self.conn = None
        self.cursor = None

        self.symbols = []

    def extractFromJson(self):
        '''
        reads Wikidata json dump from stdin.

        run with command: bunzip2 -c [path_of_dump] | python3 [path_of_python_script] 

        '''

        mainPattern = re.compile('"property":[^\[\]]+?"type":"quantity"}')
        timePattern = re.compile('"property":[^\[\]]+?"type":"time"}')

        line = input()
        line = input().strip()[:-1]
        while line != '':
            # this entity doesn't have numeric fields
            if ('"type":"qualtity"' not in line) and ('"type":"time"' not in line):
                line = input().strip()[:-1]
                continue

            entity = json.loads(line)
            # properties are in entity[qid]['claims']
            properties = entity['claims']

            try:
                # 'P31' is wikiID for 'is instance of'
                entityType = properties['P31']
                entityType = [item['mainsnak']['datavalue']['value']['id'] for item in entityType]
            except:
                line = input().strip()[:-1]
                continue

            # find all properties with numerical values
            substrings = mainPattern.findall(line)
            times = timePattern.findall(line)

            substrings.extend(times)
            for prop in substrings:
                prop = '{"main":{' + prop + '}' + '}'
                try:
                    prop = json.loads(prop)
                except:
                    continue
                               
                pid = prop['main']['property']
                
                try:
                    amount = prop['main']['datavalue']['value']['amount']
                    if ('.' in amount):
                        amount = float(amount)
                    else:
                        amount = int(amount)
                        
                    unit = prop['main']['datavalue']['value']['unit']
                    if (unit == '1'):
                        unit = '1'
                    else:
                        unit = unit[unit.rfind('/')+1:]
                except KeyError:
                    value = prop['main']['datavalue']['value']['time'].split('T')
                    amount = value[0].replace('+', '')
                    unit = 't'

                self.idMapping[pid] = ''

                for t in entityType:
                    self.relations.add((t, pid, unit))

                # keep a unique distribution for each (entity type, property) pair
                # ** note: a value with unit of datetime will not be added to distribution **

                if unit != 't':
                    for t in entityType:
                        self.distribution.append((t, pid, unit, amount,))
                        self.rid += 1
                else:
                    for t in entityType:
                        key = (t, pid)
                        self.propertyDistribution[key] = 't'

                #print(pid, entityType, unit)          
            #print(len(self.relations))
            if (len(self.relations) >= 30000):
                self.pushToDB('relation')

            if (len(self.distribution) >= 30000):
                self.pushToDB('distribution')

            line = input().strip()[:-1]
        
        print (self.rid)

    def pushToDB(self, flag='relation'):
        self.connect()

        self.insertRelation = '''INSERT INTO relation(type, property, unit) VALUES (?,?,?); '''
        self.insertSynonym = '''INSERT INTO aliases(canonical, alias) VALUES (?,?); '''
        self.insertMapping = '''INSERT INTO mapping(wikiID, name) VALUES (?,?);'''
        self.insertDistribution = '''INSERT INTO distribution(type, property, unit, amount) VALUES (?,?,?,?); '''
        
        if (flag == 'relation' or flag == 'all'):
            if (len(self.relations) != 0):
                self.cursor.execute('BEGIN TRANSACTION;')
                for row in self.relations:
                    try:
                        self.cursor.execute(self.insertRelation, row)
                    except sqlite3.IntegrityError:
                        continue

                self.cursor.execute('END TRANSACTION;')
                self.relations = set()

        if (flag == 'distribution' or flag == 'all'):
            if (len(self.distribution) != 0):
                self.cursor.execute('BEGIN TRANSACTION;')
                for row in self.distribution:
                    try:
                        self.cursor.execute(self.insertDistribution, row)
                    except sqlite3.IntegrityError:
                        continue
                    except OverflowError:
                        continue

                self.cursor.execute('END TRANSACTION;')
                self.distribution = []

        if (flag == 'synonym' or flag == 'all'):
            if (len(self.synonyms) != 0):
                self.cursor.executemany(self.insertSynonym, self.synonyms)
                self.synonyms = []

        if (flag == 'mapping'):
            query = '''INSERT INTO mapping(wid, label) VALUES (?,?);'''
            self.cursor.executemany(query, self.symbols)

            self.symbols = []
                
        self.conn.commit()
    

    def connect(self):
        if self.conn == None:
            self.conn = sqlite3.connect('./' + self.dbname)
            self.cursor = self.conn.cursor()

    def clean(self):
        self.cursor = None
        if (self.conn != None): self.conn.close()
        self.conn = None


    # no longer used
    def getTypeRelation(self, id_num=1):
        '''
        This function goes through entities in wikidata,
        and identifies (entity_type -- property -- unit) relation
        
        parameters: id_num [optional], specifies the starting qid
        
        return values: 
        
        Error: 
        '''
    
        # go through all entities in wikidata
        # start with Q1 
        while True:
            qid = 'Q' + str(id_num)
            url = self.getEntityQuery % qid
            
            res = requests.get(url)
            entity = res.json()

            # extract the 'is instance of' property and use as entity type
            # entity_type contain a set of types
            try:
                entity_type = self.__getEntityType__(entity['entities'][qid]['claims']['P31']) 
            except KeyError:
                id_num += 1
                continue      
            #print ("entity type:", entity_type)
            
            propUnitRelation = self.getProperties(entity, qid)
            if (propUnitRelation == None):
                id_num += 1
                continue
            #print ("property unit relation:", propUnitRelation)

            # TODO: add relations to graph
            print (id_num, end=' ')
            for prop in propUnitRelation:
                print (entity_type[0] + '\t' + str(prop) + '\t' + str(propUnitRelation[prop]))
            if id_num == 300:
                exit()
            id_num += 1

    def getProperties(self, entity, qid):
        '''
        This function implements taking an entity (in json format),
        and returns properties that have numeric fields.
        
        parameter:
            entity: wikidata entity in json format
        
        return value: 
            property: a dictionary contains {property: unit} relations
            
        error:
            return None: if no properties have numerical fields
            
        '''
        
        rv = {}
        #nFields = set({'time', 'quantity'})
        
        allFields = entity['entities'][qid]['claims']
        
        # check all properties for this entity
        for pid in allFields:
            field = allFields[pid]
            # possible attributes that contain numerical values
            if ('qualifiers' in field):
                qualifiers = field['qualifiers']
            else:
                qualifiers = None
            
            # get label for this property
            propertyLabel = self.__getLabelByID__(pid)

            # identify numerical fields. entity without numerical fields are disgarded
            for prop in field:
                mainsnak = prop['mainsnak']
                datatype = set({mainsnak['datatype']})
                #if (qualifiers != None):
                #    for item in qualifiers:
                #        datatype.add(qualifiers[item])
                
                # this property contains numerical field
                if ('quantity' in datatype):
                    try:
                        amount = mainsnak['datavalue']['value']['amount']
                        unitID = mainsnak['datavalue']['value']['unit']
                        # this property has no clear unit, treat as special unit. e.g. age, population, etc
                        if (unitID == '1'):
                            unit = '1'
                        else:
                            ID = unitID[unitID.rfind('/')+1:]
                            unit = self.__getLabelByID__(ID)
                                      
                        # add relation to return value
                        rv[propertyLabel] = unit
                    except KeyError:
                        pass

                    
                #elif ('time' in datatype):
                #    print (mainsnak)
                #    timeString = mainsnak['datavalue']['value']['time'].replace('+', '')
                #    dateString = timeString.split('T')[0]
                
        if len(rv) == 0:
            return None
        
        return rv    
                
            
    def __getLabelByID__(self, ID):
        '''
        This function implements getting the label of an entity given qid/pid.
        The label typically contains the human-readable name of an entity.

        parameter: qid

        return value: label as string
        '''

        url = self.getEntityQuery % ID
            
        entity = requests.get(url).json()

        label = entity['entities'][ID]['labels']['en']['value']

        return label
        
    def __getEntityType__(self, rawType):
        types = set()
        for item in rawType:
            # get qid for type label
            qid = item['mainsnak']['datavalue']['value']['id']
            instance = self.__getLabelByID__(qid)
            types.add(instance)
            
        return list(types)

    def resolve(self):
        '''
        construct mapping between wikidata IDs and labels.
        Will need to scan the Wikidata dump once.
        '''

        self.connect()

        line = input()
        line = input().strip()[:-1]
        while line != '':
            try:
                try:
                    content = json.loads(line)
                    wid = content['id']
                    label = content['labels']['en']['value']
                except KeyError:
                    line = input().strip()[:-1]
                    continue

                except json.decoder.JSONDecodeError:
                    line = input().strip()[:-1]
                    continue

                #print(wid, label)
                self.symbols.append((wid, label,))
                if (len(self.symbols) >= 30000):
                    self.pushToDB(flag='mapping')

                line = input().strip()[:-1]

            except EOFError:
                break

    def extractType(self):
        '''
        For each entity, store the type (property 'P31' in Wikidata) in database.
        '''

        self.connect()

        q = "INSERT INTO type(wid, types) VALUES(?, ?);"
        symbols = []

        line = input()
        line = input().strip()[:-1]
        while True:
            try:
                try:
                    entity = json.loads(line)
                    wid = entity['id']

                    properties = entity['claims']
                    entityType = properties['P31']
                    entityType = [item['mainsnak']['datavalue']['value']['id'] for item in entityType]
                    typeString = ','.join(entityType)
                except:
                    line = input().strip()[:-1]
                    continue


                #print(wid, typeString, len(symbols))
                symbols.append((wid, typeString,))
                if (len(symbols) >= 30000):
                    self.cursor.executemany(q, symbols)
                    self.conn.commit()
                    symbols = []

                line = input().strip()[:-1]

            except EOFError:
                break
        
        self.clean()

    def typeHierarchy(self):
        '''
        Construct a hierarchy sturcture for all types in wikidata.
        The structure is represented by a directed graph.

        edge direction: supertype --> subtype
        '''

        if os.path.exists('typeHierarchy.pickle'): return

        G = nx.DiGraph()
        allTypes = set()

        line = input()
        while True:
            try:
                line = input().strip()[:-1]
            except EOFError:
                break

            try:
                entity = json.loads(line)
                wid = entity['id']

                properties = entity['claims']
                # this entity is a type
                if 'P279' in properties:
                    # this type has already been processed
                    if wid in allTypes: continue

                    superType = properties['P279']
                    superType = [item['mainsnak']['datavalue']['value']['id'] for item in superType]

                    # add a directed edge from supertype to subtype
                    for t in superType:
                        G.add_edge(t, wid)

                    # add this type to processed list
                    allTypes.add(wid)
            except KeyError:
                continue

            except json.JSONDecodeError:
                continue

        with open('typeHierarchy.pickle', 'wb') as f:
            pickle.dump(G, f)


    def extractAliases(self):
        self.connect()

        query = '''insert into aliases(wid, alias) values (?,?);'''

        symbols = []
        line = input()
        line = input().strip()[:-1]
        while line != '':
            try:
                try:
                    content = json.loads(line)
                    wid = content['id']
                    canonical = content['labels']['en']['value']
                    symbols.append((wid, canonical,))

                    aliases = content['aliases']['en']
                    for alias in aliases:
                        label = alias['en']['value']
                        symbols.append((wid, label,))

                except KeyError:
                    line = input().strip()[:-1]
                    continue

                except json.decoder.JSONDecodeError:
                    line = input().strip()[:-1]
                    continue

                #print(wid, label)
                if (len(self.symbols) >= 30000):
                    self.cursor.execute('BEGIN TRANSACTION;')
                    for row in symbols:
                        try:
                            self.cursor.execute(query, row)
                        except sqlite3.IntegrityError:
                            continue
                        except OverflowError:
                            continue

                    self.cursor.execute('END TRANSACTION;')
                    symbols = []

                line = input().strip()[:-1]

            except EOFError:
                break


    def extractFromNT(self):
        '''
        This method scan the NTriple dump after the frist scan, and 
        1. resolves the symbols (wikiIDs) to readable names.
        2. extract synonyms for units

        Entity type, property and units are all included.

        This method reads from stdin. 

        '''

        i = 0
        units = {}

        self.connect()

        unitQuery = '''select distinct unit from relation; '''

        self.cursor.execute(unitQuery)
        result = self.cursor.fetchall()
        for row in result:
            units[row[0]] = ''

        line = input().strip()
        while line != '':
            try:
                if ('/statement/' in line):
                    line = input().strip()
                    continue

                if ('@en' not in line) or ('http://schema.org/name' not in line):
                    line = input().strip()
                    continue


                triple = line.split()
                subj = triple[0]
                pred = triple[1]
                obj = triple[2]
                # this line describes a property of an entity
                if ('/entity/Q' in subj):
                    wid = subj[subj.rfind('Q'):-1]
                elif ('/entity/P' in subj):
                    wid = subj[subj.rfind('P'):-1]
                else:
                    line = input().strip()
                    continue

                name = obj[1:obj.rfind('\"')]

                self.symbols.append((wid, name,))
                i += 1
                if (len(self.symbols) >= 30000):
                    self.pushToDB(flag='mapping')

                if wid in units:
                    units[wid] = name


                line = input().strip()
                #print(wid, name)
            except EOFError:
                break

        with open('unitMapping.pickle', 'wb') as f:
            pickle.dump(units, f)

        self.clean()
        print(i)


    def __resolveUnits__(self):
        units = {}

        self.connect()
        self.cursor.execute('select distinct unit from relation;')
        allUnits = self.cursor.fetchall()

        for unit in allUnits:
            unit = unit[0]
            if unit[0] == 'Q':
                try:
                    label = self.__getLabelByID__(unit)
                    units[unit] = label
                except KeyError:
                    continue

        self.clean()
        print(len(units))
        with open('units.txt', 'w') as f:
            for unit in units:
                f.write(unit + ': \t\t' + units[unit] + '\n')

        with open('units.pickle', 'wb') as f:
            pickle.dump(units, f)
