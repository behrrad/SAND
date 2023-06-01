from easysparql.easysparqlclass import EasySparql
from multiprocessing import Process, Pipe
import pandas as pd
from PPool.Pool import Pool
import time
import tqdm

ENDPOINT = "https://dbpedia.org/sparql"
CACHE_DIR = ".cache"
MIN_NUM_NUMS = 30


def features_gatherer(reciever_p, sender_p):
    """
    :param d:
    :param input_p:
    :param output_p:
    :return:
    """
    pairs = []
    while True:
        d = reciever_p.recv()
        if d is None:
            for pair in pairs:
                sender_p.send(pair)
            sender_p.send(None)
            return
        else:
            pairs.append(d)


def get_num(num_or_str):
    """
    :param num_or_str:
    :return: number or None if it is not a number
    """
    if pd.isna(num_or_str):
        return None
    elif isinstance(num_or_str, (int, float)):
        return num_or_str
    elif isinstance(num_or_str, str):
        if '.' in num_or_str or ',' in num_or_str or num_or_str.isdigit():
            try:
                return float(num_or_str.replace(',', ''))
            except Exception as e:
                return None
    return None


def get_numerics_from_list(nums_str_list):
    """
    :param nums_str_list: list of string or numbers or a mix
    :return: list of numbers or None if less than 50% are numbers
    """
    nums = []
    for c in nums_str_list:
        n = get_num(c)
        if n is not None:
            nums.append(n)
    if len(nums) < len(nums_str_list) / 2:
        return None
    return nums


def features_and_kinds_func(class_uri, property_uri):
    """
    :param property_uri:
    :return:
    """
    easysparql = EasySparql(endpoint=ENDPOINT, cache_dir=CACHE_DIR)
    values = easysparql.get_objects(class_uri=class_uri, property_uri=property_uri)
    nums = get_numerics_from_list(values)
    # pair = {
    #     'values': nums,
    #     'property_uri': property_uri,
    #     'class_uri': class_uri
    # }
    return nums
    # pipe.send(pair)


def get_features_and_kinds_multi_thread(class_uri):
    """
    :param class_uri:
    :return:
    """
    easysparql = EasySparql(endpoint=ENDPOINT, cache_dir=CACHE_DIR)
    properties = easysparql.get_class_properties(class_uri=class_uri, min_num=10)
    print("get_features_and_kinds_multi_thread> properties: ", str(class_uri))
    fk_pairs = {}
    for p in properties:
        fk_pair = features_and_kinds_func(class_uri, p)
        if fk_pair:
            if (class_uri, p, 't') not in fk_pairs:
                fk_pairs[(class_uri, p, 't')] = []
            fk_pairs[(class_uri, p, 't')].extend(fk_pair)
    return fk_pairs

    # features_send_pipe, features_recieve_pipe = Pipe()
    # gatherer_send_pipe, gatherer_reciever_pipe = features_send_pipe, features_recieve_pipe
    # gatherer = Process(target=features_gatherer, args=(gatherer_reciever_pipe, gatherer_send_pipe))
    # gatherer.start()
    # params = []
    # for p in properties:
    #     params.append((class_uri, p, features_send_pipe))
    #
    # pool = Pool(max_num_of_processes=1, func=features_and_kinds_func, params_list=params)
    # pool.run()
    # features_send_pipe.send(None)
    # fk_pairs = []
    # fk_pair = features_recieve_pipe.recv()
    # while fk_pair is not None:
    #     fk_pairs.append(fk_pair)
    #     fk_pair = features_recieve_pipe.recv()
    # gatherer.join()

    # return fk_pairs


if __name__ == '__main__':
    # start_time = time.perf_counter()
    pairs = get_features_and_kinds_multi_thread("http://dbpedia.org/ontology/PoliticalParty")
    # end_time = time.perf_counter()
    # print("time: ", str(end_time - start_time))
    print(len(pairs))
    for pair in pairs:
        print(pair['property_uri'])
