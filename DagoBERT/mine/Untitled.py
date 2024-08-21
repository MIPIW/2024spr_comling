#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import Counter
from functools import reduce


import pandas as pd
import string

import pandas as pd 
from tqdm import tqdm
tqdm.pandas()

from pathlib import Path
import pandas as pd

import re
from functools import reduce, partial
import numpy as np
import multiprocessing
num_cores = multiprocessing.cpu_count()
from multiprocessing import Pool
class IOUtils():
    def __init__(self):
        pass
    
    @staticmethod
    def existsFile(file_url):
        
        import os
        from pathlib import Path

        if type(file_url) == str:
            url = Path(file_url).resolve()
        else:
            url = file_url.resolve()
        
        if url.exists():
            c = 1
            while True:
                if Path(str(url.parent) + "/" + str(url.stem) + f"_{c}" + str(url.suffix)).exists():
                    c += 1
                else:
                    break

            return Path(str(url.parent) + "/" + str(url.stem) + f"_{c}" + str(url.suffix))
        
        print(f"The url of the fIle to be saved is {url}")
        return url
    
    @staticmethod
    def recentFile(file_url):
        
        import os
        from pathlib import Path

        if type(file_url) == str:
            url = Path(file_url).resolve()
        else:
            url = file_url.resolve()

        if url.exists():
            c = 1
            while True:
                if Path(str(url.parent) + "/" + str(url.stem) + f"_{c}" + str(url.suffix)).exists():
                    c += 1
                else:
                    break
        else:
            c = 0
        if c == 0:
            return f"File not exists in {url}"

        if c == 1:
            print(f'The url of the fIle to be loaded is {Path(str(url.parent) + "/" + str(url.stem + str(url.suffix)))}')
            return Path(str(url.parent) + "/" + str(url.stem + str(url.suffix)))
        else:
            print(f'The url of the fIle to be loaded is {Path(str(url.parent) + "/" + str(url.stem + f"_{c-1}" + str(url.suffix)))}')
            return Path(str(url.parent) + "/" + str(url.stem + f"_{c-1}" + str(url.suffix)))

    @staticmethod
    def checkpoint_save(file_path, data, 
                        data_type = "dataFrame", 
                        file_type = "csv", 
                        index_dataFrame = False):
        import json

        save_path = IOUtils.existsFile(file_path)
        
        if data_type == "dataFrame" or data_type == "series":
            import pandas as pd

            if file_type == "csv":
                data.to_csv(save_path, index = index_dataFrame, encoding = 'utf-8')
                                
        
        if data_type == "list":
            if file_type == "txt":
                with open(save_path, "w", encoding = 'utf-8') as f:
                    for item in data:
                        f.write(item)
                        f.write("\n")

                
            if file_type == "jsonl":
                
                with open(save_path, "w", encoding = 'utf-8') as f:
                    for item in data.items():
                        f.write(json.dumps(item))
                        f.write("\n")

        
        if data_type == "dict":
            if file_type == "json":
                
                
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent = "\t")
        
        return data # for the neatness of the code
    
    @staticmethod
    def checkpoint_load(file_path, 
                        data_type = "dataFrame", 
                        file_type = "csv"):
        load_path = IOUtils.recentFile(file_path)
        
        if data_type == "dataFrame" or data_type == "series":
            import pandas as pd

            if file_type == "csv":
                out = pd.read_csv(load_path)
                return out

        if data_type == "list":
            if file_type == "txt":
                out = None

                with open(load_path, "r", encoding = 'utf-8') as f:
                    out = [i.strip() for i in f.readlines()]
                return out

            if file_type == "jsonl":
                out = []

                with open(load_path, "r", encoding = 'utf-8') as f:
                    for line in f:
                        out.append(json.loads(line))

                return out

        
        if data_type == "dict":
            if file_type == "json":
                
                out = None
                
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent = "\t")
                
                with open(load_path, "r") as f:
                    out = json.loads(f)
                
                return out
                

# 나중에 하기
# def decorator(func, checkFunc):
#     out = func()
#     filter = checkFunc()
    

                

class ParallelizingUtils():
    
    import multiprocessing
    

    def __init__(self, func):
        self.func = func
        pass
    
    
    def do_series(self, series, num_cores, pre_assign = False):
        if num_cores == 1:
            if not pre_assign:
                return self._assign_map(series)
            else:
                return self.func(series)
    
        from multiprocessing import Pool
        import numpy as np
        import pandas as pd

        se_split = np.array_split(series, num_cores)
        pool = Pool(num_cores)
        if not pre_assign:
            df = pd.concat(pool.map(self._assign_map, se_split))
        else:
            df = pd.concat(pool.map(self.func, se_split))
        pool.close()
        pool.join()
        return df

    def _assign_map(self, serie):
        return serie.progress_map(self.func)

    def changeFunc(self, func):
        self.func = func
    


class CheckUtils():
    def __init__(self) -> None:
        pass
    
    # def checkValue(func):
    #     from functools import partial
    #     partial 
    @staticmethod
    def checkSeries(serdf, 
                 isNan = True, 
                 isEmpty = True, 
                 isInf = True,
                    isNotInstance = None,
                    isNotIn = False,
                    isNotInVal = None
                 ):
        
        import pandas as pd
        import numpy as np
        import math
            
        def _check_subsequent(x, 
                              isNotInstance = isNotInstance, 
                              isNan = isNan, 
                              isEmpty = isEmpty, 
                              isInf = isInf, 
                              isnotIn = isNotIn, 
                              isNotInVal = isNotInVal):
            out = False
            
            if isNotInstance is not None:
                x1 = (not isinstance(x, isNotInstance))
            else:
                x1 = False

            if isinstance(x, (int, float, complex)):
                x2 = pd.isna(x) if isNan else False
                x3 = math.isinf(x) if isInf else False
                x4 = False
                
            else:
                x2 = False
                x3 = False
                x4 = (len(x) == 0) if isEmpty else False
                
            x5 = isNotInVal not in x if isNotIn else False

            return out | x1 | x2 | x3 | x4 | x5
        
        
        if isinstance(serdf, pd.Series):
            return serdf.progress_map(lambda x: _check_subsequent(x))
        
        elif isinstance(serdf, pd.DataFrame):
            out  = serdf.map(lambda x: _check_subsequent(x)) # cellwise

            return out.apply(lambda x: any(x), axis = 1)
        
        else:
            raise ValueError("this is not either series or dataframe.")

    
    @staticmethod
    def isEmpty(dataframe) -> bool:
    
        if len(dataframe.index) == 0:
            return True
    
        return False
    
#     @staticmethod
#     def checkValue(inputs, 
# #                    isNone = True, 
#                    isNan = True, 
#                    isEmpty = True, 
#                    isInf = True,
#                    isNotInstance = None
#                    ):
    
#         import pandas as pd
#         import numpy as np
#         import math

#         def _check_subsequent(x, isNotInstance = isNotInstance, isNan = isNan, isEmpty = isEmpty, isInf = isInf):
#             out = False
            
#             if isNotInstance is not None:
#                 x1 = (isinstance(x, isNotInstance))
#             else:
#                 x1 = False

#             if isinstance(x, (int, float, complex)):
#                 x2 = pd.isna(x) if isNan else False
#                 x3 = math.isinf(x) if isInf else False
#                 x4 = False
                
#             else:
#                 x2 = False
#                 x3 = False
#                 x4 = (len(x) == 0) if isEmpty else False

#             return out | x1 | x2 | x3 | x4

#         return _check_subsequent(x, isNotInstance)


# In[2]:


# %%time
# #\ preprocessing

class Semiprocessing():
    
    # remove abbreviations -> pass
    # remove string containing numbersfro
    f1 = lambda x: re.sub(r"[a-z]?\d+[a-z]+", "", x.lower())
    # remove references to users/
    f2 = lambda x: re.sub(r"u/.+", "", x)
    # remove SRs
    f3 = lambda x: re.sub(r"r/.+", "", x)
    # remove full and shorteded hyperlinks
    f4 = lambda x: re.sub("https?://\S+", "", x)
    # convert British English spelling variants to American English -> pass
    # lemmatize to remove inflectional morphology
    f5 = lambda x: re.sub("(.)\1{2,}", "\1\1", x)
    f6 = lambda x: x.translate(str.maketrans('', '', string.punctuation))
    f7 = lambda x: re.sub("(\s){2,}", " ", x)

    
    # reducing repretitions of more than three letters to three letters
        
    def __init__(self, load_spacy = True):
        pass
        
    def process_assign(self, inp):
        return inp.progress_map(lambda x: Semiprocessing.f7(Semiprocessing.f6(Semiprocessing.f5(Semiprocessing.f4(Semiprocessing.f3(Semiprocessing.f2(Semiprocessing.f1(x))))))))
    
    def progess(self, inp):
        return Semiprocessing.f7(Semiprocessing.f6(Semiprocessing.f5(Semiprocessing.f4(Semiprocessing.f3(Semiprocessing.f2(Semiprocessing.f1(inp)))))))


# In[3]:


class MakePrefixStructs():
    def __init__(self, prefix_path):
        
        self.prefix_list = open(IOUtils.recentFile(prefix_path), "r").read().split("\n")        
        self.prefix_hash = self._make_char_dict(self.prefix_list)
        
        pass
    
    def _make_char_dict(self, str_list):

        str_list = sorted(str_list)

        def isempty(dic, word, idx, max_len):

            if idx == max_len-1:
                dic[word[idx]] = 1 
                return dic

            if dic.get(word[idx], 0) == 0:
                dic[word[idx]] = isempty(dict(), word, idx+1, max_len)

            return dic

        str_dic = {}
        for i in str_list:
            str_dic = isempty(str_dic, i, 0, len(i))

        return str_dic


# In[4]:


class LemmaMaker():
    
    def __init__(self, prefix_hash, spacy_type, load_spacy = True):
        self.prefix_hash = prefix_hash
                
        self.loaded_spacy = False
        self.load_model = None
        
        if load_spacy:
            import re, spacy
            self.load_model = spacy.load(spacy_type, disable = ['parser', 'ner'])
            self.loaded_spacy = True
            
        pass
    
    def lemmatize_assign(self, inputs):
        if self.load_model is None:
            raise ValueError("spacy model is not loaded. use 'load_spacy_model' to the instances")
        else:    
            return inputs.progress_map(lambda x: self.load_model(x))
    
    def lemmatize(self, inputs):
        if self.load_model is None:
            raise ValueError("spacy model is not loaded. use 'load_spacy_model' to the instances")
        else:
            return self.load_model(inputs)  
            
        
    #     def pp_for_derivation(x, prefix_list):
    def find_lemma_wo_prefix(self, x, prefix_list):
        lst_token = [""]
        lst_token = [i.text for i in x if (not i.is_stop) and self._wordInDict(i.text, 0, self.prefix_hash)]
        # 이 렘마도 prefix 떼고 dictionary에 넣어야 하는 거 아닌가?
        return {"prefixed_tokens": lst_token, "lemma": " ".join([i.lemma_ for i in x if (not i.is_stop) and i.is_alpha])}

    def find_prefix_with_existing_lemma(self, x, prefix_list):
        x = x.split(" ")
        lst_token = [i for i in x if self._wordInDict(i, 0, self.prefix_hash)]
        return {"prefixed_tokens": lst_token}
        
    def _wordInDict(self, word, curlen, prefix_hash):
        
        if curlen == len(word):
            return False
        
        if isinstance(prefix_hash.get(word[curlen], 0), dict):
            return self._wordInDict(word, curlen + 1, prefix_hash[word[curlen]])
        
        if prefix_hash.get(word[curlen], 0) == 1:
            return True
        
        if prefix_hash.get(word[curlen], 0) == 0:
            return False
        
        
    def load_spacy_model(self, spacy_type):
        
        if not self.loaded_spacy:
            self.load_model = spacy.load(spacy_type, disable = ['parser', 'ner'])
            self.loaded_spacy = True
        else:
            print("spacy model is already loaded.")


# In[5]:


class BinsMaker():
    def __init__(self, lemma_frequent, prefix_list, load_model):
        self.lemma_frequent = list(lemma_frequent)
        self.prefix_list = prefix_list
        self.load_model = load_model
        
        pass
    
    def isRealPrefix(self, word_list):
        
        def eachWord(word):
            p = ""
            
            for prefix in self.prefix_list:
                if word.startswith(prefix):
                    p = prefix
                    break
                    
        #     if load_model(word)[0].lemma_ in lemma_frequent:  # according to the paper, but wrong
        #         return (word, prefix)
        
            if len(word[len(p):]) > 0:
                lemma = self.load_model(word[len(p):])[0].lemma_
                
                if lemma in self.lemma_frequent:
                    return (word, prefix, lemma)
                
                
        return [eachWord(i) for i in word_list]
        
            
    @staticmethod
    def postprocess(series):
        x = series.explode()
        x = x[x.map(lambda x: isinstance(x, tuple))]
        out = pd.concat([pd.Series(x.index, name = "idx"), pd.DataFrame.from_records(x.values, columns = ["token", "prefix", "lemma"])], axis = 1)
        return out

    @staticmethod
    def word_to_blank(prefix_series, article_series):
        idx_lst = list(prefix_series.idx.drop_duplicates())
        pre_idx_token = prefix_series.groupby('idx')['token'].apply(list)
        pre_idx_prefix = prefix_series.groupby('idx')['prefix'].apply(list)
        article_series = article_series[idx_lst]

        for idx in tqdm(idx_lst):

            x = article_series[idx].split()
            y = pre_idx_token[idx]
            z = pre_idx_prefix[idx]

            x.append("[CROP]")

            for i in range(len(x)):
                for prefix, word in zip(z, y):
                    if x[i] == word:

                        x.append("[AND]".join([prefix, word]))
                        x.append("[ANDMORE]")

                        x[i] = "___"

            article_series[idx] = " ".join(x)

        return article_series

    @staticmethod
    def split_sentence(article_series):
        import re 

        def split_string(string):

            lst_whole = string.split('[CROP]')
            lst_sent = lst_whole[0].split("___")
            lst_prefix_word = [i.split("[AND]") for i in lst_whole[1].split("[ANDMORE]")]

            lst_final = []

            for idx in range(len(lst_sent)-1):
                t = lst_prefix_word[idx][0].strip()
                word = lst_prefix_word[idx][1].strip()
                base = word[len(t):]
                temp1 = lst_sent[idx].strip()
                temp2 = lst_sent[idx+1].strip()
                a = temp1[-150:-50]
                b = temp1[-50:] + " ___ " + temp2[:50]
                c = temp2[50:200]
                
                lst_final.append("asdf" + "|||" + word + "|||" + base + "|||" + a + "|||" + b + "|||" + c)

            return lst_final

        return article_series.progress_map(split_string).explode().reset_index(drop = True)

    ###############lemma 기준으로 count해야됨 기말때 보완하기로

    @staticmethod
    def classify_bins(series, boundary, t):
        import pickle
        
        count = Counter((series.progress_map(lambda x: x.split("|||")).map(lambda x: x[2])))
        
        with open(f"../dagobert-master/src/data/d_counter_{t}.p", "wb") as fw:
            pickle.dump(count, fw)
        
        
        lst = []
        for i in count.items():
            for j in range(len(boundary)-1):
                if boundary[j] <= i[1] < boundary[j+1]:
                    lst.append(boundary[j])
                    break

        freq_bins = {i[0]: j for i, j in zip(count.items(), lst)}   
        count_num = series.progress_map(lambda x: x.split("|||")).map(lambda x: freq_bins[x[2]])
        count_num.name = "freq"

        return pd.concat([series.copy(), count_num.copy()], axis = 1)



# In[6]:


original_path = "./data/cnn_dailymail_data/cnn_dailymail.csv"
prefix_path = "../dagobert-master/data/external/prefix_list.txt"
semiprocessed_path = "./data/cnn_dailymail_data/cnn_dailymail_semiprocessed.csv"
lemma_path = "./data/cnn_dailymail_data/lemma_common.csv"
prefix_processed_path = "./data/cnn_dailymail_data/prefix_preprocessed.csv"
binswhole_path = "./data/cnn_dailymail_data/bins.txt"
binspartial_path = "./data/cnn_dailymail_data/bins{:02d}.txt"

spacy_type = 'en_core_web_sm'

def process(
    original_path,
    prefix_path,
    semiprocessed_path,
    lemma_path,
    prefix_processed_path,
    binswhole_path,  
    binspartial_path,
    spacy_type,
    data_types,
    num_cores = 1,
    save_semiprocess = True,
    save_lemma = True,
    save_preprocess = True,
    save_bins = True,
    save_subbins = True,
    slice_len = None
):

    prefix_list = open(IOUtils.recentFile(prefix_path), "r").read().split("\n")
    semi_processor = Semiprocessing()
    # processor = Semiprocessing(prefix_list, load_spacy = False)
    
    parallel = ParallelizingUtils(semi_processor.process_assign)
    # Semiprocessing
    print("saving semiprocessed files...")    
    if save_semiprocess:
        
        # read original data
        cnn_path = IOUtils.recentFile(original_path)
        if slice_len == None:
            cnn_data = pd.read_csv(cnn_path, usecols = ["article"])
        else:
            cnn_data = pd.read_csv(cnn_path, usecols = ["article"], nrows = slice_len)
       
        # process original data: df: ['article': text] -> df: ['article': processed text]
        cnn_data_semiprocessed = parallel.do_series(cnn_data.article, num_cores = num_cores, pre_assign = True)
        filters = CheckUtils.checkSeries(cnn_data_semiprocessed,
                                         isNan = True,
                                         isEmpty = True,
                                         isInf = False,
                                         isNotInstance = str).map(lambda x: not x)
        cnn_data_semiprocessed = cnn_data_semiprocessed[filters].drop_duplicates().reset_index(drop = True)        
        cnn_data_semiprocessed = IOUtils.checkpoint_save(IOUtils.existsFile(semiprocessed_path), 
                                                         cnn_data_semiprocessed, 
                                                         data_type = "dataFrame", 
                                                         file_type = "csv", 
                                                         index_dataFrame = False)
        print("done")
        
    cnn_data_semiprocessed = IOUtils.checkpoint_load(IOUtils.recentFile(semiprocessed_path),
                                                 data_type = "dataFrame",
                                                 file_type = "csv")    

    # load prefix_dic(successive hashmap dictionary)
    prefix_structure = MakePrefixStructs(prefix_path)
    lemmaMaker = LemmaMaker(prefix_structure.prefix_hash, spacy_type, load_spacy = True)
    
    if save_lemma: 
        # lemmatize
        parallel.changeFunc(lemmaMaker.lemmatize_assign)
        print("extracting frequent lemmas: lemmatizing")
        cnn_data_process = parallel.do_series(cnn_data_semiprocessed.article, num_cores = num_cores, pre_assign = True)
        
        filters = CheckUtils.checkSeries(cnn_data_process,                                  
                                         isNan = True,                                 
                                         isEmpty = True,                                 
                                         isInf = False,                                 
                                         isNotInstance = None,                                        
                                         isNotIn = False,                                       
                                         isNotInVal = None).map(lambda x: not x)
        cnn_data_process = cnn_data_process[filters].reset_index(drop = True)
        
        
        print("extracting frequent lemmas: removing prefixes")
        parallel.changeFunc(partial(lemmaMaker.find_lemma_wo_prefix, prefix_list = prefix_structure.prefix_list))
        cnn_data_process = parallel.do_series(cnn_data_process, num_cores = num_cores, pre_assign = False)
        cnn_data_process = pd.DataFrame(list(cnn_data_process))
        filters = CheckUtils.checkSeries(cnn_data_process,                                  
                                         isNan = True,                                 
                                         isEmpty = True,                                 
                                         isInf = False,                                 
                                         isNotInstance = None,                                    
                                         isNotIn = False,                                       
                                         isNotInVal = None).map(lambda x: not x)
        cnn_data_process = cnn_data_process[filters].reset_index(drop = True)
        
        print("done")
    # do not needed when processing reddit text, just load the lemma_common.csv
        print("extracting frequent lemmas: extracting frequent lemma")
        
        count = Counter((" ".join(cnn_data_process.lemma)).split(" ")).most_common(1000)
        lemma_frequent = set([i[0] for i in count if len(i[0]) > 2 ])
        
        print("saving lemma...")
        lemma_frequent = pd.Series(sorted(list(lemma_frequent)))
        lemma_frequent = IOUtils.checkpoint_save(IOUtils.existsFile(lemma_path), 
                                                 lemma_frequent, 
                                                 data_type = "dataFrame", 
                                                 file_type = "csv", 
                                                 index_dataFrame = False)
        print("done")
    else:    
        parallel.changeFunc(partial(lemmaMaker.find_prefix_with_existing_lemma, prefix_list = prefix_structure.prefix_list))
        cnn_data_process = parallel.do_series(cnn_data_semiprocessed.article, num_cores = num_cores, pre_assign = False)
        cnn_data_process = pd.DataFrame(list(cnn_data_process))
        filters = CheckUtils.checkSeries(cnn_data_process,                                  
                                         isNan = True,                                 
                                         isEmpty = True,                                 
                                         isInf = False,                                 
                                         isNotInstance = str,
                                         isNotIn = False,                                       
                                         isNotInVal = None).map(lambda x: not x)
        cnn_data_process = cnn_data_process[filters].reset_index(drop = True)

        
    lemma_frequent = IOUtils.checkpoint_load(IOUtils.recentFile(lemma_path),
                                             data_type = "dataFrame",
                                             file_type = "csv")
    lemma_frequent = list(lemma_frequent['0'])
    

    if save_preprocess:
        # check if the derivatives candidates are computationally derivatives
        print("checking prefixes...")
        lemmaMaker.load_spacy_model(spacy_type)
        binsMaker = BinsMaker(lemma_frequent= lemma_frequent, prefix_list = prefix_structure.prefix_list, load_model = lemmaMaker.load_model)
        parallel.changeFunc(binsMaker.isRealPrefix)
        cnn_data_process = parallel.do_series(cnn_data_process.prefixed_tokens, num_cores = num_cores, pre_assign = False)   
        cnn_data_process = BinsMaker.postprocess(cnn_data_process)
        
        filters = CheckUtils.checkSeries(cnn_data_process,                                  
                                         isNan = True,                                                                      
                                         isEmpty = True,                                                                     
                                         isInf = False,                                                                      
                                         isNotInstance = None,
                                         isNotIn = False,                                       
                                         isNotInVal = None).map(lambda x: not x)
        cnn_data_process = cnn_data_process[filters].reset_index(drop = True)
        cnn_data_process = IOUtils.checkpoint_save(IOUtils.existsFile(prefix_processed_path), 
                                                   cnn_data_process, 
                                                   data_type = "dataFrame", 
                                                   file_type = "csv",                                          
                                                   index_dataFrame = False)
        print("done")

    
    cnn_data_process = IOUtils.checkpoint_load(IOUtils.recentFile(prefix_processed_path),
                                                 data_type = "dataFrame",
                                                 file_type = "csv")    

    if save_bins:
        print("binning...")
        if slice_len == None:
            bins = pd.read_csv(IOUtils.recentFile(semiprocessed_path), usecols = ["article"])
        else:
            bins = pd.read_csv(IOUtils.recentFile(semiprocessed_path), usecols = ["article"], nrows = slice_len)

        
        
        bins = BinsMaker.word_to_blank(cnn_data_process.copy(), bins.article.copy())     
        bins = BinsMaker.split_sentence(bins.copy())
    
        filters = CheckUtils.checkSeries(bins,
                                     isNan = True,
                                     isEmpty = True,
                                     isInf = False,
                                     isNotInstance = str,                                         
                                         isNotIn = False,                                       
                                         isNotInVal = None).map(lambda x: not x)
        bins = bins[filters].drop_duplicates().reset_index(drop = True)
        bins = bins.explode().reset_index(drop = True)
        
        bins = IOUtils.checkpoint_save(IOUtils.existsFile(binswhole_path), 
                                                   bins, 
                                                   data_type = "dataFrame", 
                                                   file_type = "csv",                                          
                                                   index_dataFrame = False)

        
        print("done")
    
    if save_subbins:
        print("subbinning...")
        bins = IOUtils.checkpoint_load(IOUtils.recentFile(binswhole_path),
                                                     data_type = "dataFrame",
                                                     file_type = "csv")  
        boundary = [1,2,4,8,16,32,64,128,100000000 ]
        bins = bins[bins.article.map(lambda x: isinstance(x, str))].reset_index(drop = True)
        bins = BinsMaker.classify_bins(bins.article, boundary, data_types)

        ser = bins.groupby('freq')['article'].apply(list)
        for freq in boundary[:-1]:
            if freq in ser.index:
                pd.DataFrame(ser[freq]).to_csv(IOUtils.existsFile(binspartial_path.format(freq)), header = False, index = False)
        print("done")
        
    print("Whole Jobs Done. Good Job")


# In[7]:


# %%time 

# original_path = "./data/cnn_dailymail_data/cnn_dailymail.csv"
# prefix_path = "../dagobert-master/data/external/prefix_list.txt"
# semiprocessed_path_temp = "./data/cnn_dailymail_data/temp/cnn_dailymail_semiprocessed.csv"
# lemma_path_temp = "./data/cnn_dailymail_data/temp/lemma_common.csv"
# prefix_processed_path_temp = "./data/cnn_dailymail_data/temp/prefix_preprocessed.csv"
# binswhole_path_temp = "./data/cnn_dailymail_data/temp/bins.txt"
# binspartial_path_temp = "./data/cnn_dailymail_data/temp/bins{:02d}.txt"
# slice_len = 1000
# spacy_type = 'en_core_web_sm'

# x = process(original_path = original_path,
#        prefix_path = prefix_path,
#        semiprocessed_path = semiprocessed_path_temp,
#        lemma_path = lemma_path_temp,
#        prefix_processed_path = prefix_processed_path_temp,
#        binswhole_path = binswhole_path_temp,
#        binspartial_path = binspartial_path_temp,
#         spacy_type = spacy_type,
#         num_cores = 32,
#         save_lemma = True,
#         slice_len = slice_len
#        )


# In[8]:


# original_path = "../dagobert-master/data/final/raw.txt"
# partial_bin_path = "../dagobert-master/data/final/bin{:02d}.txt"

# def reverse_process(partial_bin_path, output_path, boundary = [1,2,4,8,16,32,64]):    
#     x = pd.DataFrame()
#     for b in boundary:
#         print("merging {:02d}th bin".format(b))
#         bin01 = [i.strip() for i in open(IOUtils.recentFile(partial_bin_path.format(b)), "r").readlines()]
#         bin01 = pd.DataFrame([i.split("|||") for i in bin01])
#         bin01 = bin01.apply(lambda x: x[3] + re.sub("___", x[1], x[4]) + x[5], axis = 1)    
#         bin01.name = "article"
#         bin01 = bin01.to_frame()
#         x = pd.concat([x, bin01], axis = 0)
#     return x

# bin01 = reverse_process(partial_bin_path, original_path)ㄴ
# bin01.to_csv(IOUtils.existsFile(original_path), index = False)
# bin01 = pd.read_csv(IOUtils.recentFile(original_path))
# bin01 = bin01.drop_duplicates().reset_index(drop = True)


# In[9]:


#process with bert 

"||||||".split("|||")


# In[10]:


# original_path = "./data/cnn_dailymail_data/cnn_dailymail.csv"
# prefix_path = "../dagobert-master/data/external/prefix_list.txt"
# semiprocessed_path_temp = "./data/cnn_dailymail_data/cnn_dailymail_semiprocessed.csv"
# lemma_path_temp = "./data/cnn_dailymail_data/lemma_common.csv"
# prefix_processed_path_temp = "./data/cnn_dailymail_data/prefix_preprocessed.csv"
# binswhole_path_temp = "./data/cnn_dailymail_data/bins.txt"
# binspartial_path_temp = "./data/cnn_dailymail_data/temp/temp/bins{:02d}.txt"
# spacy_type = 'en_core_web_sm'

# slice_len = None
# spacy_type = 'en_core_web_sm'

# x = process(original_path = original_path,
#        prefix_path = prefix_path,
#        semiprocessed_path = semiprocessed_path_temp,
#        lemma_path = lemma_path_temp,
#        prefix_processed_path = prefix_processed_path_temp,
#        binswhole_path = binswhole_path_temp,
#        binspartial_path = binspartial_path_temp,
#             data_types = "cnn",
#         spacy_type = spacy_type,
#         num_cores = 40,
#         save_semiprocess = False,
#         save_lemma = False,         
#         save_preprocess = False,    
#         save_bins = True,    
#         save_subbins = True,
#         slice_len = slice_len
#        )


# In[11]:


3


# In[11]:


original_path = "../dagobert-master/data/raw.txt"
prefix_path = "../dagobert-master/data/external/prefix_list.txt"
semiprocessed_path_temp = "../dagobert-master/data/raw_semiprocessed.csv"
lemma_path_temp = "../dagobert-master/data/lemma_common.csv"
prefix_processed_path_temp = "../dagobert-master/data/raw_prefix_processed.csv"
binswhole_path_temp = "../dagobert-master/data/raw_bins.csv"
binspartial_path_temp = "../dagobert-master/data/temp/raw_bins{:02d}.txt"

slice_len = None
spacy_type = 'en_core_web_sm'

x = process(original_path = original_path,
       prefix_path = prefix_path,
       semiprocessed_path = semiprocessed_path_temp,
       lemma_path = lemma_path_temp,
       prefix_processed_path = prefix_processed_path_temp,
       binswhole_path = binswhole_path_temp,
       binspartial_path = binspartial_path_temp,
        spacy_type = spacy_type,
            data_types = "reddit",
        num_cores = 40,
        save_semiprocess = True,
        save_lemma = True,         
        save_preprocess = True,    
        save_bins = True,    
        save_subbins = True,
        slice_len = slice_len
       )


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[13]:


# out = cnn_data_process.map(lambda x: list(filter(lambda y: y[0] != "-", x)))

# x = out.explode()
# x = x[x.map(lambda x: isinstance(x, tuple))]

# out = pd.concat([pd.Series(x.index, name = "idx"), pd.DataFrame.from_records(x.values, columns = ["token", "prefix", "lemma"])], axis = 1)
# out_url = existsFile("./data/cnn_dailymail_data/prefix_preprocessed.csv")
# out.to_csv(out_url, encoding = "utf-8", index = False)



# In[14]:


# ppr_url = Path("./data/cnn_dailymail_data/prefix_preprocessed.csv").resolve()
# lem_url = Path("./data/cnn_dailymail_data/lemma_common.csv").resolve()
# cnn_url = Path("./data/cnn_dailymail_data/cnn_dailymail_semiprocessed.csv").resolve()

# cnn_pre = pd.read_csv(ppr_url)
# lemma = pd.read_csv(lem_url)
# cnn_data = pd.read_csv(cnn_url, usecols = ['article'])


# In[ ]:




