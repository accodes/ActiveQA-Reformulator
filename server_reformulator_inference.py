# For Reformulator
from px.nmt import reformulator
from px.proto import reformulator_pb2
import numpy as np
import requests
import flask_excel
import flask_excel as excel
# import xlrd
from openpyxl import Workbook
from openpyxl import load_workbook
import pandas as pd
import xlsxwriter



# For Flask
import io
import json  # for reading and writing json data
import os  # for interacting with os and perform os operation eg. saving uploaded file
from flask import Flask  # to deploy as a flask service
from flask import jsonify  # for returning json response
from flask import request  # to be able to read the incoming requests
from flask_cors import CORS  # to enable Cross-Origin Resource Sharing
from flask import jsonify, send_file
from flask import send_from_directory, send_file


## Old Paraphrase
from flask import render_template                  # to render html page
from collections import OrderedDict                # preserves the order in which the keys are inserted in a dict
import re                                          # provides support for regular expression
import json                                        # to convert string to python dictionary to save results in MongoDB
import spacy                                       # to perform tokenization and POS tagging on multilingual sentence
import csv                                         # to read/write CSV files
from OldPara.config import Config                          # importing the configrable variables
import logging 

# Creating the flask api object
app = Flask(__name__, static_url_path='')                                       # to enable CORS for the services
app.config.from_object(Config)                       # attaching the configurations to the flask app
CORS(app, resources=r"/*")


logging.basicConfig(
    filename=app.config['LOG_FILENAME'], 
    filemode='a',
    format=app.config['LOG_FORMAT'], 
    datefmt=app.config['LOG_DATE_FORMAT'], 
    level=app.config['LOG_LEVEL']
)
logging.getLogger().addHandler(logging.StreamHandler())

# Loading spacy multilingual packages so that nlp can be applied
SPACY_PACKAGE_EN = spacy.load('en_core_web_sm')      # spacy english language package
SPACY_PACKAGE_FR = spacy.load('fr_core_news_sm')     # spacy french language package
SPACY_PACKAGE_DE = spacy.load('de_core_news_sm')     # spacy german language package

def get_synonyms_from_conceptnet(input_tokens, CONCEPT_NET_LANGUAGE, features):

    # validate the input tokens, and return failure if nothing is available
    if len(input_tokens) == 0:
        return {"error" : "data not found"}
    
    # convert input items to lower case #todo: optimize
    input_tokens = [tmp.lower() for tmp in input_tokens]
    
    # remove duplicates
    input_tokens = list(set(input_tokens))

    words_for_token = {}
    antonyms_for_token = {}

    
    # this will limit the number of results to fetch from Concept Net
    number_of_results_to_request = app.config['CONCEPT_NET_RESPONSE_LIMIT']

        

    for token in input_tokens:
                
        # construct the search URL as expected by Concept Net   #Task 1
        concept_net_search_iri = '/' + app.config['CONCEPT_NET_CONCEPT_IDENTIFIER'] + '/' + CONCEPT_NET_LANGUAGE + '/' + token
        
        # tokens are replaced for space characters by underscore in a request to Concept net, it is reverted for constructing the output dictionary
        token = token.replace('_', ' ')
                
        # fetch data w.r.t token from Concept Net if not found locally
        concept_net_request = app.config['CONCEPT_NET_SERVICE_URL'] + concept_net_search_iri + '?offset=0&limit=' + str(number_of_results_to_request)
        try:
            if app.config['HTTP_PROXY'] != '':
                concept_net_response = requests.get(concept_net_request, proxies = {'http' : app.config['HTTP_PROXY']}).json()
            else:
                concept_net_response = requests.get(concept_net_request).json()
        except Exception as ex:
            raise ConnectionError("Error connecting to concept net: {0} : proxy: {1}".format(concept_net_request, app.config['HTTP_PROXY']) + str(ex))
                
        # make a copy of the received response to find and append all the available responses.
        # concept net returns with a key called 'view' if there are more results available for the token
        tmp_response = concept_net_response
        while 'view' in list(tmp_response.keys()): 
            if 'nextPage' in list(tmp_response['view'].keys()):
                tmp_response = requests.get(app.config['CONCEPT_NET_SERVICE_URL'] + tmp_response['view']['nextPage'], proxies = {'http' : app.config['HTTP_PROXY']}).json()
                concept_net_response['edges'].extend(tmp_response['edges'])
            else:
                break

        # nullify tmp_response
        tmp_response = None   

        words_for_token[token] = []        

        words_for_token[token].extend(add_words_from_conceptnet(token, 'Synonym', CONCEPT_NET_LANGUAGE, concept_net_response))

        if features['INCLUDE_ANTONYMS'] == 'true':
            words_for_token[token].extend(add_words_from_conceptnet(token, 'Antonym', CONCEPT_NET_LANGUAGE, concept_net_response))

        if features['INCLUDE_HYPER_HYPO_NYMS'] == 'true':
            words_for_token[token].extend(add_words_from_conceptnet(token, 'IsA', CONCEPT_NET_LANGUAGE, concept_net_response))

        if features['INCLUDE_RELATED'] == 'true':
            words_for_token[token].extend(add_words_from_conceptnet(token, 'RelatedTo', CONCEPT_NET_LANGUAGE, concept_net_response))


        # filter synonym lists which have less than 2 synonyms, Cocnept Net returns at least one synonym which is the original word itself.
        if len(words_for_token[token]) == 1:
            if token == words_for_token[token][0]:
                del words_for_token[token]
        elif len(words_for_token[token]) > 0:
            words_for_token[token] = [str(tmp).lower() for tmp in words_for_token[token]]
            #words_for_token[token] = [str(tmp.encode('utf-8')).lower() for tmp in words_for_token[token]]
            tmp_lowercase_synonyms = None
            words_for_token[token] = list(set(words_for_token[token]))
        else:
            # remove empty synonym lists from synoym set
            del words_for_token[token]

    return words_for_token


def add_words_from_conceptnet(token, label, CONCEPT_NET_LANGUAGE, concept_net_response):
    # initialize the synonyms item for token being processed
    words_for_token = []
    # to get the threshold weight to limit responses based on quality
    threshold_weight = get_threshold_weight(token, label, CONCEPT_NET_LANGUAGE, concept_net_response)
    # adding token synonyms in list
    words_for_token.extend([tmp['start']['label'] for tmp in concept_net_response['edges'] if tmp['@type'] == 'Edge' and tmp['rel']['label']==label and tmp['start']['language'] == CONCEPT_NET_LANGUAGE and tmp['end']['language'] == CONCEPT_NET_LANGUAGE and tmp['start']['label'] != token and tmp['weight'] >= threshold_weight])
    words_for_token.extend([tmp['end']['label'] for tmp in concept_net_response['edges'] if tmp['@type'] == 'Edge' and tmp['rel']['label']==label and tmp['start']['language'] == CONCEPT_NET_LANGUAGE and tmp['end']['language'] == CONCEPT_NET_LANGUAGE and tmp['end']['label'] != token and tmp['weight'] >= threshold_weight])
    # remove any duplicate synonyms from the list
    words_for_token = list(set(words_for_token))
    
    if label == 'Antonym':
        # adding negation before merging these antonyms in synonyms list
        if CONCEPT_NET_LANGUAGE == 'en': negateWord = "not"
        elif CONCEPT_NET_LANGUAGE == 'fr': negateWord = "ne pas"
        elif CONCEPT_NET_LANGUAGE == 'de': negateWord = "nicht"
        antonym_words = []
        for word in words_for_token:
            antonym_words.append(negateWord+' '+word)
        words_for_token = antonym_words
    
    return words_for_token



def get_threshold_weight(token, label, CONCEPT_NET_LANGUAGE, concept_net_response):
    # Get the threshold weights for antonyms greater than 1
    top_n_weight = app.config['TOP_N_WEIGHT']

    weight = set()
    for tmp in concept_net_response['edges']:
        if (tmp['@type'] == 'Edge' and tmp['rel']['label']==label and tmp['start']['language'] == CONCEPT_NET_LANGUAGE and tmp['end']['language'] == CONCEPT_NET_LANGUAGE and (tmp['start']['label'] != token or tmp['end']['label'] != token) and tmp['weight'] > 1):
            weight.add(tmp['weight'])
    weight = sorted(list(weight), reverse=True)
    if not(weight):
        threshold_weight = 1
    elif len(weight) < top_n_weight:
        threshold_weight = weight[-1]
    else:
        threshold_weight = weight[top_n_weight-1]

    return threshold_weight








def read_json(filename):
    if not os.path.exists(filename):
        return False
    with open(filename, encoding='ISO-8859-1') as json_data:
        return json.loads(json_data.read())




def read_csv(filename):
    if not os.path.exists(filename):
        return False
    with open(filename, 'r', encoding='ISO-8859-1') as csvfile:
        # csvListOfRows contains the csv file data in the form of list of (rows in list format)
        csvListOfRows = csv.reader(csvfile)
        domain_data = {'en': {}, 'fr': {}, 'de': {}}
        # noOfColRem contains n, where first n columns that is irrelevant
        noOfColRem = app.config['FIRST_N_COL_REM_FROM_CSV']
        # noOfRowRem contains n, where first n rows that is irrelevant
        noOfRowRem = app.config['FIRST_N_ROW_REM_FROM_CSV']
        # csvColLang contains the column no. that carries the language information in csv file
        csvColLang = app.config['CSV_COL_LANG']-1
        for row in csvListOfRows:
            if noOfRowRem > 0:
                noOfRowRem = noOfRowRem - 1
            else:
                break
        for row in csvListOfRows:
            if len(row) > (noOfColRem+1):
                # to remove the first n columns that is irrelevant
                transformedRow = row[noOfColRem:]
                transformedJson = {}
                # to strip the extra spaces before and after the phrase
                transformedRow = [x.lower().strip() for x in transformedRow]
                # to sort the synonyms based on no. of words in descending order to search for synonyms in sentence
                transformedRow.sort(key=lambda x: len(x.split()), reverse=True)
                # Remove empty strings from a list of strings
                transformedRow = list(filter(None, transformedRow))
                if len(transformedRow) < 2:
                    continue
                for index, phrase in enumerate(transformedRow):
                    transformedRowForJson = transformedRow[:]
                    # to check that this synonym list belongs to which language
                    if csvColLang != -1:
                        if row[csvColLang] == 'en':
                            domain_data['en'][phrase] = transformedRow[:index]+transformedRow[index+1:]
                        elif row[csvColLang] == 'fr':
                            domain_data['fr'][phrase] = transformedRow[:index]+transformedRow[index+1:]
                        elif row[csvColLang] == 'de':
                            domain_data['de'][phrase] = transformedRow[:index]+transformedRow[index+1:]
                    else:
                        domain_data['en'][phrase] = transformedRow[:index]+transformedRow[index+1:]
        return domain_data

## ----------------------------------------------------------------------------------------------------------

reformulator_instance = reformulator.Reformulator(
    hparams_path='px/nmt/example_configs/reformulator.json',
    source_prefix='<en> <2en> ',
    out_dir='./tmp/active-qa/reformulator/',
    environment_server_address= 'localhost:10000')

### To convert list of list of list... to single list.
def flatten(lst):
  return sum( ([x] if not isinstance(x, list) else flatten(x)
         for x in lst), [] )


def question_paraphrase(query):
    # print("query0:", type(query), query)
    questions = query
    all_reformulated_question =[]
    responses = reformulator_instance.reformulate(
        questions=questions,
        inference_mode=reformulator_pb2.ReformulatorRequest.BEAM_SEARCH)
    reformulations = [[rf.reformulation for rf in rsp] for rsp in responses]
    return reformulations









@app.route("/paraphrase", methods=["POST"])
def paraphrase():
    try:
        message = request.json
        query = message["questions"]
        flag = message["flag"]

        if flag =="0":
            query=list(query)
            print('Working3')
            reformulations = question_paraphrase(query)
            print('Working4')
            return jsonify({"reformulations":reformulations}) 


        if flag =="1":
            # print("query:", type(query), query)
            print('Working1')
            reformulations = question_paraphrase(query)
            # print("reformulations" ,reformulations)
            print('Working2')

            #Create DF
            # data={}
            # col = ["Original Question" , "Reformulations"]
            df = pd.DataFrame() 
            for i in range (len(query)):
                for j in range (20):
                    # print(query[i] , reformulations[i][j])
                    df = df.append({"Original Question": query[i] , "Reformulations":reformulations[i][j] }, ignore_index = True )
            # print("df:", df)

            #Convert DF
            strIO = io.BytesIO()

            writer = pd.ExcelWriter(strIO, engine='xlsxwriter')
            # writer1 = pd.ExcelWriter("utterances1.xlsx", engine='xlsxwriter')
            df.to_excel(writer, sheet_name='sheet1', index=False,  header=False)
            writer.save()
            # writer1.save()

            excel_data = strIO.getvalue()
            strIO.seek(0)

            # return excel.make_response_from_array(strIO, "xlsx")
            return send_file(strIO, attachment_filename='utterances.xlsx', as_attachment=True)
            
        # return jsonify({"reformulations":reformulations}) 
    except:
        print("No input is provided.")




    try:
        excel_file = request.get_array(field_name="file")
        query = flatten(excel_file)
        print('Working5')
        reformulations = question_paraphrase(query)
        print('Working6')

        ##Create DF
        # data={}
        # col = ["Original Question" , "Reformulations"]
        df = pd.DataFrame()
        for i in range (len(query)):
            for j in range (20):
                # print(query[i] , reformulations[i][j])
                df = df.append({"Original Question": query[i] , "Reformulations":reformulations[i][j] }, ignore_index = True )
        # print("df:", df)

        ##Convert DF
        strIO = io.BytesIO()

        writer = pd.ExcelWriter(strIO, engine='xlsxwriter',  header=False)
        # writer1 = pd.ExcelWriter("utterances1.xlsx", engine='xlsxwriter')
        df.to_excel(writer, sheet_name='sheet1', index=False)
        writer.save()
        # writer1.save()

        excel_data = strIO.getvalue()
        strIO.seek(0)
        return send_file(strIO, attachment_filename='utterances.xlsx', as_attachment=True)

    except:
        print("No excel file provided.")


## ------------------------------------------------------------------------------------------------

@app.route('/findsynonyms', methods = ['POST'])
def find_synonyms():
    # check the input recieved is as per required
    try:
        input = request.json
        logging.info (input)
    except Exception as ex:
        return jsonify({"status":"error","result":"Request received in a format other than JSON"}),501
    if "sentence" in list(input.keys()) and "lang" in list(input.keys()) and (input["lang"].lower() in app.config['LANGUAGES_SUPPORTED']):
        sentence = input["sentence"]
        lang = input["lang"].lower()
    else:
        return jsonify({"status":"error","result":"Input sentence or language not recieved"}),501

    features = {}    

    if "features" in list(input.keys()):
        input_features = input["features"]
        
        if "antonyms" in list(input_features.keys()):
            features['INCLUDE_ANTONYMS'] = input_features["antonyms"].lower()
        else:
            features['INCLUDE_ANTONYMS'] = app.config['FEATURES']['INCLUDE_ANTONYMS']

        if "hyper_hypo_nyms" in list(input_features.keys()):
            features['INCLUDE_HYPER_HYPO_NYMS'] = input_features["hyper_hypo_nyms"].lower()
        else:
            features['INCLUDE_HYPER_HYPO_NYMS'] = app.config['FEATURES']['INCLUDE_HYPER_HYPO_NYMS']

        if "related" in list(input_features.keys()):
            features['INCLUDE_RELATED'] = input_features["related"].lower()
        else:
            features['INCLUDE_RELATED'] = app.config['FEATURES']['INCLUDE_RELATED']
        
        del input_features

    else:
        features['INCLUDE_ANTONYMS'] = app.config['FEATURES']['INCLUDE_ANTONYMS']
        features['INCLUDE_HYPER_HYPO_NYMS'] = app.config['FEATURES']['INCLUDE_HYPER_HYPO_NYMS']
        features['INCLUDE_RELATED'] = app.config['FEATURES']['INCLUDE_RELATED']
    
    # Set variables for  conceptnet language, nlp model, stop words and domain data based on the input language
    if lang == 'en':
        concept_net_language = app.config['CONCEPT_NET_EN']
        stop_words = app.config['STOP_WORDS_EN']
        nlp = SPACY_PACKAGE_EN
        domain_data_json = domain_data['en']
    elif lang == 'fr':
        concept_net_language = app.config['CONCEPT_NET_FR']
        stop_words = app.config['STOP_WORDS_FR']
        nlp = SPACY_PACKAGE_FR
        domain_data_json = domain_data['fr']
    elif lang == 'de':
        concept_net_language = app.config['CONCEPT_NET_DE']
        stop_words = app.config['STOP_WORDS_DE']
        nlp = SPACY_PACKAGE_DE
        domain_data_json = domain_data['de']

    # sentence_nlp contains sentence with applied nlp from spacy
    sentence_nlp = nlp(sentence.lower())
    # tokens_list will contain the list of valid tokens extracted from the sentence
    tokens_list = []

    # this loop iterates for every token and checks if it is has a valid POS and is not a stop word
    for token in sentence_nlp:
        if (token.pos_ in app.config['ALLOWED_POS']) and (token.text not in stop_words):
            tokens_list.append(token.text)
        #logging.info (token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)

    # if no token present in sentence then return error
    if not tokens_list:
        return jsonify({"status":"error", "result":"No valid word found to find synonym"}),501

    # for structuring the output in list format that will be used in creating paraphrases
    synonymList = []
    # domain_synonyms contains synonyms extracted from domain file wrt sentence
    domain_synonyms = {}

    #listOfSentence = []
    listOfWords = []
    for token in sentence_nlp:
        listOfWords.append(token.text)
    lenOfSentence = len(listOfWords)
    extractLength = lenOfSentence
    while(extractLength>0):
        startLength = 0
        while(startLength<=(lenOfSentence-extractLength)):
            phrase = ' '.join(listOfWords[startLength:(extractLength+startLength)])
            #listOfSentence.append(phrase)
            phraseData = domain_data_json.get(phrase, None)
            substringDomain = any(phrase in string for string in list(domain_synonyms.keys()))
            if phraseData != None and substringDomain != True:
                domain_synonyms[phrase] = phraseData
            startLength = startLength+1
        extractLength = extractLength-1

    #logging.info('listOfSentence', listOfSentence)

    #logging.info ("DOMAIN------>", domain_synonyms)

    synonyms = {}
   
    # to check to get data from domain or api or both
    if app.config['BOTH_API_AND_DOMAIN'] == 'api':
       synonyms = get_synonyms_from_conceptnet(tokens_list, concept_net_language, features)
    elif app.config['BOTH_API_AND_DOMAIN'] == 'domain':
        for key in domain_synonyms.keys():
            synonyms[key] = domain_synonyms[key]
    else:
        # synonyms in format {'f2f': ['face2face', 'face to face'], 'mea': ['month end adjustment', 'month end']}
        synonyms = get_synonyms_from_conceptnet(tokens_list, concept_net_language, features)
        #logging.info (synonyms)
        for key in domain_synonyms.keys():
            if key in synonyms.keys():
                synonyms[key] = synonyms[key] + list(set(domain_synonyms[key]) - set(synonyms[key]))
            else:
                synonyms[key] = domain_synonyms[key]

    logging.info(domain_synonyms)

    for key in synonyms.keys():
        listtemp = []
        listtemp.append(key)
        listtemp.append(synonyms[key])
        synonymList.append(listtemp)

    #logging.info ("synonymlist.......", synonymList)
    return jsonify({'synonymList' : synonymList}), 201



@app.route('/buildparaphrases', methods = ['POST'])
def build_paraphrases():
    # check the input recieved is as per required
    try:
        input = request.json
        logging.info (input)
    except Exception as ex:
        return jsonify({"status":"error","result":"", "message" : "Request received in a format other than JSON"}),501
    if "sentence" in list(input.keys()) and "synonymList" in list(input.keys()):
        sentence = input["sentence"]
        synonymList = input["synonymList"]
    else:
        return jsonify({"status":"error","result":"Input sentence not recieved"}),501

    # convert the sentence to lowercase
    sentence = sentence.lower()
    # paraphrases will contain the list of all paraphrases created based on the list of synonyms
    paraphrases = []
    # appends the sentence to the paraphrases list
    paraphrases.append(sentence)
    for phrase in synonymList:
        tempSent = []
        for sent in paraphrases:
            if phrase[0] in sent:
                for pp in phrase[1]:
                    newSent = re.sub(r'\b'+phrase[0]+r'\b',pp,sent) #we make sure that we replace only complete words
                    tempSent.append(newSent)
        for s in tempSent:
            paraphrases.append(s)
    paraphrases = list(OrderedDict.fromkeys(paraphrases)) #delete duplicates
    #logging.info (paraphrases)
    return jsonify({'paraphrases' : paraphrases}),201










if __name__=='__main__':
    excel.init_excel(app)
    # app.run(debug=True,host='0.0.0.0', port=4321,threaded = True)

    try:
        # print(" Debug 1 ")
        domain_data = read_csv(app.config['DOMAIN_FILENAME'])
        # print("domain_data:", domain_data)
    except Exception as ex:
        logging.info (ex)
        raise Exception("cannot access domain file ", config['DOMAIN_FILENAME'])

    logging.info ("Starting Paraphrase service on port: " + str(5678))
    app.run(debug=True,host="0.0.0.0",port=5678,threaded=True)


## ---------------------------------------------------------------------------
# Sample Input
# {
#     "questions": [" Tell me about leave policies ", " What is a casual leave?"], 
#     "flag": "0"  
# }
#

# @ findsynonyms
# {
# "sentence": "Tell me about leave policies", 
# "lang": "en",
# "features": {"antonym": "false", "hyper_hypo_nyms": "false", "related": "true"}
# }

# @ buildparaphrase
# {
#     "sentence": "Tell me about leave policies", 
#      "synonymList": [
#         [
#             "leave",
#             [
#                 "allow",
#                 "allow for",
#                 "bequeath",
#                 "leave alone",
#                 "result",
#                 "go",
#                 "leave behind",
#                 "lead",
#                 "pull up stakes",
#                 "let alone",
#                 "entrust",
#                 "go away",
#                 "provide",
#                 "go forth",
#                 "leave of absence",
#                 "impart",
#                 "exit",
#                 "not stay",
#                 "forget",
#                 "depart",
#                 "farewell",
#                 "going",
#                 "not come"
#             ]
#         ],
#         [
#             "tell",
#             [
#                 "tell",
#                 "not ask",
#                 "assure",
#                 "narrate",
#                 "explain",
#                 "recite",
#                 "distinguish",
#                 "say",
#                 "evidence",
#                 "william tell",
#                 "state",
#                 "order",
#                 "recount"
#             ]
#         ],
#         [
#             "policies",
#             [
#                 "orange order",
#                 "policymaker",
#                 "majorism",
#                 "policy mix",
#                 "manifesto",
#                 "policy",
#                 "putinism",
#                 "viaticals",
#                 "politically correct",
#                 "pollutician",
#                 "social conservative",
#                 "policymaking",
#                 "chief executive officer",
#                 "political risk",
#                 "hansonism",
#                 "uprising",
#                 "grassroots democracy",
#                 "thanatocracy",
#                 "new wine in old wineskins",
#                 "unelectable",
#                 "neo mccarthyism",
#                 "saddamist",
#                 "board of directors",
#                 "harperite",
#                 "brownite",
#                 "death futures"
#             ]
#         ],
#         [
#             "tell me about",
#             [
#                 "what do you mean by",
#                 "give me some information on",
#                 "can you tell me about",
#                 "do you know about",
#                 "what is",
#                 "what are",
#                 "define"
#             ]
#         ]
#     ]

# }