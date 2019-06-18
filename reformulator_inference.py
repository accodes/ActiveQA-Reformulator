from px.nmt import reformulator
from px.proto import reformulator_pb2

# model_checkpoint_path = './tmp/active-qa/translate.ckpt-1460356'
questions = [ ' How many casual leaves one is entitled in a year?'] ## tell me about leave policies?

reformulator_instance = reformulator.Reformulator(
    hparams_path='px/nmt/example_configs/reformulator.json',
    source_prefix='<en> <2en> ',
    out_dir='./tmp/active-qa/reformulator/',
    environment_server_address= 'localhost:10000') ## 'localhost:10000'

# Change from GREEDY to BEAM if you want 20 rewrites instead of one.
responses = reformulator_instance.reformulate(
    questions=questions,
    inference_mode=reformulator_pb2.ReformulatorRequest.BEAM_SEARCH)

# Since we are using greedy decoder, keep only the first rewrite.
# reformulations = [r[0].reformulation for r in responses]
reformulations = [[rf.reformulation for rf in rsp] for rsp in responses]

for i in range(len(reformulations)):
	print('----------------------------------')
	print('The reformulations of "', questions[i], '" are:', reformulations[i])











# print ('reformulations:', reformulations)

###-------------------------------------------------------------------------------------
 # custom_questions = ['How can i apply for nsa?']
 #  responses = reformulator_instance.reformulate(
 #      questions=custom_questions,
 #      inference_mode=reformulator_pb2.ReformulatorRequest.GREEDY)

 #  # Discard answers.
 #  custom_reformulations = [[rf.reformulation for rf in rsp] for rsp in responses]