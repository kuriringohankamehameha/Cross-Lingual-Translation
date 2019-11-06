import ibm_1
import naive_ibm
import time
import os
DATA_PATH = '/home/tobirama/3-1/academics/ir/ass/Cross-Lingual-Translator'
BASE_PATH = os.path.join(DATA_PATH, 'dataset')
OUTPUT_PATH = os.path.join(os.getcwd(), 'Output') + '/'
ENG_OFFSET = "English_"
DUT_OFFSET = "Dutch_"
EXT = ".txt"
eng_list = []
dut_list = []
ENG_NAME = "English"
DUT_NAME = "Dutch"

#for i in range(0, 11):
#    eng_list.append(BASE_PATH + ENG_OFFSET + str(i) + EXT)
#    dut_list.append(BASE_PATH + DUT_OFFSET + str(i) + EXT)
eng_list.append(BASE_PATH + '/' + ENG_NAME + EXT)
dut_list.append(BASE_PATH + '/' + DUT_NAME + EXT)

assert len(eng_list) == len(dut_list)

ibm_model = ibm_1.IBM(1.0)
#ibm_model = naive_ibm.IBM(1.0)
print('Using IBM Model', ibm_model.__version__)

OUTPUT_FILE = OUTPUT_PATH + 'output.txt'

count = 0
for efile, dfile in zip(eng_list, dut_list):
    print('Processing Batch #%i' % count)
    biwords = ibm_model.get_biwords(efile, dfile)
    start = time.time()
    if count == 0:
        prob_dict = ibm_model.init_translation_probabilities(biwords)
        counter = None
        total = None
        total_s = None
    else:
        OUTPUT_FILE1 = OUTPUT_PATH + 'prob_' + str(count-1) + '.txt'
        OUTPUT_FILE2 = OUTPUT_PATH + 'count_' + str(count-1) + '.txt'
        OUTPUT_FILE3 = OUTPUT_PATH + 'total_' + str(count-1) + '.txt'
        OUTPUT_FILE4 = OUTPUT_PATH + 'total_s_' + str(count-1) + '.txt'
        prob_dict, counter, total, total_s = ibm_model.get_prob_dict_from_file(OUTPUT_FILE1, OUTPUT_FILE2, OUTPUT_FILE3, OUTPUT_FILE4)
        print('Initialization Time:', round(time.time() - start, 3))
    #print('Test alignments')
    #for j in range(10):
    #    ee, dd = biwords[j]
    #    print(ee, dd)
    #    print(ibm_model.do_alignment(ee, dd, prob_dict))
        #print(' '.join(ee[i] for i in ibm_model.do_alignment(ee, dd, prob_dict)))
    #ee = ['new','year']
    #print(' '.join(ee[i] for i in ibm_model.do_alignment(ee, ['nieuwjaar'], prob_dict)))
    #break
    start = time.time()
    counter, total, total_s = ibm_model.train_em(biwords, prob_dict, counter, total, total_s, 15,
    one_to_many=True)
    print('Training Time:', round(time.time() - start, 3))

    OUTPUT_FILE1 = OUTPUT_PATH + 'prob_' + str(count) + '.txt'
    OUTPUT_FILE2 = OUTPUT_PATH + 'count_' + str(count) + '.txt'
    OUTPUT_FILE3 = OUTPUT_PATH + 'total_' + str(count) + '.txt'
    OUTPUT_FILE4 = OUTPUT_PATH + 'total_s_' + str(count) + '.txt'
    ibm_model.log_output(biwords, prob_dict, counter, total, total_s, OUTPUT_FILE1 , OUTPUT_FILE2, OUTPUT_FILE3, OUTPUT_FILE4, 0.05)
    count += 1
