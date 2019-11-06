from collections import defaultdict, OrderedDict
from itertools import islice
import re
from math import log
import time


class IBM():
    """
        IBM Model 1 for Translation and Alignment
        1. Initialize t(e|f), possibly randomly
        2. Normalize the random values for all e's for a particular f
        3. Consider all Eng-Dutch word pairs that occur in the same sentence
        pair.
        4. Perform #EM Iterations to get good values of the translation
        probablilities t(e|f) for use in further IBM Models
    """
    def __init__(self, version):
        self.__version__ = version

    def get_biwords(self, file1, file2):
        biwords = [[re.sub(r'[^\w]', ' ', sentence).lower().split()
                    for sentence in pair]
                   for pair in zip(open(file1), open(file2))]
        return biwords

    def init_translation_probabilities(self, biwords=None):
        t = defaultdict(lambda: defaultdict(float))
        for (e_sent, d_sent) in biwords:
            d_sent.append("NULL")
            #e_sent.append("NULL")
            length = len(e_sent)
            for e in set(e_sent):
                for d in set(d_sent):
                    t[e][d] = 1.0 / length
        return t

    def update_translation_probabilities(self, t, biwords=None):
        temp = defaultdict(lambda: defaultdict(float))
        for key in t:
            for nkey in t[key]:
                temp[key][nkey] = t[key][nkey]
        for (e_sent, d_sent) in biwords:
            d_sent.append("NULL")
            length = len(e_sent)
            for e in set(e_sent):
                for d in set(d_sent):
                    t[e][d] = 1.0 / length

            for nkey in temp[key]:
                t[key][nkey] = temp[key][nkey]
        del temp

    def get_alignments(self, e_sent, d_sent, t, q):
        def max_alignment_index(idx):
            alignment_candidates = [(i, t[e_sent[idx]][d_sent[i]]) for i in range(len(d_sent))]
            return max(alignment_candidates, key=lambda x: x[1])[0]
        op = []
        arg_max = -1
        d_sent.append('NULL')
        for j,e in enumerate(e_sent):
            p = 0
            for i,d in enumerate(d_sent):
                if q[(i, j, len(e_sent), len(d_sent)-1)]*t[e][d] > p:
                    p = q[(i, j, len(e_sent), len(d_sent)-1)]*t[e][d]
                    #print(e, d, p)
                    arg_max = i
            if arg_max != -1:
                op.append(arg_max)
                arg_max = -1
        return op

    def do_alignment(self, e_sent, d_sent, t, to_lang='eng'):
        # For every english/dutch word, pick whichever index
        # has the maximum probability
        temp = defaultdict(lambda: defaultdict(float))
        #print(t['resumption']['hervatting'])
        if to_lang == 'eng':
            d_sent.append('NULL')
            for e in e_sent:
                for d in d_sent:
                    temp[e][d] = t[e][d]
            def max_alignment_index(idx):
                alignment_candidates = [(i, temp[e_sent[idx]][d_sent[i]]) for i in range(len(d_sent))]
                #print(e_sent[idx], max(alignment_candidates, key=lambda x:x[1])[0])
                return max(alignment_candidates, key=lambda x: x[1])[0]
            op = []
            for i in range(len(e_sent)):
                op.append(max_alignment_index(i))
                #del temp[e_sent[max_alignment_index(i)]]
            del temp
            # TODO: Update procedure for reverse translation
            #print(op)
            #return op
            return [v for i, v in enumerate(op) if i == 0 or v != op[i-1]]
            #return OrderedDict.fromkeys(op).keys()
            #return [max_alignment_index(i) for i in range(len(d_sent))]
        else:
            def max_alignment_index(idx):
                alignment_candidates = [(i, t[e_sent[idx]][d_sent[i]]) for i in range(len(d_sent))]
                return max(alignment_candidates, key=lambda x: x[1])[0]
            return [max_alignment_index(i) for i in range(len(e_sent))]

    def one_to_many_translation(self, e_sent, d_sent, t, to_lang='eng'):
        if to_lang == 'eng':
            #Translate nieuwjaar to new year
            proximity = 0.1
            e_len = len(e_sent)
            for d in d_sent:
                for i in range(e_len - 1):
                    for j in range(i+1,e_len):
                        #If two words have similar probability, merge them
                        v = t[e_sent[i]][d]
                        u = t[e_sent[j]][d]
                        if abs(v - u) <= proximity and min(u, v) >= 0.1:
                            t[' '.join([e_sent[i], e_sent[j]])][d] = max(v, u) + 0.05
        else:
            raise NotImplementedError

    def align_sentence(self, e_sent, d_sent, t, q, to_lang='eng'):
        if to_lang == 'eng':
            # Get the dutch to english alignments
            d_to_e = []
            e_len = len(e_sent)
            d_len = len(d_sent)
            for i, d in enumerate(d_sent):
                probable_alignment = 0
                max_prob = 0
                for j, e in enumerate(e_sent):
                    if q[(j, i, e_len, d_len)] > 0 and e in t and d in t[e]:
                        #print(q[(j, i, e_len, d_len)], e)
                        temp = q[(j, i, e_len, d_len)] * t[e][d]
                    else:
                        temp = -1
                    if temp > max_prob:
                        max_prob = temp
                        probable_alignment = j
                d_to_e.append(probable_alignment)
            print(' '.join(e_sent[i] for i in d_to_e))
            return d_to_e
        else:
            raise NotImplementedError

    def align_and_predict(self, e_sent, d_sent, t, q, to_lang='eng'):
        if to_lang == 'eng':
            e_len = len(e_sent)
            d_len = len(d_sent)
            a = self.align_sentence(e_sent, d_sent, t, q)
            prob = 0
            assert d_len == len(a)
            #for i in islice(q, 10):
            #    print(i, q[i])
            for i, d in enumerate(d_sent):
                e = e_sent[a[i]]
                if q[(a[i], i, e_len, d_len)] > 0 and e in t and d in t[e]:
                    try:
                        prob += abs(log(q[(a[i], i, e_len, d_len)] * t[e][d]))
                    except ValueError:
                        print("---ValueError at---")
                        print(a[i], i, e_len, d_len)
                        print(e)
                        print(d)
                        print(t[e][d])
                        print("-----------------")
                        prob += -500
                else:
                    prob += -500
            return prob
        else:
            raise NotImplementedError

    def align_corpus(self, e_sent, d_sent, t, q, align_lang='eng'):
        if align_lang == 'eng':
            for d in d_sent:
                max_idx = -1
                max_prob = 0
                for i, e in enumerate(e_sent):
                    prob = self.align_and_predict(e_sent, d_sent, t, q)
                    if i == 0:
                        max_prob = prob
                        max_idx = i
                    if prob > max_prob:
                        max_prob = prob
                        max_idx = i
                return ' '.join(e[max_idx][1:])
        else:
            raise NotImplementedError

    def train_em(self, biwords, t, count=None, total=None, total_s=None, q=None, num_loops=1000, one_to_many=False):
        # Performs the EM Algorithm on the IBM Model
        i = 10
        curr_iteration = 1
        # Until convergence ...
        if count is None or total is None or total_s is None:
            count = defaultdict(lambda: defaultdict(float))
            total = defaultdict(float)
            total_s = defaultdict(float)
            param1 = defaultdict(float)
            param2 = defaultdict(float)
        if q is None:
            q = defaultdict(float)
        param1 = defaultdict(float)
        param2 = defaultdict(float)
        for (e_sent, d_sent) in biwords:
            eng_len = len(e_sent)
            dut_len = len(d_sent)
            for i in range(eng_len+1):
                for j in range(dut_len+1):
                    if q[(j, i, eng_len, dut_len)] == 0:
                        q[(j, i, eng_len, dut_len)] = 1/float(dut_len + 1)
        while i > 0.01 and curr_iteration <= num_loops:
            start = time.time()
            print('Loop #%i' % curr_iteration)
            for k, (e_sent, d_sent) in enumerate(biwords):
                eng_len = len(e_sent)
                dut_len = len(d_sent)
                for i, e in enumerate(e_sent):
                    # Update total translation prob for every English word
                    # for normalizing the probabilities in the next step
                    total_s[e] = 0.0
                    for j, d in enumerate(d_sent):
                        if t[e][d] == 0.0:
                            t[e][d] = 1/len(e_sent)
                        total_s[e] += t[e][d]*q[(j, i, eng_len, dut_len)]
                for i,e in enumerate(e_sent):
                    for j,d in enumerate(d_sent):
                        # Get the normalized counts for every (e, d) pair
                        delta = (t[e][d] / total_s[e]) * (q[(j, i, eng_len, dut_len)])
                        count[e][d] += delta
                        total[d] += delta
                        param1[(j, i, eng_len, dut_len)] += delta
                        param2[(i, eng_len, dut_len)] += delta

            i = 0
            t_new = defaultdict(lambda: defaultdict(float))
            for (e_sent, d_sent) in biwords:
                for e in e_sent:
                    for d in d_sent:
                        t_new[e][d] = count[e][d] / total[d]
                        temp = abs(t_new[e][d] - t[e][d])
                        if i < temp:
                            i = temp
                        t[e][d] = count[e][d] / total[d]

            for (j, i, eng, dut) in param1.keys():
                #try:
                q[(j, i, eng, dut)] = param1[(j, i, eng, dut)]/param2[(i, eng, dut)]
                #print(j, i, eng, dut, q[(j, i, eng, dut)])
                #except ZeroDivisionError:
                #    print(i, eng, dut, param2[(i, eng, dut)])
                #    continue
            curr_iteration += 1
            end = time.time()
            print('Time Elapsed :', round(end - start, 3))
        return count, total, total_s, q

    def print_probabilities(self, biwords, t, threshold=0.07):
        for (e_sent, d_sent) in biwords:
            for (i, d) in enumerate(d_sent):
                for (j, e) in enumerate(e_sent):
                    if t[(e, d)] > threshold:
                        print("(" + e + ',' + d +
                              ') -> ' + str(round(t[(e, d)], 3)))

    def log_output(self, biwords, t, count, total, total_s, q, OUTPUT_FILE_1, OUTPUT_FILE_2, OUTPUT_FILE_3, OUTPUT_FILE_4, OUTPUT_FILE_5, threshold=0.07):
        start = time.time()
        with open(OUTPUT_FILE_1, 'w') as f1, open(OUTPUT_FILE_2, 'w') as f2, open(OUTPUT_FILE_3, 'w') as f3, open(OUTPUT_FILE_4, 'w') as f4:
            #f.write('ENG,DUT,PROB\n')
            for (e_sent, d_sent) in biwords:
                gt_than = False
                for i, d in enumerate(d_sent):
                    dut_len = len(d)
                    for j, e in enumerate(e_sent):
                        eng_len = len(e)
                        if t[e][d] > threshold:
                            f1.write(e + ',' + d + ',' + str(round(t[e][d], 3)) + '\n')
                            f2.write(e + ',' + d + ',' + str(round(count[e][d], 3)) + '\n')
                            f4.write(e + ',' + str(round(total_s[e],3)) + '\n')
                            # To avoid duplicates
                            t[e][d] = threshold
                            gt_than = True
                    if gt_than:
                        f3.write(d + ',' + str(round(total[d], 3)) + '\n')
                        gt_than = False
            for i, e in enumerate(t):
                eng_len = len(e)
                for j, d in enumerate(t[e]):
                    dut_len = len(d)
                    if t[e][d] > threshold:
                            f1.write(e + ',' + d + ',' + str(round(t[e][d], 3)) + '\n')
                            f2.write(e + ',' + d + ',' + str(round(count[e][d], 3)) + '\n')
                            f4.write(e + ',' + str(round(total_s[e],3)) + '\n')
                            # To avoid duplicates
                            t[e][d] = threshold
                            gt_than = True
                    if gt_than:
                        gt_than = False
                        f3.write(d + ',' + str(round(total[d], 3)) + '\n')
        with open(OUTPUT_FILE_5, 'w') as f5:
            for (i, j, eng_len, dut_len) in q.keys():
                f5.write(str(i) + ',' + str(j) + ',' + str(eng_len) + ',' + str(dut_len) + ',' + str(round(q[(i, j, eng_len, dut_len)], 3)) + '\n')
        print('Time Elapsed for IO:', round(time.time() - start, 3))

    def get_prob_dict_from_file(self, INPUT_FILE_1, INPUT_FILE_2, INPUT_FILE_3, INPUT_FILE_4, INPUT_FILE_5=None):
        t = defaultdict(lambda: defaultdict(float))
        count = defaultdict(lambda: defaultdict(float))
        total = defaultdict(float)
        total_s = defaultdict(float)
        q = defaultdict(float)
        with open(INPUT_FILE_1, 'r') as f1, open(INPUT_FILE_2, 'r') as f2, open(INPUT_FILE_3, 'r') as f3, open(INPUT_FILE_4, 'r') as f4, open(INPUT_FILE_5, 'r') as f5:
            for line1, line2, line3, line4, line5 in zip(f1, f2, f3, f4, f5):
                ls = line1.strip().split(',')
                assert len(ls) == 3
                t[ls[0]][ls[1]] = float(ls[2])
                ls = line2.strip().split(',')
                count[ls[0]][ls[1]] = float(ls[2])
                ls = line3.strip().split(',')
                total[ls[0]] = float(ls[1])
                ls = line4.strip().split(',')
                total_s[ls[0]] = float(ls[1])
                ls = line5.strip().split(',')
                q[(int(ls[0]), int(ls[1]), int(ls[2]), int(ls[3]))] = float(ls[4])
        return t, count, total, total_s, q

    def get_prob_dict_from_file_no_q(self, INPUT_FILE_1, INPUT_FILE_2, INPUT_FILE_3, INPUT_FILE_4):
        t = defaultdict(lambda: defaultdict(float))
        count = defaultdict(lambda: defaultdict(float))
        total = defaultdict(float)
        total_s = defaultdict(float)
        with open(INPUT_FILE_1, 'r') as f1, open(INPUT_FILE_2, 'r') as f2, open(INPUT_FILE_3, 'r') as f3, open(INPUT_FILE_4, 'r') as f4:
            for line1, line2, line3, line4 in zip(f1, f2, f3, f4):
                ls = line1.strip().split(',')
                assert len(ls) == 3
                t[ls[0]][ls[1]] = float(ls[2])
                ls = line2.strip().split(',')
                count[ls[0]][ls[1]] = float(ls[2])
                ls = line3.strip().split(',')
                total[ls[0]] = float(ls[1])
                ls = line4.strip().split(',')
                total_s[ls[0]] = float(ls[1])
        return t, count, total, total_s

    def test_data(self, test_sentence, t, to_lang='dutch'):
        output = []
        if to_lang == 'dutch':
            for word in re.sub(r'[^\w]', ' ', test_sentence).lower().split():
                try:
                    output.append(max(t[word], key=t[word].get))
                except ValueError:
                    print('Error: No matching translation for', word)
                    output.append('__NOMATCH__')
                    continue
            return ' '.join(output)
        elif to_lang == 'english':
            for word in re.sub(r'[^\w]', ' ', test_sentence).lower().split():
                max_key = None
                max_prob = -1
                for e in t:
                    if max_key is None:
                        max_prob = t[e][word]
                        max_key = e
                    else:
                        if t[e][word] > max_prob:
                            max_prob = t[e][word]
                            max_key = e
                if max_prob > 0:
                    output.append(max_key)
                else:
                    print('Error: No matching translation for', word)
                    output.append('__NOMATCH__')
            return ' '.join(output)
        else:
            return None
