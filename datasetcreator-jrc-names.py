import datetime, os, shutil, csv, itertools, random, os.path
from alphabet_detector import AlphabetDetector
from random import shuffle
from whoosh.analysis import  NgramFilter, RegexTokenizer
from whoosh.fields import SchemaClass, TEXT, ID
from whoosh.index import create_in, open_dir
from whoosh.qparser.default import MultifieldParser
from whoosh import qparser
from progressbar import Bar, ETA, Percentage, ProgressBar

class SampleSchema(SchemaClass):
    id = ID(stored=True, unique=False)
    title = TEXT(stored=True,analyzer=RegexTokenizer() | NgramFilter(4))

#Name: create_whoosh_index
#Inputs: * filename_in
#Must do: For a given file creates n reversed search index using SampleSchema class
def create_whoosh_index(filename_in):
    l_max_val = 0
    if filename_in == "jrc_person":
        l_max_val = 620000
    else:
        l_max_val = 43000

    widgets = ['Index build progress:' + filename_in, Percentage(), ' ', Bar(marker='0', left='[', right=']'),' ', ETA(), ' ']
    pbar_indx = ProgressBar(widgets=widgets, maxval=l_max_val)
    pbar_indx.start()

    if not os.path.exists("index_"+filename_in):
            os.mkdir("index_"+filename_in)

    with open(filename_in+".csv", encoding="utf8") as csvfile:
        l_row_ix = 1
        l_curr_ix = 1
        reader = csv.DictReader(csvfile, fieldnames=["ID","name"], delimiter='|')
        ix = create_in("index_"+ filename_in, SampleSchema)
        writer = ix.writer()
        for row in reader:
            l_row_ix += 1
            if l_row_ix % 50000 == 0:
                print("Index build" + str(l_row_ix))
                pbar_indx.update(l_row_ix)
            writer.add_document(id=row["ID"], title=row["name"])
    writer.commit()

    pbar_indx.finish()

#Name: search_whoosh_files
#Inputs: * filename_in
#Must do: Searches the jrc entity file for non matching pairs using 4 n grams filters
#         For each entry uses at most 3 non matching pairs
def search_whoosh_files (filename_in):
    ngt1 = RegexTokenizer() | NgramFilter(4)
    l_aux_i=0
    filename_aux = "dataset_match_" + filename_in
    ix1 = open_dir("index_" + filename_in)

    #aux max val for progress bar
    if filename_in == "jrc_person":
        max_val = 3000000
    else:
        max_val = 3000000

    widgets = ['Progress Searching ' + filename_in + ': ', Percentage(), ' ', Bar(marker='0', left='[', right=']'),' ', ETA(), ' ']
    pbar = ProgressBar(widgets=widgets, maxval=max_val) #454000
    pbar.start()

    with ix1.searcher() as searcher:
        parser = MultifieldParser(['title'], ix1.schema)
        parser.remove_plugin_class(qparser.WildcardPlugin)
        parser.remove_plugin_class(qparser.PlusMinusPlugin)
        with open("dataset_non_match_"+filename_in+".csv_tmp", 'w', encoding="utf-8") as inW2:
            with open("dataset_match_" + filename_in + ".csv"  , encoding="utf8") as csvfile:
                for row in csvfile:
                    l_aux_i = l_aux_i + 1
                    if l_aux_i % 20000 == 0:
                        print ("Index search" + str(l_aux_i))
                        pbar.update(l_aux_i)
                    l_row_idx = row.split('|')[0]
                    l_row_aux = row.split('|')[1]
                    search_list = [token.text for token in ngt1(l_row_aux)]
                    if len(search_list)>0:
                        l_row_str = random.sample(search_list, 1)
                        query = parser.parse(l_row_str[0])
                        results = searcher.search(query)
                        results_aux = []
                        for result in results:
                            if result['id'] != l_row_idx:
                                results_aux.append([result['id'], result['title']])
                        if len(results_aux)>0:
                            shuffle(results_aux)
                            line_new = l_row_idx + "|" + l_row_aux + "|" + results_aux[0][0] + "|" + results_aux[0][1]
                            inW2.write(line_new.strip() + '\n')
                            if len(results_aux)>1:
                                if results_aux[1][0] != results_aux[0][0]:
                                    line_new = l_row_idx + "|" + l_row_aux + "|" + results_aux[1][0] + "|" + results_aux[1][1]
                                    inW2.write(line_new.strip() + '\n')
                            if len(results_aux)>2:
                                if results_aux[2][0] != results_aux[1][0]:
                                    line_new = l_row_idx + "|" + l_row_aux + "|" + results_aux[2][0] + "|" + results_aux[2][1]
                                    inW2.write(line_new.strip() + '\n')
        pbar.finish()


# Name: match_shuffle_dataset
# Inputs: * filename_in
# Must do: Shuffles the generated dataset
#          - removes duplicates pairs
#          - shuffles the dataset order
def match_shuffle_dataset(filename_in):
    l_list_random = []
    i = 0
    with open("dataset_non_match_" + filename_in + ".csv", 'w', encoding="utf-8") as inW:
        with open("dataset_non_match_" + filename_in + ".csv_tmp", encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=["ID1","NAME1","ID2","NAME2"], delimiter='|')
            for row in reader:
                i = i+1
                if i%50000 == 0:
                    print (filename_in + " " + str(i))
                if int(row["ID1"]) < int(row["ID2"]):
                    l_line = row["ID1"] + "|" + row["NAME1"] + "|" + row["ID2"] + "|" + row["NAME2"]
                else:
                    l_line = row["ID2"] + "|" + row["NAME2"] + "|" + row["ID1"] + "|" + row["NAME1"]
                l_list_random.append(l_line.strip()+"\n")
            print ("Setting")
            shuffle(l_list_random)
            l_list_random_set = set(l_list_random)
            inW.writelines(l_list_random_set)

#Name: dataset_split
#Inputs: * filename_in
#        * type_in
#Must do:  Depending on type_in (O,P) generates a new file with the corresponding entity names
def split_dataset(filename_in, type_in):
    with open(filename_in+".csv", 'w', encoding="utf-8") as inW:
        with open("entities_jrc", encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile,fieldnames=["ID","TYPE","t3","name"], delimiter='\t')
            for row in reader:
                l_curr_line_idx = row["ID"]
                l_curr_line_name = row["name"]
                l_curr_line_type = row["TYPE"]
                if l_curr_line_type == type_in:
                    l_print_line = l_curr_line_idx.strip() + "|" + l_curr_line_name.strip()
                    inW.write(l_print_line + '\n')

# Name: match_joins_1liner
# Inputs: * filename_in
# Must do: Merges each matching id into a 1liner
def match_joins_1liner(filename_in):
    l_prev_line_idx = ""
    l_print_line = ""
    with open(filename_in + "_1liner.csv_tmp", 'w', encoding="utf-8") as inW:
        with open(filename_in + ".csv", encoding="utf8") as csvfile:
            reader = csv.DictReader(csvfile, fieldnames=["ID", "name"], delimiter='|')
            for row in reader:
                l_curr_line_idx = row['ID']
                l_curr_line_name = row['name']
                if l_prev_line_idx == "":
                    l_prev_line_idx = l_curr_line_idx
                    l_print_line = l_curr_line_idx
                if l_curr_line_idx == l_prev_line_idx:
                    l_print_line = l_print_line.rstrip() + "|" + l_curr_line_name
                else:
                    inW.write(l_print_line + "\n")
                    l_prev_line_idx = l_curr_line_idx
                    l_print_line = l_curr_line_idx + "|" + l_curr_line_name

# Name: match_generate_dataset
# Inputs: * filename_in
# Must do: Generates ordered dataset of matching pairs
def match_generate_dataset(filename_in):
    with open(filename_in + "_matching.csv_tmp", 'w', encoding="utf-8") as inW:
        with open(filename_in + "_1liner.csv_tmp", encoding="utf8") as csvfile:
            for row in csvfile:
                l_tabcount = row.count("|")  # counts nr of elements
                if l_tabcount > 1:  # at least 2 elements
                    l_row_aux = row.split('|')[1:]
                    l_row_idx = row.split('|')[0]
                    l_list_inter = []
                    for l_i in range(l_tabcount):
                        l_list_inter.append(l_i)
                    l_list_pairs = list(itertools.combinations(l_list_inter, 2))  # combinations 2 by 2
                    for l_print_line_aux in l_list_pairs:
                        if l_row_aux[l_print_line_aux[0]] != l_row_aux[l_print_line_aux[1]]:
                            l_print_line = l_row_idx + "|" + l_row_aux[l_print_line_aux[0]] + "|" + \
                                           l_row_aux[l_print_line_aux[1]]
                            inW.write(l_print_line.strip('\n') + '\n')


# Name: match_shuffle_dataset
# Inputs: * filename_in
# Must do: Shuffles the generated dataset
#          - randomizes order of values in matching pairs (coin flip)
#          - shuffles the dataset order
def match_shuffle_dataset(filename_in):
    l_list_random = []
    with open("dataset_match_" + filename_in + ".csv", 'w', encoding="utf-8") as inW:
        with open(filename_in + "_matching.csv_tmp", encoding="utf8") as csvfile:
            for row in csvfile:
                l_row_idx = row.split('|')[0].strip('\n')
                l_row_1st = row.split('|')[1].strip('\n')
                l_row_2nd = row.split('|')[2].strip('\n')
                l_coinflip = random.randint(0, 1)
                if l_coinflip == 1:
                    l_list_random.append(l_row_idx + '|' + l_row_1st + '|' + l_row_2nd)
                else:
                    l_list_random.append(l_row_idx + '|' + l_row_2nd + '|' + l_row_1st)
            shuffle(l_list_random)
            for l_print_line in l_list_random:
                inW.write(l_print_line.strip('\n') + '\n')

#Name: moves move_files
#Inputs: * dest_in
#Must do: moves temporary files to backup table
def move_files(dest_in):
    if not os.path.exists(dest_in):
        os.mkdir(dest_in)
    l_files = os.listdir(os.getcwd())
    for f in l_files:
        if f.endswith("_tmp"):
            shutil.move(f, os.getcwd() + "/" + dest_in)

#Name: mesh_dataset, size_in
#Inputs: * filename_in
#Must do: Generates final dataset
def mesh_dataset(filename_in, size_in):
    end_list = []
    ad = AlphabetDetector()
    dict_lists = {}
    match_max_len = 0
    match_min_len = 0
    match_total_len = 0
    match_total_rows = 0
    nomatch_max_len = 0
    nomatch_min_len = 0
    nomatch_total_len = 0
    nomatch_total_rows = 0
    l_index = 0

    with open("dataset_match_" + filename_in + ".csv", encoding="utf8") as csvfile1:
        reader = csv.DictReader(csvfile1, fieldnames=["ID1", "NAME1", "NAME2"], delimiter='|')
        for row in reader:
            l_index = l_index + 1
            alpha = []
            alpha_1 = ad.detect_alphabet(row["NAME1"].strip())
            alpha_2 = ad.detect_alphabet(row["NAME2"].strip())
            for item in alpha_1:
                if 'KATAKANA' in item or 'HANGUL' in item or 'HIRAGANA' in item:
                    aux_item = 'CJK'
                else:
                    aux_item = item
                if not aux_item in alpha:
                    alpha.append(aux_item)
            for item in alpha_2:
                if 'KATAKANA' in item or 'HANGUL' in item or 'HIRAGANA' in item:
                    aux_item = 'CJK'
                else:
                    aux_item = item
                if not aux_item in alpha:
                    alpha.append(aux_item)

            len_1 = len(row["NAME1"])
            len_2 = len(row["NAME2"])
            if len_1 > match_max_len or match_max_len == 0:
                match_max_len = len_1
            if len_2 > match_max_len:
                match_max_len = len_2
            if len_1 < match_min_len or match_min_len == 0:
                match_min_len = len_1
            if len_2 < match_min_len:
                match_min_len = len_2
            match_total_rows += 2
            match_total_len += len_1 + len_2

            alpha_line = ""
            for item in alpha:
                alpha_line = alpha_line + item + ";"
            alpha_line = alpha_line.strip(";")
            if alpha_line.count(";") > 0 and 'LATIN' in alpha_line:
                alpha_line_aux = "MIXED WITH LATIN"
            elif alpha_line.count(";") > 0:
                alpha_line_aux = "MIXED WITHOUT LATIN"
            else:
                alpha_line_aux = alpha_line

            if not alpha_line_aux in dict_lists:
                dict_lists[alpha_line_aux] = 1
            else:
                dict_lists[alpha_line_aux] += 1
            line = row["NAME1"].strip() + "|" + row["NAME2"].strip() + "|1|" + alpha_line
            end_list.append(line)
            if l_index % (size_in/2) == 0:
                break

    with open("dataset_non_match_" + filename_in + ".csv", encoding="utf8") as csvfile2:
        reader = csv.DictReader(csvfile2, fieldnames=["ID1", "NAME1", "ID2", "NAME2"], delimiter='|')
        for row in reader:
            l_index = l_index + 1
            alpha = []
            alpha_1 = ad.detect_alphabet(row["NAME1"].strip())
            alpha_2 = ad.detect_alphabet(row["NAME2"].strip())
            for item in alpha_1:
                if 'KATAKANA' in item or 'HANGUL' in item or 'HIRAGANA' in item:
                    aux_item = 'CJK'
                else:
                    aux_item = item
                if not aux_item in alpha:
                    alpha.append(aux_item)
            for item in alpha_2:
                if 'KATAKANA' in item or 'HANGUL' in item or 'HIRAGANA' in item:
                    aux_item = 'CJK'
                else:
                    aux_item = item
                if not aux_item in alpha:
                    alpha.append(aux_item)

            len_1 = len(row["NAME1"])
            len_2 = len(row["NAME2"])
            if len_1 > nomatch_max_len or nomatch_max_len == 0:
                nomatch_max_len = len_1
            if len_2 > nomatch_max_len:
                nomatch_max_len = len_2
            if len_1 < nomatch_min_len or nomatch_min_len == 0:
                nomatch_min_len = len_1
            if len_2 < nomatch_min_len:
                nomatch_min_len = len_2
            nomatch_total_rows += 1
            nomatch_total_len = nomatch_total_len + len_1 + len_2

            alpha_line = ""
            for item in alpha:
                alpha_line = alpha_line + item + ";"
            alpha_line = alpha_line.strip(";")
            if alpha_line.count(";") > 0 and 'LATIN' in alpha_line:
                alpha_line_aux = "MIXED WITH LATIN"
            elif alpha_line.count(";") > 0:
                alpha_line_aux = "MIXED WITHOUT LATIN"
            else:
                alpha_line_aux = alpha_line

            if not alpha_line_aux in dict_lists:
                dict_lists[alpha_line_aux] = 1
            else:
                dict_lists[alpha_line_aux] += 1
            line = row["NAME1"].strip() + "|" + row["NAME2"].strip() + "|0|" + alpha_line
            end_list.append(line)
            if l_index % size_in == 0:
                break

    shuffle(end_list)

    with open("dataset_final_" + filename_in + ".csv", 'w', encoding="utf-8") as inW1:
        for row in end_list:
            inW1.write(row.strip() + '\n')

    with open("Report_DataSet_" + filename_in + ".txt", 'w', encoding="utf-8") as inW2:
        inW2.write(filename_in.upper() + '\n')
        inW2.write('\n')
        inW2.write("Total Pairs: " + str((match_total_rows + nomatch_total_rows) /2) + '\n')
        inW2.write("Matching Pairs: " + str(match_total_rows/2) + '\n')
        inW2.write("Non Matching Pairs: " + str(nomatch_total_rows/2) + '\n')
        for key, value in sorted(dict_lists.items(), key=lambda x: x[1], reverse=True):
            inW2.write(key + ": " + str(value) + "\n")

        inW2.write("\n")
        inW2.write("Matching INFO:" + "\n")
        inW2.write("Max Length Matching:" + str(match_max_len) + "\n")
        inW2.write("Min Length Matching:" + str(match_min_len) + "\n")
        inW2.write("Avg Matching:" + str(round(match_total_len / match_total_rows, 3)) + "\n")
        inW2.write("Total Max LEN:" + str(match_total_len) + "\n")
        inW2.write("Total Max ROWS:" + str(match_total_rows) + "\n")
        inW2.write("\n")
        inW2.write("Non Matching INFO:" + "\n")
        inW2.write("Max Length Non Matching:" + str(nomatch_max_len) + "\n")
        inW2.write("Min Length Non Matching:" + str(nomatch_min_len) + "\n")
        inW2.write("Avg Non Matching:" + str(round(nomatch_total_len / nomatch_total_rows, 3)) + "\n")
        inW2.write("\n")
        inW2.write("Dataset INFO:" + "\n")
        inW2.write("Max Length Dataset:" + str(max(match_max_len, nomatch_max_len)) + "\n")
        inW2.write("Min Length Dataset:" + str(min(match_min_len, nomatch_min_len)) + "\n")
        inW2.write("Avg Dataset:" + str(
            round((match_total_len + nomatch_total_len) / (match_total_rows + nomatch_total_rows), 3)) + "\n")


if __name__ == "__main__":
    print("Start " + str(datetime.datetime.now()))
    run_dir = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print ('Must have: File "entities_jrc" in root dir')
    print ("Matching Pairs (1/3): Step 1/5: Splitting dataset into jrc_person and jrc_organization")
    dataset_matching.split_dataset(filename_in="jrc_person", type_in="P")
    dataset_matching.split_dataset(filename_in="jrc_organization", type_in="O")

    print("Matching Pairs (1/3): Step 2/5: Joining unique ids into 1 liners")
    dataset_matching.match_joins_1liner(filename_in="jrc_person")
    dataset_matching.match_joins_1liner(filename_in="jrc_organization")

    print("Matching Pairs (1/3): Step 3/5: Generating matching ordered dataset")
    dataset_matching.match_generate_dataset(filename_in="jrc_person")
    dataset_matching.match_generate_dataset(filename_in="jrc_organization")

    print("Matching Pairs (1/3): Step 4/5: Shuffling matching dataset")
    dataset_matching.match_shuffle_dataset (filename_in="jrc_person")
    dataset_matching.match_shuffle_dataset (filename_in="jrc_organization")


    print("Non Matching Pairs (2/3): Step 1/3: Creating whoosh index \n")
    dataset_non_matching.create_whoosh_index(filename_in="jrc_person")
    dataset_non_matching.create_whoosh_index(filename_in="jrc_organization")

    print("Non Matching Pairs (2/3): Step 2/3: Generating non matching ordered dataset \n")
    dataset_non_matching.search_whoosh_files(filename_in="jrc_person")
    dataset_non_matching.search_whoosh_files(filename_in="jrc_organization")


    print("Non Matching Pairs (2/3): Step 3/3: Shuffling non matching dataset")
    dataset_non_matching.match_shuffle_dataset (filename_in="jrc_person")
    dataset_non_matching.match_shuffle_dataset (filename_in="jrc_organization")

    print("Final Dataset (3/3): Step 1/1: Generating Final Dataset")
    mesh_dataset(filename_in="jrc_person", size_in=5000000)
    mesh_dataset(filename_in="jrc_organization", size_in=800000)

    move_files(dest_in=run_dir)
    print("End " + str(datetime.datetime.now()))
