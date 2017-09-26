#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import csv
import sys
import math
import random
import jellyfish
import pyxdameraulevenshtein
import numpy as np
import itertools
import unicodedata
from alphabet_detector import AlphabetDetector

fields = [ "geonameid" ,
           "name" ,
           "asciiname" ,
           "alternatenames" ,
           "latitude" ,
           "longitude" ,
           "feature class" ,
           "feature_code" ,
           "country_code" ,
           "cc2" ,
           "admin1_code" ,
           "admin2_code" ,
           "admin3_code" ,
           "admin4_code" ,
           "population" ,
           "elevation" ,
           "dem" ,
           "timezone" ,
           "modification_date" ]

def check_alphabet(str, alphabet, only=True):
    ad = AlphabetDetector()
    if only:
        return ad.only_alphabet_chars(str, alphabet.upper())
    else:
        for i in str:
            if ad.is_in_alphabet(i, alphabet.upper()): return True
        return False


def detect_alphabet(str):
    ad = AlphabetDetector()
    uni_string = unicode(str, "utf-8")
    ab = ad.detect_alphabet(uni_string)
    if "CYRILLIC" in ab:
        return "CYRILLIC"
    return ab.pop() if len(ab) != 0 else 'UND'


# The geonames dataset can be obtained from http://download.geonames.org/export/dump/allCountries.zip
def build_dataset_from_geonames(input='allCountries.txt', output='dataset-unfiltered.txt'):
    csv.field_size_limit(sys.maxsize)
    lastname = None
    lastname2 = None
    lastid = None
    country = None
    skip = random.randint(10, 100)
    file = open(output, "w+")
    with open(input) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=fields, delimiter='\t')
        for row in reader:
            skip = skip - 1
            if skip > 0: continue
            names = set([name.strip() for name in ("" + row['alternatenames']).split(",") if len(name.strip()) > 2])
            if len(names) < 5: continue
            lastid = row['geonameid']
            firstcountry = row['country_code']
            lastname = random.sample(names, 1)[0]
            lastname2 = random.sample(names, 1)[0]
            while True:
                lastname2 = random.sample(names, 1)[0]
                if not (lastname2.lower() == lastname.lower()): break
    with open(input) as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=fields, delimiter='\t')
        for row in reader:
            names = set([name.strip() for name in ("" + row['alternatenames']).split(",") if len(name.strip()) > 2])
            if len(row['name'].strip()) > 2: names.add(row['name'].strip())
            if len(unicode(row['asciiname'], "utf-8").strip()) > 2: names.add(row['asciiname'].strip())
            if len(names) < 3: continue
            id = row['geonameid']
            country = row['country_code']
            randomname1 = random.sample(names, 1)[0]
            randomname3 = random.sample(names, 1)[0]
            randomname5 = random.sample(names, 1)[0]
            while True:
                randomname2 = random.sample(names, 1)[0]
                if not (randomname1.lower() == randomname2.lower()): break
            attempts = 1000
            while attempts > 0:
                attempts = attempts - 1
                randomname3 = random.sample(names, 1)[0]
                if lastname is None or (
                        jaccard(randomname3, lastname) > 0.0 and not (randomname3.lower() == lastname.lower())): break
                if damerau_levenshtein(randomname3, lastname) == 0.0 and random.random() < 0.5: break
            if attempts <= 0:
                auxl = lastname
                lastname = lastname2
                lastname2 = auxl
                attempts = 1000
                while attempts > 0:
                    attempts = attempts - 1
                    randomname3 = random.sample(names, 1)[0]
                    if lastname is None or (jaccard(randomname3, lastname) > 0.0 and not (
                        randomname3.lower() == lastname.lower())): break
                    if damerau_levenshtein(randomname3, lastname) == 0.0 and random.random() < 0.5: break
            if attempts <= 0:
                lastid = id
                lastname = randomname1
                lastname2 = randomname2
                firstcountry = row['country_code']
                continue
            if randomname1 is None or randomname2 is None or id is None or country is None:
                continue
            print randomname1 + "\t" + randomname2 + "\tTRUE\t" + id + "\t" + id + "\t" + detect_alphabet(
                randomname1) + "\t" + detect_alphabet(randomname2) + "\t" + country + "\t" + country
            if not (
                lastid is None): print lastname + "\t" + randomname3 + "\tFALSE\t" + lastid + "\t" + id + "\t" + detect_alphabet(
                lastname) + "\t" + detect_alphabet(randomname3) + "\t" + firstcountry + "\t" + country
            lastname = randomname1
            if len(names) < 5:
                lastid = id
                lastname2 = randomname2
                firstcountry = country
                continue
            while True:
                randomname4 = random.sample(names, 1)[0]
                if not (randomname4.lower() == randomname1.lower()) and not (
                    randomname4.lower() == randomname2.lower()): break
            attempts = 1000
            while attempts > 0:
                attempts = attempts - 1
                randomname5 = random.sample(names, 1)[0]
                if lastname2 is None or (jaccard(randomname5, lastname2) > 0.0 and not (
                    randomname5.lower() == lastname2.lower()) and not (
                    randomname5.lower() == randomname3.lower())): break
                if damerau_levenshtein(randomname5, lastname2) == 0.0 and random.random() < 0.5: break
            if attempts > 0:
                aux = random.sample([randomname1, randomname2], 1)[0]
                print randomname4 + "\t" + aux + "\tTRUE\t" + id + "\t" + id + "\t" + detect_alphabet(
                    randomname4) + "\t" + detect_alphabet(aux) + "\t" + country + "\t" + country
                if not (
                    lastid is None): print lastname2 + "\t" + randomname5 + "\tFALSE\t" + lastid + "\t" + id + "\t" + detect_alphabet(
                    lastname2) + "\t" + detect_alphabet(randomname5) + "\t" + firstcountry + "\t" + country
            lastname2 = random.sample([randomname2, randomname4], 1)[0]
            lastid = id


def filter_dataset( input='dataset-unfiltered.txt' , num_instances=2500000):
    pos = [ ]
    neg = [ ]
    file = open("dataset-string-similarity.txt","w+")
    print "Filtering for {0}...".format(num_instances*2)
    for line in open(input):
        splitted = line.split('\t')
        if not(splitted[2] == "TRUE" or splitted[2] == "FALSE") or \
            not(len(unicode(splitted[7], "utf-8")) == 2 and len(unicode(splitted[8], "utf-8")) == 3) or \
            not(splitted[5] != "UND" and splitted[6] != "UND") or \
            not(splitted[3].isdigit() and splitted[4].isdigit()) or \
            len(splitted) != 9 or \
            len(unicode(splitted[1], "utf-8")) < 3 or \
            len(unicode(splitted[0], "utf-8")) < 3:
            continue
        if '\tTRUE\t' in line : pos.append(line)
        else: neg.append(line)
    pos = random.sample(pos, len(pos))
    neg = random.sample(neg, len(neg))
    for i in range(num_instances):
        file.write(pos[i])
        file.write(neg[i])
    print "Filtering ended."
    file.close()

def skipgrams(sequence, n, k):
    sequence = " " + sequence + " "
    res = [ ]
    for ngram in { sequence[i:i+n+k] for i in xrange(len(sequence) - ( n + k - 1 ) ) }:
        if k == 0 : res.append( ngram )
        else: res.append( ngram[0:1] + ngram[k+1:len(ngram)] )
    return res

def skipgram ( str1 , str2 ):
    a1 = set( skipgrams( str1 , 2 , 0 ) )
    a2 = set( skipgrams( str1 , 2 , 1 ) + skipgrams( str1 , 2 , 2 ) )
    b1 = set( skipgrams( str2 , 2 , 0 ) )
    b2 = set( skipgrams( str2 , 2 , 1 ) + skipgrams( str1 , 2 , 2 ) )
    c1 = a1.intersection(b1)
    c2 = a2.intersection(b2)
    d1 = a1.union(b1)
    d2 = a2.union(b2)
    try: return float(len(c1) + len(c2)) / float(len(d1) + len(d2))
    except:
        if str1 == str2 : return 1.0
        else: return 0.0

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def davies ( str1 , str2 ):
    a = strip_accents( str1.lower() ).replace(u'-',u' ').split(' ')
    b = strip_accents( str2.lower() ).replace(u'-',u' ').split(' ')
    for i in range( len(a) ):
            if len(a[i]) > 1 or not(a[i].endswith(u'.')) : continue
            replacement = len( str2 )
            for j in range( len(b) ):
                    if b[j].startswith( a[i].replace(u'.','') ):
                            if len(b[j]) < replacement:
                                    a[i] = b[j]
                                    replacement = len( b[j] )
    for i in range( len(b) ):
        if len(b[i]) > 1 or not(b[i].endswith(u'.')) : continue
        replacement = len( str1 )
        for j in range( len(a) ):
            if a[j].startswith( b[i].replace(u'.','') ):
                if len(a[j]) < replacement:
                    b[i] = a[j]
                    replacement = len( a[j] )
    a = set( a )
    b = set( b )
    aux1 = sorted_winkler( str1 , str2 )
    intersection_length = ( sum( max( jaro_winkler(i, j) for j in b ) for i in a ) + sum( max( jaro_winkler(i, j) for j in a ) for i in b ) ) / 2.0
    aux2 = float(intersection_length)/( len(a) + len(b) - intersection_length )
    return ( aux1 + aux2 ) / 2.0

def cosine ( str1 , str2 ):
    str1 = " " + str1 + " "
    str2 = " " + str2 + " "
    x = list( itertools.chain.from_iterable( [ [str1[i:i+n] for i in range(len(str1)-(n-1))] for n in [2,3] ] ) )
    y = list( itertools.chain.from_iterable( [ [str2[i:i+n] for i in range(len(str2)-(n-1))] for n in [2,3] ] ) )
    vectorIndex={ }
    offset=0
    for offset, word in enumerate( set( x + y ) ): vectorIndex[word] = offset
    vector = np.zeros( len( vectorIndex ) )
    for word in x : vector[vectorIndex[word]] += 1
    x = vector
    vector = np.zeros( len( vectorIndex ) )
    for word in y : vector[vectorIndex[word]] += 1
    y = vector
    numerator = sum(a*b for a,b in zip(x,y))
    denominator = math.sqrt(sum([a*a for a in x])) * math.sqrt(sum([a*a for a in y]))
    try : return numerator / denominator
    except:
        if str1 == str2 : return 1.0
        else : return 0.0

def damerau_levenshtein ( str1 , str2 ):
    aux = pyxdameraulevenshtein.normalized_damerau_levenshtein_distance( str1 , str2 )
    return 1.0 - aux

def jaro ( str1 , str2 ):
    aux = jellyfish.jaro_distance( str1 , str2 )
    return aux

def jaro_winkler ( str1 , str2 ):
    aux = jellyfish.jaro_winkler( str1 , str2 )
    return aux

def monge_elkan_aux( str1 , str2 ):
    cummax = 0
    for ws in str1.split(" "):
        maxscore=0
        for wt in str2.split(" "):
            maxscore = max( maxscore , jaro_winkler(ws,wt) )
        cummax += maxscore
    return cummax / len(str1.split(" "))

def monge_elkan( str1 , str2 ):
    return ( monge_elkan_aux( str1 , str2 ) + monge_elkan_aux( str2 , str1 ) ) / 2.0

# http://www.catalysoft.com/articles/StrikeAMatch.html
def strike_a_match( str1 , str2 ):
    pairs1 = {str1[i:i+2] for i in xrange(len(str1) - 1)}
    pairs2 = {str2[i:i+2] for i in xrange(len(str2) - 1)}
    union  = len(pairs1) + len(pairs2)
    hit_count = 0
    for x in pairs1:
        for y in pairs2:
            if x == y:
                hit_count += 1
                break
    try: return (2.0 * hit_count) / union
    except:
            if str1 == str2 : return 1.0
            else: return 0.0

def jaccard ( str1 , str2 ):
    str1 = " " + str1 + " "
    str2 = " " + str2 + " "
    a = list( itertools.chain.from_iterable( [ [str1[i:i+n] for i in range(len(str1)-(n-1))] for n in [2,3] ] ) )
    b = list( itertools.chain.from_iterable( [ [str2[i:i+n] for i in range(len(str2)-(n-1))] for n in [2,3] ] ) )
    a = set( a )
    b = set( b )
    c = a.intersection(b)
    try: return float(len(c)) / ( float((len(a) + len(b) - len(c))) )
    except:
        if str1 == str2 : return 1.0
        else: return 0.0

def soft_jaccard( str1 , str2 ):
    a = set( str1.split(" ") )
    b = set( str2.split(" ") )
    intersection_length = ( sum( max( jaro_winkler(i, j) for j in b ) for i in a ) + sum( max( jaro_winkler(i, j) for j in a ) for i in b ) ) / 2.0
    return float(intersection_length)/(len(a) + len(b) - intersection_length)

def sorted_winkler ( str1 , str2 ):
    a = sorted( str1.split(" ") )
    b = sorted( str2.split(" ") )
    a = " ".join( a )
    b = " ".join( b )
    return jaro_winkler( a , b )

def permuted_winkler ( str1 , str2 ):
    a = str1.split(" ")
    b = str2.split(" ")
    if len(a) > 5: a = a[0:5] + [ u''.join( a[5:] ) ]
    if len(b) > 5: b = b[0:5] + [ u''.join( b[5:] ) ]
    lastscore = 0.0
    for a in itertools.permutations(a):
        for b in itertools.permutations(b):
            sa = u' '.join( a )
            sb = u' '.join( b )
            score = jaro_winkler( sa , sb )
            if score > lastscore : lastscore = score
    return lastscore
