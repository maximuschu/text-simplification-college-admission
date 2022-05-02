from asyncio.windows_events import NULL
from posixpath import split
import re
import nltk
from numpy import Inf
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import random

file_numbers = [4, 10, 18, 20, 22, 23, 24, 25, 26, 30, 31, 32, 34, 37, 38, 45, 46, 48, 49, 50, 51, 52, 53, 56, 58, 59, 61, 65, 71, 73, 76, 77, 85, 88, 91, 94, 97, 98, 100, 101, 102, 103, 105, 106, 107, 108, 109, 110, 114, 115, 119, 121, 123, 127, 129, 130, 140, 142, 143, 149, 156, 157, 163, 164, 166, 170, 171, 172, 174, 175, 180, 184, 191, 192, 199, 200, 211, 213, 215, 216, 217, 218, 221, 223, 224, 227, 229, 232, 235, 237, 241, 243, 246, 247, 250, 253, 258, 259, 272, 273, 278, 279, 286,
                291, 301, 308, 309, 310, 311, 317, 325, 330]
# Threshold is 0.4 for Jaccard Similarity, 0.7 for Sentence-BERT.
#threshold = 0.4
threshold = 0.7
filter_out_info = True
filter_by_length = False
filter_length = 5
choose_multiple_sentences = True
multiple_limit = Inf
multiple_difference = 0.1
create_train_alignments = False
if(create_train_alignments == True):
    random.seed(18)
    random.shuffle(file_numbers)
    file_numbers = file_numbers[:int(len(file_numbers)*0.5)]


# Produce ngrams for desired sentence.
def n_gram(n, sentence):
    sentence = sentence.split()
    output = set()
    for i in range(len(sentence)-n+1):
        current = sentence[i:i+n]
        result = ""
        for word in current:
            result += word
        output.add(result)
    return output


# Find alignments in a file.
def calculate_alignment(file_original, file_simple):
    def get_score(item):
        return item[4]
    simple_lines = []
    alignments = []
    for index_simple, line_simple in nonblank_lines(file_simple):
        if(filter_out_info == False):
            line_simple = line_simple.replace('<INFO>', '')
        simple_lines.append((index_simple, line_simple))
    for index_original, line_original in nonblank_lines(file_original):
        max_score = 0
        max_index = NULL
        final_simple = NULL
        multiple_lines = []

        # Jaccard Similarity, uncomment if needed.
        """
        temp_original = line_original.lower().split()
        words_original = set()
        for word in temp_original:
            word = re.sub('\W+', '', word)
            if(word != ''):
                words_original.add(word)

        for index_simple, line_simple in simple_lines:
            temp_simple = line_simple.lower().split()
            words_simple = set()
            for word in temp_simple:
                word = re.sub('\W+', '', word)
                words_simple.add(word)
            intersection = words_original.intersection(words_simple)
            union = words_original.union(words_simple)
            current_score = float(len(intersection))/len(union)
            if(current_score > max_score):
                max_score = current_score
                max_index = index_simple
                final_simple = line_simple
        """

        # Sentence-BERT
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        emb1 = model.encode(line_original)
        for index_simple, line_simple in simple_lines:
            emb2 = model.encode(line_simple)
            current_score = util.cos_sim(emb1, emb2).item()
            if(choose_multiple_sentences == False):
                if(current_score > max_score):
                    max_score = current_score
                    max_index = index_simple
                    final_simple = line_simple
            else:
                if(current_score >= threshold):
                    original_filter = remove_punctuation(line_original)
                    simple_filter = remove_punctuation(line_simple)
                    original_unigram = n_gram(1, original_filter)
                    simple_unigram = n_gram(1, simple_filter)
                    original_bigram = n_gram(2, original_filter)
                    simple_bigram = n_gram(2, simple_filter)
                    original_trigram = n_gram(3, original_filter)
                    simple_trigram = n_gram(3, simple_filter)
                    simple_unigram_difference = simple_unigram.difference(
                        original_unigram)
                    simple_bigram_difference = simple_bigram.difference(
                        original_bigram)
                    simple_trigram_difference = simple_trigram.difference(
                        original_trigram)
                    simple_unigram_unique = float(
                        len(simple_unigram_difference))/len(simple_unigram)
                    if(len(simple_bigram) == 0):
                        simple_bigram_unique = None
                    else:
                        simple_bigram_unique = float(
                            len(simple_bigram_difference))/len(simple_bigram)
                    if(len(simple_trigram) == 0):
                        simple_trigram_unique = None
                    else:
                        simple_trigram_unique = float(
                            len(simple_trigram_difference))/len(simple_trigram)
                    if(filter_out_info == True and "<INFO>" in line_simple):
                        continue
                    if(filter_by_length == True and len(simple_unigram) <= filter_length):
                        continue
                    multiple_lines.append([index_original, index_simple,
                                           line_original, line_simple, current_score, simple_unigram_unique, simple_bigram_unique, simple_trigram_unique])

        # Only choose best score.
        if(choose_multiple_sentences == False):
            if(max_score >= threshold):
                original_filter = remove_punctuation(line_original)
                simple_filter = remove_punctuation(final_simple)
                original_unigram = n_gram(1, original_filter)
                simple_unigram = n_gram(1, simple_filter)
                original_bigram = n_gram(2, original_filter)
                simple_bigram = n_gram(2, simple_filter)
                original_trigram = n_gram(3, original_filter)
                simple_trigram = n_gram(3, simple_filter)

                simple_unigram_difference = simple_unigram.difference(
                    original_unigram)
                simple_bigram_difference = simple_bigram.difference(
                    original_bigram)
                simple_trigram_difference = simple_trigram.difference(
                    original_trigram)
                simple_unigram_unique = float(
                    len(simple_unigram_difference))/len(simple_unigram)
                if(len(simple_bigram) == 0):
                    simple_bigram_unique = None
                else:
                    simple_bigram_unique = float(
                        len(simple_bigram_difference))/len(simple_bigram)
                if(len(simple_trigram) == 0):
                    simple_trigram_unique = None
                else:
                    simple_trigram_unique = float(
                        len(simple_trigram_difference))/len(simple_trigram)

                if(filter_out_info == True and "<INFO>" in final_simple):
                    continue
                if(filter_by_length == True and len(simple_unigram) <= filter_length):
                    continue
                alignments.append([index_original, max_index,
                                   line_original, final_simple, max_score, simple_unigram_unique, simple_bigram_unique, simple_trigram_unique])
        else:
            count = 0
            result = sorted(multiple_lines, key=get_score)
            result.reverse()
            last_score = 0
            for current_line in result:
                if(count >= multiple_limit):
                    break
                if(count == 0):
                    alignments.append(current_line)
                    last_score = current_line[4]
                else:
                    current_score = current_line[4]
                    if((last_score - current_score) < multiple_difference):
                        alignments.append(current_line)
                        last_score = current_score
                    else:
                        break
                count = count + 1
    return alignments


# Returns line if not blank.
def nonblank_lines(f):
    for index, l in enumerate(f):
        line = l.rstrip()
        if line:
            yield index, line


# Strips all punctuation for ngram purposes.
def remove_punctuation(sentence):
    temp_original = sentence.lower().split()
    result = ""
    for word in temp_original:
        word = re.sub('\W+', '', word)
        result += word
        result += " "
    result = result.rstrip(result[-1])
    return result


# Use for reformatting file so that one sentence per line.
def split_sentences(f, location, file_number):
    file_format = open(
        f"Reformat/{location}/{file_number}.txt", "w", encoding='utf-8')
    for index_original, line_original in enumerate(f):
        if("." in line_original):
            sentences = nltk.tokenize.sent_tokenize(line_original)
            for current in sentences:
                file_format.write(str(current) + '\n')
        else:
            file_format.write(str(line_original))


# Reformats data, splits sentences in lines.
def reformat():
    for file_number in file_numbers:
        nltk.download('punkt')
        file_original = open(
            f"Filtered Data/Original/{file_number} - ADM.txt", "r", encoding='utf-8')
        file_simple = open(
            f"Filtered Data/Simplified/{file_number} - ADM - ACCEPTED.txt", "r", encoding='utf-8')
        split_sentences(file_original, "Original", file_number)
        split_sentences(file_simple, "Simplified", file_number)


# Finds the precision, recall, and F1 score across all files.
def calculate_accuracy(predicted_alignments):
    actual_alignments = []
    for file_number in file_numbers:
        file = open(
            f"Output/Final/{file_number}.txt", "r", encoding='utf-8')
        current_file_actual_aligning = []
        current_aligning = None
        current_original = None
        current_simple = None
        current_generated = None
        result = None
        for index, line in enumerate(file):
            if(index % 5 == 0):
                current_aligning = line
                current_aligning = remove_punctuation(current_aligning)
                result = [int(s)
                          for s in current_aligning.split() if s.isdigit()]

            if(index % 5 == 1):
                current_original = line
                if(filter_out_info == False):
                    current_original = current_original.replace('<INFO>', '')
            if(index % 5 == 2):
                current_simple = line
                if(filter_out_info == False):
                    current_simple = current_simple.replace('<INFO>', '')
            if(index % 5 == 3):
                current_generated = line
                if(filter_out_info == False):
                    current_generated = current_generated.replace('<INFO>', '')
            if(index % 5 == 4):
                if(filter_out_info == True and "<INFO>" in current_simple):
                    continue
                current_file_actual_aligning.append(result)
        actual_alignments.append(current_file_actual_aligning)

    accuracy_results = open(
        f"Output/Sentences/Accuracy Results.txt", "w", encoding='utf-8')

    total_precision = 0.0
    total_recall = 0.0
    total_f1_score = 0.0
    for index, file_number in enumerate(file_numbers):
        current_predicted_alignments = predicted_alignments[index]
        current_actual_alignments = actual_alignments[index]
        number_predicted = len(current_predicted_alignments)
        number_actual = len(current_actual_alignments)

        total_correct = 0
        for prediction in current_predicted_alignments:
            for actual in current_actual_alignments:
                if(len(actual) == 1):
                    continue
                elif(len(actual) == 2):
                    actual = tuple(actual)
                    if(prediction == actual):
                        total_correct = total_correct + 1
                else:
                    weight = 1/(len(actual)-1)
                    for i in range(1, len(actual)):
                        current_actual = (actual[0], actual[i])
                        if(prediction == current_actual):
                            total_correct = total_correct + weight

        precision = 0
        if(number_predicted != 0):
            precision = total_correct/number_predicted
        recall = 0
        if(number_actual != 0):
            recall = total_correct/number_actual
        f1_score = 0
        if((precision + recall) != 0):
            f1_score = (2*precision*recall) / (precision + recall)
        accuracy_results.write(str(file_number) + ": \n")
        accuracy_results.write("Precision: " + str(precision) + "\n")
        accuracy_results.write("Recall: " + str(recall) + "\n")
        accuracy_results.write("F1 Score: " + str(f1_score) + "\n\n")
        total_precision = total_precision + precision
        total_recall = total_recall + recall
        total_f1_score = total_f1_score + f1_score
    total_precision = total_precision / len(file_numbers)
    total_recall = total_recall / len(file_numbers)
    total_f1_score = total_f1_score / len(file_numbers)
    accuracy_results.write("Total Averages: \n")
    accuracy_results.write("Precision: " + str(total_precision) + "\n")
    accuracy_results.write("Recall: " + str(total_recall) + "\n")
    accuracy_results.write("F1 Score: " + str(total_f1_score) + "\n\n")


def main():
    # Record both n_gram results as well as producing automatic alignments.
    ngram_results = open(f"Ngram/Results.txt",
                         "w", encoding="utf-8")

    unigram_all_count = 0
    bigram_all_count = 0
    trigram_all_count = 0
    unigram_all_sum = 0.0
    bigram_all_sum = 0.0
    trigram_all_sum = 0.0
    predicted_alignments = []
    total_original = []
    total_simplified = []
    for file_number in file_numbers:
        unigram_count = 0
        bigram_count = 0
        trigram_count = 0
        unigram_sum = 0.0
        bigram_sum = 0.0
        trigram_sum = 0.0
        current_file_predicted_alignments = []

        file_original = open(
            f"Facebook Output/{file_number}.txt", "r", encoding='utf-8')
        file_simple = open(
            f"Reformat/Simplified/{file_number}.txt", "r", encoding='utf-8')

        alignments = calculate_alignment(file_original, file_simple)
        f = open(
            f"Facebook Alignments/{file_number}.txt", "w", encoding="utf-8")
        for current_line in alignments:
            unigram_sum += current_line[5]
            unigram_count += 1
            if(current_line[6] != None):
                bigram_sum += current_line[6]
                bigram_count += 1
            if(current_line[7] != None):
                trigram_sum += current_line[7]
                trigram_count += 1
            f.write(str(current_line) + '\n')
            current_file_predicted_alignments.append(
                (current_line[0], current_line[1]))
            total_original.append(current_line[2])
            total_simplified.append(current_line[3])
        f.close()
        unigram_all_count += unigram_count
        bigram_all_count += bigram_count
        trigram_all_count += trigram_count
        unigram_all_sum += unigram_sum
        bigram_all_sum += bigram_sum
        trigram_all_sum += trigram_sum
        unigram_current_result = None
        bigram_current_result = None
        trigram_current_result = None
        if(unigram_count != 0):
            unigram_current_result = unigram_sum/unigram_count
        if(bigram_count != 0):
            bigram_current_result = bigram_sum/bigram_count
        if(trigram_count != 0):
            trigram_current_result = trigram_sum/trigram_count
        ngram_results.write(
            f"{file_number}: Unigram: {unigram_current_result} Bigram: {bigram_current_result} Trigram: {trigram_current_result}\n")
        predicted_alignments.append(current_file_predicted_alignments)

    if(create_train_alignments == True):
        df = pd.DataFrame({'Original': total_original,
                           'Simplified': total_simplified})
        df.to_csv('train_alignments.csv')
    ngram_results.write(
        f"Total Results: Unigram: {unigram_all_sum/unigram_all_count} Bigram: {bigram_all_sum/bigram_all_count} Trigram: {trigram_all_sum/trigram_all_count}\n")
    ngram_results.close()
    calculate_accuracy(predicted_alignments)


if __name__ == '__main__':
    main()
