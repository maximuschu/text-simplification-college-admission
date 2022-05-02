from asyncio.windows_events import NULL

number_of_files = 341
lower_bound = 0.3
upper_bound = 0.7


def get_length(item):
    return item[1]


# Filter out files that aren't within percentage range
def filter_by_percentage():
    files = []
    for current_file in range(1, number_of_files+1):
        f = open(
            f"Data/2019 ADM TXTS/{current_file} - ADM.txt", "r", encoding='utf-8')
        f_simple = NULL
        try:
            f_simple = open(
                f"Data/2019 ADM TXTS SIMPLE/{current_file} - ADM - ACCEPTED.txt", "r", encoding='utf-8')
        except OSError as e:
            continue
        current_percentage = len(f_simple.read())/len(f.read())
        if(current_percentage >= lower_bound and current_percentage <= upper_bound):
            files.append(current_file)
    return files


# Sort the files by longest to shortest
def filter_by_length():
    lengths = []
    for current_file in range(1, number_of_files+1):
        f = open(
            f"Data/2019 ADM TXTS/{current_file} - ADM.txt", "r", encoding='utf-8')
        f_simple = NULL
        try:
            f_simple = open(
                f"Data/2019 ADM TXTS SIMPLE/{current_file} - ADM - ACCEPTED.txt", "r", encoding='utf-8')
        except OSError as e:
            continue
        lengths.append((current_file, len(f.read())))

    result = sorted(lengths, key=get_length)
    result.reverse()
    files = []
    for current in result:
        files.append(current[0])
    return files


def main():
    percents = filter_by_percentage()
    lengths = filter_by_length()
    files = []
    for current_p in percents:
        for i in range(len(lengths)):
            if(current_p == lengths[i]):
                files.append((current_p, i))

    result = sorted(files, key=get_length)
    result_files = []
    for file in result:
        result_files.append(file[0])

    result_files.sort()
    print(result_files)


if __name__ == '__main__':
    main()
