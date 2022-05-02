file_numbers = [4, 10, 18, 20, 22, 23, 24, 25, 26, 30, 31, 32, 34, 37, 38, 45, 46, 48, 49, 50, 51, 52, 53, 56, 58, 59, 61, 65, 71, 73, 76, 77, 85, 88, 91, 94, 97, 98, 100, 101, 102, 103, 105, 106, 107, 108, 109, 110, 114, 115, 119, 121, 123, 127, 129, 130, 140, 142,
                143, 149, 156, 157, 163, 164, 166, 170, 171, 172, 174, 175, 180, 184, 191, 192, 199, 200, 211, 213, 215, 216, 217, 218, 221, 223, 224, 227, 229, 232, 235, 237, 241, 243, 246, 247, 250, 253, 258, 259, 272, 273, 278, 279, 286, 291, 301, 308, 309, 310, 311, 317, 325, 330]

for file_number in file_numbers:
    file_original = open(
        f"Simplified/{file_number}.txt", "r", encoding='utf-8')
    file_indexed = open(
        f"Indexed/Simplified/{file_number}.txt", "w", encoding='utf-8')
    for index, line in enumerate(file_original):
        file_indexed.write(str(index) + '\n')
        file_indexed.write(line)
