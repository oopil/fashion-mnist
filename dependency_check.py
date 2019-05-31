def dependency_check():
    # file_path = './euijin_code/one-hot-testresult.txt'
    file_path = '/home/soopil/Desktop/github/fashion-mnist/euijin_code/one-hot-testresult.txt'
    fd = open(file_path)
    lines = fd.readlines()
    contents = []
    for i, line in enumerate(lines):
        if i != 0 and line != '\n':
            contents.append(line.replace('\n', '').replace(':', ''))

    print(lines)
    print(contents)

    depend_array = []

    for i in range(len(contents)):
        # print(contents[i])
        assert len(contents[i]) != 0
        if i % 3 == 0:
            # print(contents[i])
            # print(contents[i].split('='))
            # print(contents[i+1].split('='))
            # print(contents[i+2].split('='))
            # contents[i+1].split(',')[0].split('=')[1]
            # print(contents[i+1].split(',')[0].split('=')[1])
            new_line = [contents[i],
                        contents[i + 1].split(',')[0].split('=')[1], contents[i + 1].split(',')[1].split('=')[1],
                        contents[i + 2].split(',')[0].split('=')[1], contents[i + 2].split(',')[1].split('=')[1]]
            print(new_line)
            depend_array.append(new_line)
    return depend_array

if __name__ == "__main__":
    dependency_check()