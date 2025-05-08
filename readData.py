def ReadData():
    fid = open('shakespeare.txt', "r")
    book_data = fid.read()
    fid.close()
    unique_chars = list(set(book_data))
    return book_data, unique_chars


data, unique_chars = ReadData()

print(data)
