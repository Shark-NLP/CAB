from transformers import GPT2Tokenizer
import sys
tokenizer = GPT2Tokenizer.from_pretrained('cache')


def tokenize(file):
    f_in = open(file, encoding='utf-8')
    f_out = open(file+'.tokens', "w", encoding='utf-8')

    _ = 0
    while True:
        line = f_in.readline()
        if line == "":
            break
        if _ % 50000 == 0:
            print(f"{file}: {_}")
        _ += 1
        tokens = tokenizer.tokenize(line.strip())
        f_out.write(f"{' '.join(tokens)}" + '\n')

    f_in.close()
    f_out.close()

    print("Complete")


if __name__ == '__main__':
    tokenize(sys.argv[1])
