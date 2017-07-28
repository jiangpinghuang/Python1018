# Stanford Sentiment Corpus for label and sentence.

def main():    
    srcFile = "/Users/hjp/MacBook/Workspace/Workshop/Corpus/ssc/ssc_train.txt"
    tarFile = "/Users/hjp/MacBook/Workspace/Workshop/Corpus/ssc/tar_train.txt"
    
    wFile = open(tarFile, 'w')
    
    for line in open(srcFile, 'r'):
        print line
        label = line[1:2]
        print label
        sent = ""
        tokens = line.split()
        for i in range(len(tokens)):
            if ")" in tokens[i]:
                print tokens[i]
                words = tokens[i].split(')') 
                if "LRB" not in words[0] and "RRB" not in words[0] and "--" not in words[0]:
                    print words[0]
                    if len(sent) == 0:
                        sent = words[0]
                    else:
                        sent = sent + " " + words[0]
        print sent
        wFile.write(label + "\t" + sent + "\n")

if __name__ == "__main__":
    main()