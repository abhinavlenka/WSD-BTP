


f1 = open("fry_news_2020_10K-words_t.txt","r")
f2 = open("fry_news_2020_10K-words.txt","r")
f = open("lexicon.txt","w")
for i,j in zip(f1.readlines(),f2.readlines()):
    f.write(i+" "+ j)
f1.close()
f2.close()
f.close()