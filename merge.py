fin1 = open('result/minicpm_result.txt', 'r')
fin2 = open('result/nv_result.txt', 'r')
fout = open('result/result.txt', 'w')
for i, j in zip(fin1, fin2):
    i = i.strip()
    j = j.strip()
    if i == "":
        continue
    ids1 = i.split(',')
    ids2 = j.split(',')
    common = set(ids1).intersection(set(ids2))
    ids = []
    for id in ids2:
        if id in common:
            ids.append(id)
    new = []
    for id in ids2[:10] + ids1[:10]:
        if id not in ids and id not in new:
            new.append(id)

    for id in ids2[-10:]+ids1[-10:]:
        if id not in ids and id not in new:
            new.append(id)

    ids += new
    ids = ids[:20]
    line = ",".join(ids)
    fout.write(line+"\n")

fout.close()