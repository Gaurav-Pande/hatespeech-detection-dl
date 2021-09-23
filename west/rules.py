import load_data
data, y = load_data.read_file('../hatespeech', True)
sup_data = []
sup_idx = []
sup_label = []
file_name = 'doc_id.txt'
infile = open('../hatespeech/doc_50.txt', mode='r',
              encoding='utf-8')
text = infile.readlines()
for i, line in enumerate(text):
    line = line.split('\n')[0]
    class_id, doc_ids = line.split(':')
    assert int(class_id) == i
    seed_idx = doc_ids.split(',')
    seed_idx = [int(idx) for idx in seed_idx]
    sup_idx.append(seed_idx)
    for idx in seed_idx:
        sup_data.append("".join(data[idx]))
        sup_label.append(i)


train_data, train_label = sup_data, sup_label
res= []
for t in train_data:
    res.append(t.split(" "))

for r in res:
    print(r)
