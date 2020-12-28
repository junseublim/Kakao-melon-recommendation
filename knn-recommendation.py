from collections import Counter
import pandas as pd
import numpy as np
import io
import os
import json
import distutils.dir_util
from collections import Counter
import scipy.sparse as spr
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

song_meta = pd.read_json("res/song_meta.json")
train = pd.read_json("res/train.json")
test = pd.read_json("res/val.json")

def write_json(data, fname):
    def _conv(o):
        if isinstance(o, np.int64) or isinstance(o, np.int32):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath("./res/" + parent)
    with io.open("./res/" + fname, "w", encoding="utf8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)


def load_json(fname):
    with open(fname, encoding='utf8') as f:
        json_obj = json.load(f)

    return json_obj

def debug_json(r):
    print(json.dumps(r, ensure_ascii=False, indent=4))



train_data = train[['tags','songs','updt_date','id']]
test_data = test[['tags', 'songs', 'updt_date', 'id']]
n_train = len(train_data)
n_test = len(test_data)

plylst = pd.concat([train_data, test_data], ignore_index=True)




all_tags = plylst['tags']
tag_counter = Counter([tg for tgs in all_tags for tg in tgs])
tag_dict = {x: tag_counter[x] for x in tag_counter}
all_songs = plylst['songs']
song_counter = Counter([song for songs in all_songs for song in songs])
song_dict = {x: song_counter[x] for x in song_counter}


n_tags = len(tag_dict)
n_songs = len(song_meta)



tag_id_tid = dict()
tag_tid_id = dict()
for i, t in enumerate(tag_dict):
    tag_id_tid[t] = i
    tag_tid_id[i] = t



plylst['tags_id'] = plylst['tags'].map(lambda x: [tag_id_tid.get(t) for t in x if tag_id_tid.get(t) != None])


plylst_use = plylst[['updt_date','songs','tags_id', 'id']]
plylst_use.loc[:,'num_songs'] = plylst_use['songs'].map(len)
plylst_use.loc[:,'num_tags'] = plylst_use['tags_id'].map(len)


plylst_use['song_count'] = plylst_use['songs'].map(lambda x: [1/((song_dict.get(song)-1)**(0.44)+1)  for song in x])


plylst_train = plylst_use.iloc[:n_train,:]
plylst_test = plylst_use.iloc[n_train:,:]


row = np.repeat(range(n_train), plylst_train['num_songs'])
col = [song for songs in plylst_train['songs'] for song in songs]
dat = np.repeat(1, plylst_train['num_songs'].sum())
train_songs_A = spr.csr_matrix((dat, (row, col)), shape=(n_train, n_songs))

row2 = np.repeat(range(n_test), plylst_test['num_songs'])
col2 = [song for songs in plylst_test['songs'] for song in songs]
dat2 = np.repeat(1, plylst_test['num_songs'].sum())
test_songs_A = spr.csr_matrix((dat2, (row2, col2)), shape=(n_test, n_songs))


similarity = cosine_similarity(test_songs_A, train_songs_A)

song_cound_data = np.concatenate(plylst_train['song_count'])
row = np.repeat(range(n_train), plylst_train['num_songs'])
col = [song for songs in plylst_train['songs'] for song in songs]
train_songs_freq = spr.csr_matrix((song_cound_data, (row, col)), shape=(n_train, n_songs))

frequency = test_songs_A.dot(train_songs_freq.T)

frequency_array = frequency.toarray()

total = frequency_array * similarity


class CustomEvaluator:
    def _idcg(self, l):
        return sum((1.0 / np.log(i + 2) for i in range(l)))

    def __init__(self):
        self._idcgs = [self._idcg(i) for i in range(101)]

    def _ndcg(self, gt, rec):
        dcg = 0.0
        for i, r in enumerate(rec):
            if r in gt:
                dcg += 1.0 / np.log(i + 2)

        return dcg / self._idcgs[len(gt)]

    def _eval(self, gt_fname, rec_fname):
        gt_playlists = load_json(gt_fname)
        gt_dict = {g["id"]: g for g in gt_playlists}
        rec_playlists = load_json(rec_fname)
        
        music_ndcg = 0.0
        tag_ndcg = 0.0

        for rec in rec_playlists:
            gt = gt_dict[rec["id"]]
            music_ndcg += self._ndcg(gt["songs"], rec["songs"][:100])
            tag_ndcg += self._ndcg(gt["tags"], rec["tags"][:10])

        music_ndcg = music_ndcg / len(rec_playlists)
        tag_ndcg = tag_ndcg / len(rec_playlists)
        score = music_ndcg * 0.85 + tag_ndcg * 0.15

        return music_ndcg, tag_ndcg, score

    def evaluate(self, gt_fname, rec_fname):
        try:
            music_ndcg, tag_ndcg, score = self._eval(gt_fname, rec_fname)
            print(f"Music nDCG: {music_ndcg:.6}")
            print(f"Tag nDCG: {tag_ndcg:.6}")
            print(f"Score: {score:.6}")
        except Exception as e:
            print(e)


songs_orer = sorted(song_dict.items(), key=lambda x: x[1], reverse=True)
tags_order = sorted(tag_dict.items(), key=lambda x: x[1], reverse=True)
most_songs = [item[0] for item in songs_orer]
most_tags = [item[0] for item in tags_order]


def rec(pids):
    print("start recommendation")
    tt = 1
    res = []
    amplifier = 2
    for pid in pids:
        spid = pid - n_train
        top100 = total[spid].argsort()[-1500:][::-1]
        p = np.zeros((n_songs,1))
        t = np.zeros((n_tags,1))
        maxV = max(total[spid])
        Suv = 0
        plyst_id = plylst_test.iloc[spid].id
        if maxV == 0:
            maxV = 0.01
        for top in top100:
            suv = total[spid][top]  
            new_suv = pow(suv, amplifier)
            for song in plylst_train.loc[top, 'songs']:    
                p[song] += new_suv
            for tag in plylst_train.loc[top, 'tags_id']:
                t[tag] += new_suv
            Suv += suv
        cand_song_idx = p.reshape(-1).argsort()[-500:][::-1]
        cand_tag_idx = t.reshape(-1).argsort()[-50:][::-1]
        songs_already = plylst_test.loc[pid, "songs"]
        tags_already = plylst_test.loc[pid, "tags_id"]
        cand_song_idx = cand_song_idx[np.isin(cand_song_idx, songs_already) == False][:100]
        cand_tag_idx = cand_tag_idx[np.isin(cand_tag_idx, tags_already) == False][:10]
        rec_song_idx = [i for i in cand_song_idx]
        rec_tag_idx = [tag_tid_id[i] for i in cand_tag_idx]
            
        if Suv <= 0.01:
            res.append({
                "id": plyst_id,
                "songs": most_songs[:100],
                "tags": most_tags[:10]
            })
        else:
            res.append({
                "id": plyst_id,
                "songs": rec_song_idx,
                "tags": rec_tag_idx
            })
        
        if tt % 1000 == 0:
            print(tt)

        tt += 1
    return res


answer = rec(plylst_test.index)
write_json(answer, "result/answer.json")


