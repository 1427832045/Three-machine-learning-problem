from keras.models import load_model
from keras.datasets import imdb
import numpy as np

word_index = imdb.get_word_index(path='./imdb_word_index.json')


def vect_seq(seqs, dimension=50000):
    results = np.zeros((len(seqs), dimension))
    for i, seq in enumerate(seqs):
        results[i, seq] = 1.
    return results


s = """        
Iron Man is the best bad-ass action MCU movie definitely my favorite in the MARVEL comic book films. It has a great story great origin story about Tony Stark 8rober Downey Jr.) how he become armored avenger "Iron Man". The movie has high-flying action sequence and super speed. Iron man was build and not born. Excellent performance from Robert Downey Jr. as playboy billionaire Tony Stark/Iron Man. The first movie in the trilogy is the best one it is my favorite MCU film it is also intelligent and interesting. How from an selfish, arrogant prick turns in to a good heart person who want to do the right thing to protect innocent and destroy the weapons his designed that fall in to terrorists hands.
""".strip().replace('.', '').replace(',', '')  # s为评论
l = np.asarray([[int(word_index.get(i)) for i in s.split(' ') if
                 int(word_index.get(i, '0')) < 50000 and int(word_index.get(i, '0')) != 0]])
x_test = vect_seq(l)
model = load_model('my_model.h5')
data = model.predict(x_test)
print(data)  # 接近1为正面，接近0为负面评论
