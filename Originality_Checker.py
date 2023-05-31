'''
    Класс Originality_Checker выполняет проверку аудиозаписей на предмет синтезированности
    Чтобы выполнить проверку:
    1. Создайте экземпляр класса (если нужна только проверка файла параметры не нужны)
    2. Вызовите метод get_characteristics с параметром path_to_audio - путь до файла
    !!! Модуль работает некорректно с 1 файлом, лучше передавать список файлов
    или хотя бы продублировать файл несколько раз!!!
    3. Вызвать метод Originality_Checker с параметром features - результат работы метода get_characteristics

    Метод возвращает список с вероятностями принадлежности аудиофайла к классу оригинальных сообщений

    Пример использования:
    # Инициализация полей класса
    csv_file = 'data.csv'
    path_to_original_file = 'audio/original human voice'
    path_to_spoof_file = 'audio/spoof human voice'

    # Создание экземпляра класса Originality_Checker
    checker = Originality_Checker(csv_file, path_to_original_file, path_to_spoof_file)

    # Создание списка с файлами
    files = [file1, file2, file3]

    # Извлечение характеристик
    features = [checker.get_characteristics(i) for i in files]

    # Предсказание по набору характеристик
    res = checker.Originality_Checker(features, 'models/keras_10_model.h5', label)
    print(res)
    '''

import csv
import os
import random

import keras
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


class Originality_Checker:
    def __init__(self, csv_file=None, path_to_original_file=None, path_to_spoof_file=None):
        self.csv_file = csv_file
        self.path_to_original_file = path_to_original_file
        self.path_to_spoof_file = path_to_spoof_file


    def get_characteristics(self, path_to_audio: str):
        self.path_to_audio = path_to_audio

        # массив со значимыми для дальнейшего анализа характеристиками
        characteristics = np.array([])

        # y - одномерный массив, sr - sample rate - частота дискретизации
        y, sr = librosa.load(path_to_audio)

        # разделение сигнала на гармоническую и перкуссионную части
        y_harmonic, _ = librosa.effects.hpss(y)

        # mfccs - 15 Мел-кепстральных коэффициентов
        mfccs = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=30)
        # plt.figure(figsize=(15, 5))

        # mfccs_mean - Средние значения Мел-кепстральных коэффициентов
        mfccs_mean=np.mean(mfccs,axis=1)
        characteristics = np.append(characteristics, mfccs_mean)

        # график для mfccs_mean
        """
        sns.barplot(x=np.arange(0, 30), y=mfccs_mean)
        plt.show()
        """

        # mfccs_mean - Стандартные отклонения Мел-кепстральных коэффициентов
        mfccs_std = np.std(mfccs,axis=1)
        characteristics = np.append(characteristics, mfccs_std)

        # график для mfccs_std
        """
        sns.barplot(x=np.arange(0, 30), y=mfccs_std)
        plt.show()
        """

        # cent - Спектральный центроид
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)

        # cent_mean - среднее значение cent
        cent_mean = np.mean(cent)
        # cent_std - стандартное отклонение cent
        cent_std = np.std(cent)
        # cent_skew - наклон cent
        cent_skew = scipy.stats.skew(cent,axis=1)[0]
        characteristics = np.append(characteristics, [cent_mean, cent_std, cent_skew])

        # rolloff - спректральный спад
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        # rolloff_mean - среднее значение rolloff
        rolloff_mean = np.mean(rolloff)
        # rolloff_std - стандартное отклонение rolloff
        rolloff_std = np.std(rolloff)
        characteristics = np.append(characteristics, [rolloff_mean, rolloff_std])

        return characteristics


    def wrire_data_to_csv(self, file: str, label: str) -> None:

        # Извлечение характеристик аудиофайла
        if label == 'spoof':
            new_row = self.get_characteristics(f'{self.path_to_spoof_file}/{file}')
        elif label == 'original':   
            new_row = self.get_characteristics(f'{self.path_to_original_file}/{file}')
        
        # Добавление метки класса аудиофайла
        new_row = np.append(new_row, label)

        # Добавление извлеченных характеристик в файл-хранилище
        self.storage.append(new_row)

        # Запись харакристик, извлеченных из 1000 аудиофайлов, в csv-файл
        if len(self.storage) >= 1000:    
            self.writer.writerows(self.storage)
            self.storage.clear()


    def training(self):
        # Считывание данных из csv-файла в dataframe
        df = pd.read_csv(self.csv_file, sep=';')

        # Считывание загловков для 
        x = df[df.columns[:-1]]
        y = df[df.columns[-1]]

        # Подготовка данных для обучения 
        x1 = [list(map(float, df[i])) for i in x]
        Y = np.array([0 if i == 'spoof' else 1 for i in y])
        X = np.array([list(i) for i in list(zip(*x1))])

        # Приведение данных к стандартному виду
        X = preprocessing.StandardScaler().fit(X).transform(X).astype(np.float32)

        # Разделение выборки на тренировочную и тестовую в отношении 80/20
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=17)

        model = Sequential()
        model.add(Dense(units=35, input_shape=(X_train.shape[1],), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)))
        model.add(Dropout(0.3))
        model.add(Dense(units=1, activation='sigmoid'))

        sgd = SGD(learning_rate=0.001, momentum=0.5)
        model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
        
        model.fit(X_train, y_train, batch_size=64, epochs=100, validation_split=0.25)

        # Оценка качества модели
        score = model.evaluate(X_test, y_test)
        print(score)

        # Сохраняем модель и веса моддели в папку models
        model.save('models/keras_10_model.h5')


    def Originality_Checker(self, features, label=None, path_to_model='models/keras_10_model.h5'):
        # Приведение данных к стандартному виду
        X = np.array(features)
        X = preprocessing.StandardScaler().fit(X).transform(X).astype(np.float32)
        model = keras.models.load_model(path_to_model)
        if label is None:
            return model.predict(X)
        else:
            return model.evaluate(X, label)


    def main(self) -> None:

        # Проверка инициализации
        if any(list(map(lambda x: x is None, [self.csv_file, self.path_to_original_file, self.path_to_spoof_file]))):
            print('Неправильная инициализация экземпляра класса')
            return None

        # Создание заголовков для csv-файла
        header = [f'mfcc_mean_{i}' for i in range(1, 31)]
        header.extend([f'mfcc_std_{i}' for i in range(1, 31)])
        header.extend(['cent_mean', 'cent_std', 'cent_skew', 'rolloff_mean', 'rolloff_std', 'label'])

        # Открытие файла и запись в него заголовков
        dataset = open(file=self.csv_file, mode='w', newline='')
        self.writer = csv.writer(dataset, delimiter=';')
        self.writer.writerow(header)

        # Извлечение синтезированных и оригинальных файлов из соответствующих директорий
        _, _, files_spoof = next(os.walk(self.path_to_spoof_file))
        _, _, files_original = next(os.walk(self.path_to_original_file))

        # Добавление к списку с файлами метку
        files_spoof = [(i, 'spoof') for i in files_spoof]
        files_original = [(i, 'original') for i in files_original]

        # Объединение и перемешивание списков
        files = files_spoof + files_original
        random.shuffle(files)

        # Список для хранения характеристик 1000 файлов
        self.storage = []

        # Извлечение характеристик и запись в файл
        for file, label in files:
            self.wrire_data_to_csv(file, label)
        
        # Дозапись характеристик файлов, незаписанных в функции wrire_data_to_csv
        self.writer.writerows(self.storage)

        # Закрытие файла
        dataset.close() 
        




